"""
diorio_GNCA_edge_gates.py -- sandbox for the graph NCA with edge gates (G).

Pick the graph, the models and every parameter below, then run

    python -u diorio_GNCA_edge_gates.py

Every model in MODELS is trained (or, with LOAD_RUN, loaded from an earlier
run) and then compared in figures and tables.

    graph_ncas.py  the model: GraphData, GraphNCA, EdgeGate
    setups.py      this project's graphs, targets and starting state
    training.py    loss, validation, training, checkpoints, figures

Output (timestamped, one folder per run)
    models/<timestamp>/    one checkpoint per model, summary.csv, config.json
    figures/<timestamp>/   figures, gifs and manifest.json
"""

import json
import time
from functools import partial
from pathlib import Path

import torch
from matplotlib.image import imread
from tqdm import tqdm

import graph_ncas as gn
import setups
import training as tr

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

# Graph and NCA
TOPOLOGY = "lattice"
GRID = 32  # nodes sit at the GRID x GRID integer positions (i, j)
NUM_CHANNELS = 8  # state values per node (the first ones are graded)
HIDDEN = 64  # neurons in the hidden layers of the NCA
UPDATE_PROB = 0.5  # chance that a node updates at a given step
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target: "smiley" is binary (graded on channel 0). "emoji" is an RGB picture
# read from EMOJI_PATH (graded on channels 0, 1 and 2), e.g. a 128 x 128
# emoji png from Growing NCAs. Transparent pixels become EMOJI_BACKGROUND.
TARGET_NAME = "smiley"
EMOJI_PATH = None  # path to the image, e.g. "emojis/lizard.png"
EMOJI_BACKGROUND = (1.0, 1.0, 1.0)  # (red, green, blue) between 0 and 1

# G, the network that decides which messages a cell listens to
GATE_HIDDEN = 16  # neurons in G's hidden layer
GATE_INIT_BIAS = 3.0  # G starts delivering sigmoid(3) = 95% of messages
GATE_TEMPERATURE = 1.0  # smoothness of G's gradient (forward is hard 0/1)

# Models to train: name -> message weight in the loss. None means no G.
MODELS = {
    "baseline (no G)": None,
    "G weight=0": 0.0,
    "G weight=0.001": 0.001,
    "G weight=0.003": 0.003,
    "G weight=0.01": 0.01,
    "G weight=0.03": 0.03,
}

TRAINING = dict(
    seed=0,
    iterations=4000,  # gradient updates per model
    batch=8,  # rollouts per iteration
    min_steps=64,  # each rollout runs between min_steps and max_steps
    max_steps=96,
    learning_rate=2e-3,  # Adam, cosine schedule down to 0
    pool_size=1024,  # the state pool starts as this many seeds
    ramp_iterations=1500,  # the message weight grows from 0 over these
)

# Validation: full graph, from the seed state, with fixed randomness
VALIDATION = dict(
    every=100,  # validate every this many training iterations
    steps=256,  # rollout length that judges (and picks) checkpoints
    batch=16,  # rollouts averaged
    seed=999,
)

FIGURES = dict(
    curve_steps=255,  # rollout length of the loss-against-messages plot
    record_every=5,  # one point every this many steps
    loss_threshold=0.05,  # table: messages needed to first get below this
    snapshot_steps=(0, 16, 32, 64, 128, 255),
    video_steps=128,  # length of the edge-probability gif
    video_every=2,  # one frame every this many steps
    video_fps=10,
    dpi=200,
)

LOAD_RUN = None  # a folder name inside models/ to load, or None to train

# Paths: models/<timestamp>/ and figures/<timestamp>/ next to this script
ROOT = Path(__file__).resolve().parent
MODEL_RUN = LOAD_RUN or time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "models" / MODEL_RUN
FIGURES_DIR = ROOT / "figures" / MODEL_RUN

# Everything above, saved with every checkpoint (numbers and text only)
CONFIG = {
    name: value for name, value in globals().items()
    if name.isupper() and isinstance(
        value, (int, float, str, tuple, dict, type(None)))
}

# ----------------------------------------------------------------------
# The problem: graph, target and starting state
# ----------------------------------------------------------------------

GRAPH = gn.GraphData(
    setups.make_lattice(GRID), setups.grid_offset(GRID), DEVICE
)
if TARGET_NAME == "smiley":
    TARGET = setups.make_binary_target(GRAPH, GRID)
elif TARGET_NAME == "emoji":
    # imread gives a float array (height, width, 3 or 4) for a png and
    # integers for a jpg; make_rgb_target understands both.
    TARGET = setups.make_rgb_target(
        GRAPH, GRID, imread(EMOJI_PATH), EMOJI_BACKGROUND
    )
else:
    raise ValueError(f"TARGET_NAME must be 'smiley' or 'emoji', "
                     f"not {TARGET_NAME!r}")

# The target's channels are graded, so the state needs at least that many.
if NUM_CHANNELS < TARGET.shape[-1]:
    raise ValueError(f"NUM_CHANNELS = {NUM_CHANNELS}, but the {TARGET_NAME} "
                     f"target has {TARGET.shape[-1]} channels")

SEED = partial(setups.make_seed, GRAPH, GRID, NUM_CHANNELS)
PROBLEM = tr.Problem(GRAPH, GRID, TARGET, SEED)

print(f"device: {DEVICE}")
print(f"models go to {RUN_DIR}, figures to {FIGURES_DIR}")
print(f"{TOPOLOGY}: {GRAPH.num_nodes} cells, {GRAPH.num_edges} edges, "
      f"{GRAPH.num_messages} messages", flush=True)


def build_model(message_weight):
    """A fresh NCA, and a fresh G unless message_weight is None."""
    model = gn.GraphNCA(
        NUM_CHANNELS, HIDDEN, UPDATE_PROB, GRAPH.num_edge_features
    ).to(DEVICE)
    gate = None
    if message_weight is not None:
        gate = gn.EdgeGate(
            NUM_CHANNELS, GRAPH.num_edge_features, GATE_HIDDEN,
            GATE_INIT_BIAS, GATE_TEMPERATURE,
        ).to(DEVICE)
    return model, gate


# ----------------------------------------------------------------------
# Train (or load) every model
# ----------------------------------------------------------------------

if LOAD_RUN is None:
    RUN_DIR.mkdir(parents=True)
    (RUN_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

RUNS = []
for name, weight in MODELS.items():
    path = tr.checkpoint_path(RUN_DIR, tr.model_label(TOPOLOGY, weight))
    tr.seed_everything(TRAINING["seed"])
    model, gate = build_model(weight)
    if LOAD_RUN is None:
        print(f"\nTraining: {name}", flush=True)
        progress = partial(tqdm, desc=name, mininterval=60)
        run = tr.train_model(name, PROBLEM, model, gate, weight, path,
                             CONFIG, TRAINING, VALIDATION, progress)
    else:
        run = tr.load_run(name, PROBLEM, model, gate, path, CONFIG)
    RUNS.append(run)

if LOAD_RUN is None:
    tr.write_run_summary(RUNS, RUN_DIR, VALIDATION["steps"])
tr.print_table(tr.summary_rows(RUNS, VALIDATION["steps"]),
               "Best checkpoint of each model")

# ----------------------------------------------------------------------
# Output: figures and tables
# ----------------------------------------------------------------------

save = partial(tr.save_figure, model_run=MODEL_RUN,
               figures_dir=FIGURES_DIR, dpi=FIGURES["dpi"])

save(tr.plot_validation_curves(RUNS, VALIDATION["steps"]),
     f"validation_curves_{VALIDATION['steps']}_steps")

CURVES = tr.message_curves(PROBLEM, RUNS, VALIDATION,
                           FIGURES["curve_steps"], FIGURES["record_every"])
save(tr.plot_loss_vs_messages(CURVES, FIGURES["curve_steps"],
                              FIGURES["record_every"]), "loss_vs_messages")
tr.print_table(tr.curve_rows(CURVES, FIGURES["loss_threshold"]),
               "Loss against messages passed")

frame_steps = max(max(FIGURES["snapshot_steps"]), FIGURES["video_steps"]) + 1
for run in RUNS:
    frames = tr.record_frames(PROBLEM, run, VALIDATION, frame_steps)
    save(tr.plot_snapshots(PROBLEM, run, frames, FIGURES["snapshot_steps"]),
         f"snapshots_{run.label}")
    if run.gate is not None:  # without G there are no probabilities to show
        tr.save_edge_gif(PROBLEM, run, frames, FIGURES_DIR, MODEL_RUN,
                         FIGURES["video_steps"], FIGURES["video_every"],
                         FIGURES["video_fps"])

print(f"\nDone. Models: {RUN_DIR}   Figures: {FIGURES_DIR}")
