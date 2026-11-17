"""
training.py -- everything around the model: loss, validation, training,
checkpoints, and the figures.

Like graph_ncas.py this file sets no parameter values. The sandbox script
passes in every setting, the training ones as the dictionaries `training`
and `validation`:

    training    seed, iterations, batch, min_steps, max_steps,
                learning_rate, pool_size, ramp_iterations
    validation  every, steps, batch, seed

Contents
    1. Problem, loss and validation
    2. Runs and checkpoints
    3. Training
    4. Figures and tables
"""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

import setups

plt.switch_backend("Agg")  # no display needed: figures are only saved


# ======================================================================
# 1. Problem, loss and validation
# ======================================================================


@dataclass
class Problem:
    """What the model is asked to do.

    graph      a graph_ncas.GraphData
    grid       side length of the grid the nodes sit on (for images)
    target     the values each node should output, shape (nodes, channels):
               one column for the binary smiley, three for an RGB picture.
               The model is graded on the first `channels` state channels.
    make_seed  make_seed(batch) -> the starting states
    """

    graph: object
    grid: int
    target: torch.Tensor
    make_seed: object


def seed_everything(seed):
    """Seed every random number generator the training uses."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def per_sample_loss(state, target):
    """Squared error of the output channels (the first target.shape[-1]
    channels of the state) against the target, averaged over the nodes and
    the channels: one number per sample. For a one-channel target this is
    the error of channel 0 alone."""
    graded = target.shape[-1]
    return ((state[..., :graded] - target) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def validation_rollout(problem, model, gate, steps, batch, seed,
                       on_step=None):
    """Roll out from the seed state with fixed randomness, so that every
    model (and every checkpoint) is judged on the same draws. The global
    random state is restored afterwards, so validating never disturbs
    training. Returns what GraphNCA.rollout returns."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        state = problem.make_seed(batch)
        return model.rollout(problem.graph, state, steps, gate, on_step)


def validation_scores(problem, model, gate, validation):
    """(loss, message share) after validation["steps"] steps, averaged over
    validation["batch"] rollouts. The share is messages used / possible."""
    state, used, possible = validation_rollout(
        problem, model, gate, validation["steps"], validation["batch"],
        validation["seed"],
    )
    loss = per_sample_loss(state, problem.target).mean().item()
    share = (used / possible.clamp(min=1)).mean().item()
    return loss, share


# ======================================================================
# 2. Runs and checkpoints
#    Names such as "_0pct_removed" and "0.00_removed_256_steps" come from
#    the experiments with random edge removal. They are kept so that every
#    checkpoint saved so far still loads.
# ======================================================================


@dataclass
class Run:
    """One trained (or loaded) model, with its best-checkpoint scores."""

    name: str
    model: nn.Module
    gate: object  # an EdgeGate, or None
    message_weight: object  # a number, or None for a model without G
    label: str
    history: dict
    best_iteration: int
    best_loss: float
    best_share: float


def model_label(topology, message_weight):
    """File-name part of a model: 'lattice' without G, and for a model
    with G and message weight 0.1 'lattice_gate0p1'."""
    if message_weight is None:
        return topology
    return f"{topology}_gate{message_weight:g}".replace(".", "p")


def checkpoint_path(run_dir, label):
    """The checkpoint file of one model inside models/<timestamp>/."""
    return Path(run_dir) / f"{label}_0pct_removed.pt"


def validation_key(steps):
    """Name under which the validation losses are stored in the history."""
    return f"0.00_removed_{steps}_steps"


def validation_losses(history):
    """The validation loss curve stored in a run's history."""
    return next(iter(history["validation"].values()))


def selection_rule(message_weight, steps):
    """Words describing how the best checkpoint of a run is chosen."""
    if message_weight is None:
        return f"lowest validation loss at {steps} steps"
    return (f"lowest validation loss + {message_weight:g} x message share "
            f"at {steps} steps")


def copy_weights(module):
    """A detached CPU copy of a module's weights (None for no module)."""
    if module is None:
        return None
    return {key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()}


def save_checkpoint(path, graph, config, best, history, message_weight,
                    steps):
    """Save the best weights of a run with everything needed to use them
    later: the graph, all settings and the validation history."""
    torch.save(
        {
            "model_state": best["model_state"],
            "gate_state": best["gate_state"],  # None without G
            "topology": config["TOPOLOGY"],
            "removal_fraction": 0.0,
            "best_iteration": best["iteration"],
            "best_validation_loss": best["loss"],
            "best_message_share": best["share"],
            "best_objective": best["objective"],
            "message_weight": message_weight,
            "selection_rule": selection_rule(message_weight, steps),
            "history": history,
            "graph_nodes": graph.nodes,
            "graph_edges": graph.edges,
            "config": config | {"MESSAGE_WEIGHT": message_weight},
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch_version": str(torch.__version__),
            "networkx_version": str(nx.__version__),
        },
        path,
    )


def load_run(name, problem, model, gate, path, config):
    """Load a saved checkpoint into a freshly built `model` and `gate`
    (None if there is no G) and return it as a Run. The sizes saved with
    the checkpoint must match the current `config`, and the graph must be
    the one the model was trained on."""
    info = torch.load(path, map_location="cpu", weights_only=True)

    for key in ("GRID", "NUM_CHANNELS", "HIDDEN", "UPDATE_PROB"):
        if info["config"][key] != config[key]:
            raise ValueError(
                f"{path.name}: trained with {key}={info['config'][key]}, "
                f"but this script uses {key}={config[key]}"
            )
    saved_edges = [tuple(edge) for edge in info["graph_edges"]]
    if saved_edges != problem.graph.edges:
        raise ValueError(f"{path.name}: trained on a different graph")
    if (gate is None) != (info.get("gate_state") is None):
        raise ValueError(f"{path.name}: G is on in only one of the "
                         "checkpoint and this script")

    model.load_state_dict(info["model_state"])
    if gate is not None:
        gate.load_state_dict(info["gate_state"])

    weight = info.get("message_weight")
    history = info["history"]
    # A checkpoint without G delivers every message.
    history.setdefault("message_share", [1.0] * len(history["iteration"]))
    return Run(
        name, model, gate, weight, model_label(info["topology"], weight),
        history, info["best_iteration"], info["best_validation_loss"],
        info.get("best_message_share", 1.0),
    )


def write_run_summary(runs, run_dir, steps):
    """models/<timestamp>/summary.csv: one row per run."""
    with open(Path(run_dir) / "summary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "topology", "trained_with_fraction_removed", "best_iteration",
            f"best_validation_loss_at_{steps}_steps", "file",
            "message_weight", "best_message_share",
        ])
        for run in runs:
            weight = "" if run.message_weight is None else run.message_weight
            writer.writerow([
                run.label, 0.0, run.best_iteration, f"{run.best_loss:.6f}",
                checkpoint_path(run_dir, run.label).name, weight,
                f"{run.best_share:.6f}",
            ])


# ======================================================================
# 3. Training
# ======================================================================


def train_model(name, problem, model, gate, message_weight, path, config,
                training, validation, progress=None):
    """Train `model` (and `gate`, if there is one) and return a Run.

    loss = loss of the picture after the last step
           + message_weight * ramp * (messages used / messages possible)

    The ramp grows from 0 to 1 over training["ramp_iterations"] iterations,
    so that the picture is learned first. A model without G has no message
    term. Call seed_everything and build the model just before this.

    The best checkpoint is saved to `path` whenever a validation beats the
    best so far, and the best model (not the last one) is returned. The
    best model has the lowest validation value of the quantity that the
    model is trained on: loss + message_weight x share, which is the
    validation loss alone for a model without G.
    """
    graph, target = problem.graph, problem.target
    steps = validation["steps"]
    weight = message_weight or 0.0

    parameters = list(model.parameters())
    if gate is not None:
        parameters += list(gate.parameters())  # one optimiser for both
    optimizer = torch.optim.Adam(parameters, lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, training["iterations"]
    )
    pool = problem.make_seed(training["pool_size"])

    history = {
        "iteration": [],
        "validation": {validation_key(steps): []},
        "message_share": [],
        "objective": [],
    }
    best = {"objective": float("inf"), "model_state": None,
            "gate_state": None, "iteration": -1, "loss": None,
            "share": None}
    skipped = 0  # batches dropped because of non-finite numbers

    iterations = range(training["iterations"])
    if progress is not None:
        iterations = progress(iterations)

    for iteration in iterations:
        # 1. A batch from the pool, its worst state swapped for a seed.
        pool_idx = torch.randperm(
            training["pool_size"], device=graph.device
        )[:training["batch"]]
        state = pool[pool_idx].clone()
        worst = per_sample_loss(state, target).argmax()
        state[worst] = problem.make_seed(1)[0]

        # 2. Run the NCA (and G) for a random number of steps.
        num_steps = int(np.random.randint(
            training["min_steps"], training["max_steps"] + 1
        ))
        state, used, possible = model.rollout(graph, state, num_steps, gate)

        # 3. The loss: how wrong the picture is, plus how many messages it
        #    took (switched on gradually).
        ramp = 1.0
        if training["ramp_iterations"] > 0:
            ramp = min(1.0, iteration / training["ramp_iterations"])
        share = used / possible.clamp(min=1)
        loss = (per_sample_loss(state, target)
                + weight * ramp * share).mean()

        # A batch with inf or NaN must not be allowed to reach the weights:
        # dividing an infinite gradient by its norm gives NaN, and NaN
        # weights stay NaN. Drop the batch and put fresh seeds in the pool.
        if not torch.isfinite(loss):
            skipped += 1
            print(f"  iteration {iteration + 1}: loss is not finite "
                  f"(largest |state| {state.abs().max().item():.3g}), "
                  "batch skipped", flush=True)
            pool[pool_idx] = problem.make_seed(len(pool_idx))
            scheduler.step()
            continue

        # 4. Update (each parameter's gradient is normalised to length 1).
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in parameters if p.grad is not None]
        if not all(torch.isfinite(grad).all() for grad in grads):
            skipped += 1
            print(f"  iteration {iteration + 1}: a gradient is not finite, "
                  "update skipped", flush=True)
            pool[pool_idx] = problem.make_seed(len(pool_idx))
            scheduler.step()
            continue
        for grad in grads:
            grad /= grad.norm() + 1e-8
        optimizer.step()
        scheduler.step()

        # 5. The outputs go back into the pool.
        pool[pool_idx] = state.detach()

        # 6. Validate, and keep the best checkpoint.
        if (iteration + 1) % validation["every"] != 0:
            continue
        val_loss, val_share = validation_scores(
            problem, model, gate, validation
        )
        objective = val_loss + weight * val_share
        if not np.isfinite(objective):
            print(f"  iteration {iteration + 1}: validation is not finite, "
                  f"stopping this model early ({skipped} batches skipped)",
                  flush=True)
            break
        history["iteration"].append(iteration + 1)
        validation_losses(history).append(val_loss)
        history["message_share"].append(val_share)
        history["objective"].append(objective)
        print(f"  iteration {iteration + 1}: validation loss "
              f"{val_loss:.4f}, message share {val_share:.3f}", flush=True)
        if objective < best["objective"]:
            best = {"objective": objective,
                    "model_state": copy_weights(model),
                    "gate_state": copy_weights(gate),
                    "iteration": iteration + 1, "loss": val_loss,
                    "share": val_share}
            save_checkpoint(path, graph, config, best, history,
                            message_weight, steps)

    if skipped:
        print(f"  {skipped} batches were skipped (non-finite values)",
              flush=True)

    if best["model_state"] is not None:  # return the best, not the last
        model.load_state_dict(best["model_state"])
        if gate is not None:
            gate.load_state_dict(best["gate_state"])
        save_checkpoint(path, graph, config, best, history, message_weight,
                        steps)  # again, now with the full history
    return Run(
        name, model, gate, message_weight,
        model_label(config["TOPOLOGY"], message_weight), history,
        best["iteration"], best["loss"], best["share"],
    )


# ======================================================================
# 4. Figures and tables
# ======================================================================


def print_table(rows, title=None):
    """Print a list of dictionaries as an aligned text table."""
    if title:
        print(f"\n{title}")
    columns = list(rows[0])
    widths = {c: max(len(str(c)), *(len(str(row[c])) for row in rows))
              for c in columns}
    print("  ".join(str(c).ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in columns))


def add_to_manifest(figures_dir, filename, model_run):
    """Record in figures/<timestamp>/manifest.json which model run a file
    was made from."""
    manifest_path = Path(figures_dir) / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest[filename] = model_run
    manifest_path.write_text(json.dumps(manifest, indent=2))


def save_figure(fig, name, model_run, figures_dir, dpi):
    """Save `fig` as <figures_dir>/<name>.png and close it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    add_to_manifest(figures_dir, f"{name}.png", model_run)


def run_label(run):
    """Legend text of a run."""
    if run.message_weight is None:
        return run.name
    return f"{run.name} (message weight {run.message_weight})"


# --- Validation during training ---------------------------------------


def plot_validation_curves(runs, steps):
    """Validation loss and message share against training iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for k, run in enumerate(runs):
        colour = f"C{k}"
        history = run.history
        axes[0].plot(history["iteration"], validation_losses(history),
                     colour, label=run_label(run))
        axes[1].plot(history["iteration"], history["message_share"], colour)
        # A star marks the saved checkpoint.
        axes[0].plot(run.best_iteration, run.best_loss, "*", c=colour,
                     ms=12, mec="white")
        axes[1].plot(run.best_iteration, run.best_share, "*", c=colour,
                     ms=12, mec="white")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"validation loss after {steps} steps")
    axes[1].set_ylabel("messages used / messages possible")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("training iteration")
    axes[0].legend(frameon=False)
    fig.suptitle("Validation during training (star = saved checkpoint)")
    fig.tight_layout()
    return fig


def summary_rows(runs, steps):
    """One table row per run: its best checkpoint."""
    return [
        {
            "model": run.name,
            "message weight": ("-" if run.message_weight is None
                               else run.message_weight),
            "best iteration": run.best_iteration,
            f"validation loss @ {steps} steps": round(run.best_loss, 4),
            "messages used (share of possible)": f"{run.best_share:.0%}",
        }
        for run in runs
    ]


# --- Loss against messages passed -------------------------------------


def message_curves(problem, runs, validation, curve_steps, record_every):
    """For every run: the loss after `curve_steps` validation steps against
    the messages used so far, one point every `record_every` steps. Returns
    {run name: (messages used, loss)}, both averaged over the batch."""
    curves = {}
    for run in runs:
        losses, useds = [], []

        def record(state, edge_probs, used):
            losses.append(per_sample_loss(state, problem.target).cpu())
            useds.append(used.cpu())

        validation_rollout(problem, run.model, run.gate, curve_steps,
                           validation["batch"], validation["seed"], record)
        messages = torch.stack(useds).cumsum(dim=0).mean(dim=1)
        loss = torch.stack(losses).mean(dim=1)
        points = slice(record_every - 1, None, record_every)
        curves[run.name] = (messages[points].numpy(), loss[points].numpy())
    return curves


def plot_loss_vs_messages(curves, curve_steps, record_every):
    """Loss against the messages used, one line per run."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (name, (messages, losses)) in enumerate(curves.items()):
        ax.plot(messages, losses, "o-", ms=4, c=f"C{k}", label=name)
    ax.set_xlabel("messages used (received by a cell that updated)")
    ax.set_ylabel("validation loss")
    ax.set_title(f"One point every {record_every} steps, "
                 f"up to {curve_steps} steps")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def curve_rows(curves, loss_threshold):
    """One table row per run: the messages needed to first reach
    `loss_threshold`, and where the curve ends."""
    rows = []
    for name, (messages, losses) in curves.items():
        reached = np.flatnonzero(losses <= loss_threshold)
        first = (f"{messages[reached[0]]:,.0f}" if len(reached)
                 else "not reached")
        rows.append({
            "model": name,
            f"messages until loss <= {loss_threshold}": first,
            "loss at the last point": round(float(losses[-1]), 4),
            "total messages at the last point": f"{messages[-1]:,.0f}",
        })
    return rows


# --- Drawing the network ----------------------------------------------


def output_image(problem, node_values):
    """Per-node values, shape (nodes, channels) -> an image to show: a
    grid x grid greyscale image for one channel, grid x grid x 3 colours for
    three. Values are clipped to 0 to 1, because the output channels are
    free to go outside that range and imshow cannot draw such colours."""
    values = torch.as_tensor(node_values)[None].to(problem.graph.device)
    image = setups.to_image(problem.graph, problem.grid, values)[0].cpu()
    if image.shape[-1] == 1:  # one channel: imshow wants a 2-D array
        image = image[..., 0]
    return image.clamp(0, 1)


def edge_colormap():
    """White to dark blue: an edge that is almost surely removed fades
    into the page."""
    colours = plt.cm.Blues(np.linspace(0.08, 1.0, 256))
    return LinearSegmentedColormap.from_list("listening", colours)


def edge_segments(graph, grid):
    """The line segments that draw each undirected edge, as (x = column,
    y = row) pairs. An edge that wraps around the border is drawn as two
    stubs that leave the picture on one side and enter on the other."""
    segments = []
    for u, v in graph.edges:
        di = setups.wrap(v[0] - u[0], grid)
        dj = setups.wrap(v[1] - u[1], grid)
        if (u[0] + di, u[1] + dj) == v:
            segments.append([[(u[1], u[0]), (v[1], v[0])]])
        else:
            segments.append([
                [(u[1], u[0]), (u[1] + dj, u[0] + di)],
                [(v[1], v[0]), (v[1] - dj, v[0] - di)],
            ])
    return segments


def half_edge_segments(graph, grid):
    """Every edge is drawn as two halves, one per direction, because G
    decides per message: the half next to a cell is coloured by the
    probability that THAT cell listens along the edge. Returns
    (segments, message_of_segment): the line segments, and for each one the
    index of the message whose probability colours it.

    GraphData stores the messages of edge e as 2e (u -> v, received by v)
    and 2e + 1 (v -> u, received by u)."""
    receivers = torch.tensor(
        [graph.node_index[v] for _, v in graph.edges], device=graph.device
    )
    assert torch.equal(graph.dst[0::2], receivers), "unexpected order"

    segments, message_of_segment = [], []
    for e, lines in enumerate(edge_segments(graph, grid)):
        into_v, into_u = 2 * e, 2 * e + 1
        if len(lines) == 1:  # an ordinary edge: cut it in the middle
            start, end = lines[0]  # start is at u, end is at v
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            segments += [[start, middle], [middle, end]]
        else:  # a wrap-around edge: one stub leaves each end
            segments += [lines[0], lines[1]]
        message_of_segment += [into_u, into_v]
    return segments, np.array(message_of_segment)


def draw_network(ax, problem, edge_probs):
    """Draw the graph on `ax`, every half-edge coloured by its probability
    (0 to 1; one number per message). Returns (line collection,
    message_of_segment) so an animation can recolour the same lines."""
    graph = problem.graph
    segments, message_of_segment = half_edge_segments(graph, problem.grid)
    lines = LineCollection(segments, cmap=edge_colormap(),
                           norm=Normalize(vmin=0, vmax=1), linewidths=0.9)
    lines.set_array(np.asarray(edge_probs)[message_of_segment])
    ax.add_collection(lines)
    rows, cols = setups.node_positions(graph)
    ax.scatter(cols.cpu(), rows.cpu(), s=2, c="0.35", zorder=3)  # the cells
    ax.set_xlim(-0.5, problem.grid - 0.5)
    ax.set_ylim(problem.grid - 0.5, -0.5)  # row 0 at the top, as an image
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return lines, message_of_segment


# --- One recorded rollout: snapshots and the video ---------------------


def record_frames(problem, run, validation, steps):
    """One validation rollout of a single sample, recorded step by step.
    Returns {"probs": (steps, messages),
             "outputs": (steps + 1, nodes, channels)}.
    probs[t] is what G produced from the state shown in outputs[t]. Only
    the graded channels are recorded: as many as the target has."""
    graded = problem.target.shape[-1]
    probs = []
    outputs = [problem.make_seed(1)[0, :, :graded].cpu()]  # starting output

    def record(state, edge_probs, used):
        probs.append(edge_probs[0].cpu())
        outputs.append(state[0, :, :graded].cpu())

    validation_rollout(problem, run.model, run.gate, steps, 1,
                       validation["seed"], record)
    return {"probs": torch.stack(probs).numpy(),
            "outputs": torch.stack(outputs).numpy()}


def plot_snapshots(problem, run, frames, snapshot_steps):
    """The output channel at a few steps and, for a model with G, the edge
    probabilities below it."""
    has_gate = run.gate is not None
    rows = 2 if has_gate else 1
    columns = 1 + len(snapshot_steps)
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.6 * columns, 2.9 * rows), squeeze=False
    )
    target = problem.target.cpu().numpy()
    axes[0, 0].imshow(output_image(problem, target), vmin=0, vmax=1)
    axes[0, 0].set_title("target")
    lines = None
    for col, step in enumerate(snapshot_steps, start=1):
        output = frames["outputs"][step]
        loss = float(((output - target) ** 2).mean())
        axes[0, col].imshow(output_image(problem, output), vmin=0, vmax=1)
        axes[0, col].set_title(f"step {step}\nloss {loss:.4f}", fontsize=9)
        if has_gate:
            probs = frames["probs"][step]
            lines, _ = draw_network(axes[1, col], problem, probs)
            axes[1, col].set_title(f"mean probability {probs.mean():.2f}",
                                   fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    if has_gate:
        axes[1, 0].axis("off")
        fig.colorbar(lines, ax=axes[1, :].tolist(), fraction=0.02,
                     pad=0.01, label="probability that the cell listens")
    title = f"{run.name}: output channel"
    fig.suptitle(title + (" and edge probabilities" if has_gate else ""))
    return fig


def save_edge_gif(problem, run, frames, figures_dir, model_run, steps,
                  every, fps):
    """Save a gif of the first `steps` steps: the output channel on the
    left and the edge probabilities on the right, at the same time step.
    One frame every `every` steps."""
    probs = frames["probs"][:steps]
    outputs = frames["outputs"][:steps]
    target = problem.target.cpu().numpy()

    fig, (ax_output, ax_network) = plt.subplots(1, 2, figsize=(11, 5.4))
    image = ax_output.imshow(output_image(problem, outputs[0]),
                             vmin=0, vmax=1)
    ax_output.set_title("output channel")
    ax_output.set_xticks([])
    ax_output.set_yticks([])
    lines, message_of_segment = draw_network(ax_network, problem, probs[0])
    fig.colorbar(lines, ax=ax_network, fraction=0.046, pad=0.02,
                 label="probability that the cell listens along the edge")
    heading = fig.suptitle("")

    def draw_frame(t):
        loss = ((outputs[t] - target) ** 2).mean()
        lines.set_array(probs[t][message_of_segment])
        image.set_data(output_image(problem, outputs[t]))
        heading.set_text(f"{run.name}   step {t}   loss {loss:.4f}   "
                         f"mean edge probability {probs[t].mean():.2f}")
        return []

    movie = animation.FuncAnimation(fig, draw_frame,
                                    frames=range(0, steps, every),
                                    blit=False)
    filename = f"edge_probabilities_{run.label}.gif"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    movie.save(Path(figures_dir) / filename,
               writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    add_to_manifest(figures_dir, filename, model_run)
