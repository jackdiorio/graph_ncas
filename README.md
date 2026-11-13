## Documentation

Each .py file (except for highly mutable ones) has multiline comment documentation.

## Workflow

1. Activate diorio_venv
2. Edit/run sandbox files
	- Any model checkpoints & related outputs are stored in the 'models' directory in timestamped subdirs
	- Any plots or figures are stored in the 'figures' directory in timestamped subdirs

### Legacy files

- diorio_GNCA_topologies trains models according to [Experiment 2 in Resilient GNCAs (obsidian)]
- diorio_GNCA_analysis creates figures according to [Experiment 2 in Resilient GNCAs (obsidian)]
- diorio_GNCA_plot makes a bunch of plots according to [Experiment 3 in Resilient GNCAs (obsidian)]
- old_graph_ncas.py has a bunch of boilerplate code for the model, can be called by marimo files
- diorio_GNCA_edge_weights.py is the old version of my edge weight probability experiment.