"""
setups.py -- the problem-specific parts of the experiments.

graph_ncas.py works for any graph and any target. This file is where this
project's choices live: the graphs (a torus lattice, a spanning tree, a
small-world graph), the smiley target, the starting state and the offset
that describes each message. Like graph_ncas.py it sets no parameter
values: grid size, seeds and so on are arguments.

Every graph has the same nodes, named by their (i, j) position on a grid,
so every graph is graded against the same smiley.
"""

import math
import random

import networkx as nx
import torch


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def wrap(distance, grid):
    """Shortest signed distance between two coordinates on a circle of
    `grid` positions. For grid = 32: wrap(1) = 1, wrap(31) = -1,
    wrap(-31) = 1, wrap(16) = -16."""
    return (distance + grid // 2) % grid - grid // 2


def signed_log(distance):
    """Squash a coordinate difference so that long edges do not give
    inputs far bigger than short ones: +-1 -> +-0.69, +-4 -> +-1.61,
    +-16 -> +-2.83. Keeps the sign and the order, maps 0 to 0."""
    return math.copysign(math.log1p(abs(distance)), distance)


def grid_offset(grid):
    """The `edge_feature` function that GraphData takes: where the sender
    sits relative to the receiver, going the short way round the torus.
    It depends only on the difference of the two positions, so it does not
    change if the whole picture is shifted."""

    def offset(sender, receiver):
        return [
            signed_log(wrap(sender[0] - receiver[0], grid)),
            signed_log(wrap(sender[1] - receiver[1], grid)),
        ]

    return offset


# ----------------------------------------------------------------------
# Graphs: networkx graphs whose nodes are (i, j) tuples
# ----------------------------------------------------------------------


def make_lattice(grid):
    """A torus: every node is linked to its 8 neighbours (up, down, left,
    right and the 4 diagonals) and the grid wraps around."""
    graph = nx.Graph()
    for i in range(grid):
        for j in range(grid):
            graph.add_node((i, j))
    for i in range(grid):
        for j in range(grid):
            # Look "forward" only, so each edge is added exactly once:
            # right, down, down-right, down-left.
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                graph.add_edge((i, j), ((i + di) % grid, (j + dj) % grid))
    return graph


def make_minimum_spanning_tree(grid, seed):
    """A random spanning tree of the lattice: every node is reached from
    every other by exactly one path (grid * grid - 1 edges). Each lattice
    edge gets a random weight and the minimum spanning tree is kept."""
    rng = random.Random(seed)
    lattice = make_lattice(grid)
    for u, v in sorted(lattice.edges):
        lattice.edges[u, v]["weight"] = rng.random()
    return nx.minimum_spanning_tree(lattice, weight="weight")


def make_small_world(grid, rewire_prob, seed):
    """The lattice, except that each edge is (with probability
    `rewire_prob`) rewired: one end moves to a random node. This gives
    long-range shortcuts and keeps the number of edges."""
    rng = random.Random(seed)
    graph = make_lattice(grid)
    all_nodes = sorted(graph.nodes)
    for u, v in sorted(graph.edges):  # sorted() copies, so editing is safe
        if rng.random() < rewire_prob:
            new_end = rng.choice(all_nodes)
            if new_end != u and not graph.has_edge(u, new_end):
                graph.remove_edge(u, v)
                graph.add_edge(u, new_end)
    return graph


# ----------------------------------------------------------------------
# Target, starting state and images
# ----------------------------------------------------------------------


def node_positions(graph):
    """Row and column of every node, as two index tensors (graph.nodes
    are (i, j) tuples)."""
    rows = torch.tensor([i for i, _ in graph.nodes], device=graph.device)
    cols = torch.tensor([j for _, j in graph.nodes], device=graph.device)
    return rows, cols


def make_smiley(grid, device):
    """A grid x grid image of the smiley face (1 = face, 0 = background)."""
    y, x = torch.meshgrid(
        torch.arange(grid), torch.arange(grid), indexing="ij"
    )
    centre = (grid - 1) / 2
    radius = torch.sqrt((y - centre) ** 2 + (x - centre) ** 2)

    disc = radius < 11
    left_eye = (y - (centre - 3)) ** 2 + (x - (centre - 4)) ** 2 < 3.5
    right_eye = (y - (centre - 3)) ** 2 + (x - (centre + 4)) ** 2 < 3.5
    mouth = (radius > 5) & (radius < 7.5) & (y > centre + 1)

    face = disc & ~left_eye & ~right_eye & ~mouth
    return face.float().to(device)


def make_target(graph, grid):
    """The smiley value each node should output: one number per node."""
    rows, cols = node_positions(graph)
    return make_smiley(grid, graph.device)[rows, cols]


def make_seed(graph, grid, num_channels, batch):
    """The starting state: every channel of the centre node is 1 and
    everything else is 0. Shape (batch, nodes, channels)."""
    state = torch.zeros(batch, graph.num_nodes, num_channels,
                        device=graph.device)
    centre = (grid // 2, grid // 2)
    state[:, graph.node_index[centre], :] = 1.0
    return state


def to_image(graph, grid, values):
    """(batch, nodes) per-node values -> (batch, grid, grid) images."""
    rows, cols = node_positions(graph)
    image = torch.zeros(values.shape[0], grid, grid, device=values.device)
    image[:, rows, cols] = values
    return image
