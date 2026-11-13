"""
graph_ncas.py -- the graph NCA model, and nothing else.

There is no target, loss, starting state, figure or training loop in this
file, and no parameter value: every size, probability and temperature is an
argument that the calling script passes in. Nothing here has to be edited to
change an experiment.

Contents
    GraphData ........ any networkx graph, as tensors
    GraphNCA ......... the message-passing NCA (one step, or many)
    EdgeGate ......... G: decides which messages each cell listens to
    sample_network ... turns G's probabilities into a 0/1 mask
    random_edge_masks, edge_keep ... remove edges from outside, without G
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# The graph
# ----------------------------------------------------------------------


class GraphData:
    """A networkx graph, as the tensors the NCA works with.

    edge_feature(sender, receiver) is called once for every directed
    message and returns a list of numbers that describe it (on a lattice:
    where the sender sits relative to the receiver). Every message must get
    the same number of features.

    Attributes
      nodes          node names, sorted so the order is the same everywhere
      node_index     node name -> its position in `nodes`
      edges          undirected edges (u, v), sorted
      src, dst       per directed message: index of the sending and of the
                     receiving node
      edge_id        per directed message: index of its undirected edge
                     (u -> v and v -> u share one id)
      edge_features  (messages, features) tensor
    """

    def __init__(self, graph, edge_feature, device):
        self.graph = graph
        self.device = device

        self.nodes = sorted(graph.nodes)
        self.node_index = {node: k for k, node in enumerate(self.nodes)}
        self.edges = sorted(tuple(sorted(edge)) for edge in graph.edges)

        src, dst, edge_id, features = [], [], [], []
        for e, (u, v) in enumerate(self.edges):
            for sender, receiver in ((u, v), (v, u)):  # one message each way
                src.append(self.node_index[sender])
                dst.append(self.node_index[receiver])
                edge_id.append(e)
                features.append(edge_feature(sender, receiver))

        self.src = torch.tensor(src, device=device)
        self.dst = torch.tensor(dst, device=device)
        self.edge_id = torch.tensor(edge_id, device=device)
        self.edge_features = torch.tensor(
            features, dtype=torch.float32, device=device
        )

        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)
        self.num_messages = len(src)
        self.num_edge_features = self.edge_features.shape[1]


# ----------------------------------------------------------------------
# The NCA
# ----------------------------------------------------------------------


class GraphNCA(nn.Module):
    """A message-passing neural cellular automaton.

    At every step:
      1. every directed message (sender -> receiver) is built from the
         receiver's state, the sender's state and the message's features;
      2. every node averages the messages that are delivered to it (divides
         by the number of delivered messages);
      3. every node feeds (own state, averaged messages) to the update
         network and adds the result to its state, if the random update
         draw lets it.

    No weight depends on the graph, so one class serves every graph.

    Arguments
      num_channels       state values per node
      hidden             neurons in the hidden layers
      update_prob        chance that a node updates at a given step
      num_edge_features  numbers per message (GraphData.num_edge_features)
    """

    def __init__(self, num_channels, hidden, update_prob, num_edge_features):
        super().__init__()
        self.hidden = hidden
        self.update_prob = update_prob

        # (receiver state, sender state, edge features) -> message.
        self.message = nn.Sequential(
            nn.Linear(2 * num_channels + num_edge_features, hidden),
            nn.LeakyReLU(0.1),
        )
        # (own state, averaged messages) -> change in state.
        self.update = nn.Sequential(
            nn.Linear(num_channels + hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, num_channels),
        )
        # Start by doing nothing: with a zero last layer a fresh model
        # leaves the state unchanged.
        nn.init.zeros_(self.update[2].weight)
        nn.init.zeros_(self.update[2].bias)

    def average_messages(self, graph, messages, keep):
        """Average the messages that arrive at each node. Messages with
        keep = 0 are not delivered; a node that receives none gets zeros."""
        messages = messages * keep[..., None]
        batch = messages.shape[0]
        total = torch.zeros(
            batch, graph.num_nodes, self.hidden, device=graph.device
        )
        total = total.index_add(1, graph.dst, messages)
        count = torch.zeros(batch, graph.num_nodes, device=graph.device)
        count = count.index_add(1, graph.dst, keep)
        return total / count.clamp(min=1)[..., None]

    def perceive(self, graph, state, keep):
        """Each node's input: its own state plus the average message."""
        sender = state[:, graph.src]  # (batch, messages, channels)
        receiver = state[:, graph.dst]
        features = graph.edge_features.expand(state.shape[0], -1, -1)
        messages = self.message(torch.cat([receiver, sender, features], -1))
        return torch.cat(
            [state, self.average_messages(graph, messages, keep)], dim=-1
        )

    def step(self, graph, state, gate=None, keep=None):
        """One time step on the whole graph.

        gate  an EdgeGate: it picks, from the receiving cell's state, which
              messages are delivered. None: no gate.
        keep  without a gate, an optional fixed (batch, messages) 0/1 mask
              of the delivered messages (see edge_keep). None: all of them.

        Returns (new_state, edge_probs, messages_used, messages_possible).
        edge_probs is the probability of every message being delivered (1
        or 0 when there is no gate). A message is "used" if it is delivered
        to a node that updates in this step; "possible" is what the full
        graph would have used with the same update draw. Both counts are
        one number per sample.
        """
        assert gate is None or keep is None, "pass a gate or keep, not both"
        if gate is not None:
            edge_probs = gate(graph, state)
            keep = sample_network(edge_probs, gate.temperature)
        else:
            if keep is None:
                keep = torch.ones(
                    state.shape[0], graph.num_messages, device=graph.device
                )
            edge_probs = keep

        update = self.update(self.perceive(graph, state, keep))
        draw = torch.rand_like(state[..., :1])
        updating = (draw < self.update_prob).float()
        new_state = state + update * updating

        updating_receiver = updating[:, graph.dst, 0]  # (batch, messages)
        used = (updating_receiver * keep).sum(dim=1)
        possible = updating_receiver.sum(dim=1)
        return new_state, edge_probs, used, possible

    def rollout(self, graph, state, steps, gate=None, on_step=None):
        """Run `steps` time steps. Returns (state, used, possible): the
        message counts of step(), summed over the rollout.

        on_step(state, edge_probs, used), if given, is called after every
        step. Use it to record anything you want to see along the way."""
        used_total = torch.zeros(state.shape[0], device=graph.device)
        possible_total = torch.zeros_like(used_total)
        for _ in range(steps):
            state, edge_probs, used, possible = self.step(graph, state, gate)
            used_total = used_total + used
            possible_total = possible_total + possible
            if on_step is not None:
                on_step(state, edge_probs, used)
        return state, used_total, possible_total


# ----------------------------------------------------------------------
# G: the edge gate
# ----------------------------------------------------------------------


class EdgeGate(nn.Module):
    """G: for every directed message, the probability that the receiving
    cell listens to it.

    Input, per message   the receiving cell's current state and the
                         message's edge features.
    Output, per message  one probability between 0 and 1.

    G never sees the sender's state: that would mean the message had
    already been sent.

    One Linear layer on [state, features] equals a Linear on the state plus
    a Linear on the features. They are written separately because the state
    part is then computed once per cell and shared by all of its messages.

    Arguments
      init_bias    G starts with every message delivered with probability
                   sigmoid(init_bias), whatever the state
      temperature  how smooth the gradient through sample_network is (the
                   forward pass is always a hard 0/1 draw)
    """

    def __init__(self, num_channels, num_edge_features, hidden, init_bias,
                 temperature):
        super().__init__()
        self.temperature = temperature
        self.from_state = nn.Linear(num_channels, hidden)
        # Named from_offset so that checkpoints saved before the refactor
        # still load. It reads the edge features (the offsets, on a lattice).
        self.from_offset = nn.Linear(num_edge_features, hidden, bias=False)
        self.output = nn.Linear(hidden, 1)
        # The last layer starts out as a constant, so training has to find
        # how the probabilities should depend on the state.
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, init_bias)

    def forward(self, graph, state):
        """state (batch, nodes, channels) -> (batch, messages)."""
        from_state = self.from_state(state)[:, graph.dst]
        from_features = self.from_offset(graph.edge_features)
        hidden = F.leaky_relu(from_state + from_features, 0.1)
        return torch.sigmoid(self.output(hidden).squeeze(-1))


def sample_network(edge_probs, temperature):
    """Draw a pruned network from G's probabilities: a (batch, messages)
    mask with 1 = delivered and 0 = removed. It is the `keep` that
    GraphNCA.perceive takes, so the NCA itself needs no change.

    The forward pass is a real random draw, so the NCA always sees a hard
    0/1 network. A hard draw has no gradient, so the backward pass uses a
    smooth stand-in (the straight-through trick): G is trained as if the
    mask were a smooth function of its probabilities."""
    logits = torch.logit(edge_probs, eps=1e-6)
    uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    noise = torch.log(uniform) - torch.log1p(-uniform)  # logistic noise
    smooth = torch.sigmoid((logits + noise) / temperature)
    hard = (logits + noise > 0).float()
    # The forward value is `hard`, the gradient is smooth's. Keep the
    # brackets: (smooth - smooth.detach()) is exactly 0, so the mask stays
    # exactly 0 or 1.
    return hard + (smooth - smooth.detach())


# ----------------------------------------------------------------------
# Removing edges from outside (no G)
# ----------------------------------------------------------------------


def random_edge_masks(graph, batch, fraction, generator=None):
    """One random set of removed edges per sample, shape (batch, edges),
    1 = removed. Each sample loses exactly round(fraction * edges) edges.
    Pass a seeded `generator` to make the masks reproducible."""
    num_removed = round(fraction * graph.num_edges)
    masks = torch.zeros(batch, graph.num_edges, device=graph.device)
    if num_removed > 0:
        scores = torch.rand(
            batch, graph.num_edges, device=graph.device, generator=generator
        )
        masks.scatter_(1, scores.topk(num_removed, dim=1).indices, 1.0)
    return masks


def edge_keep(graph, removed_edges):
    """(batch, edges) removed-edge mask -> (batch, messages) `keep` mask
    for GraphNCA.step. Both messages of a removed edge are dropped."""
    return 1 - removed_edges[:, graph.edge_id]
