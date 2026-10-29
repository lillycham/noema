"""Training data for the mixing world: a hidden first velocity and a hidden turn, rolled out, as tokens.

The tokens are those of data_torus.py: a position is d coordinate tokens and a SEP ends the step.
Nothing in them says which turn a sequence has, or that turns exist.

A sequence needs three positions to show its first two velocities (v0 = p1 - p0, v1 = p2 - p1 = R v0),
and so the first step the turn can be read from is the fourth. Of the 8 turns in 2D, a pair of
velocities leaves a few possible, and the ideal learner (below) guesses among them.

Held-out turns are whole turns: with the 8 in 2D, 2 are held out (1 in 4, at least one). A model that
has learned a table of what each turn does fails on those. A model that reads the turn from the two
velocities, and applies it, does not.
"""
import numpy as np

from noema import data_torus as dt, mixing

SIZE, DIM, STEPS = dt.SIZE, dt.DIM, dt.STEPS

class TurnSet:
    """The turns a run trains on and the ones it holds out. train() never returns a held-out turn,
    and held() returns only held-out ones. A turn is a (d, d) signed permutation matrix."""

    def __init__(self, dim=DIM, seed=0):
        self.dim, self.seed = dim, seed
        self.group = mixing.signed_permutations(dim)
        order = np.random.default_rng(seed).permutation(len(self.group))
        n_held = max(1, len(self.group) // 4)
        self.held_turns, self.train_turns = self.group[order[:n_held]], self.group[order[n_held:]]

    def is_held_out(self, turn):
        return any((np.asarray(turn) == h).all() for h in self.held_turns)

    def train(self, rng, B):
        return self.train_turns[rng.integers(0, len(self.train_turns), size=B)]

    def held(self, rng, B):
        return self.held_turns[rng.integers(0, len(self.held_turns), size=B)]

def trajectory_batch(rng, B, turns, size=SIZE, dim=DIM, steps=STEPS, advance_fn=None):
    """(tokens (B, steps * (dim + 1)) int64, turns (B, dim, dim)): B sequences, each with its own
    turn, first velocity and start. turns is an array (n, dim, dim) to choose from, or a function
    (rng, B) -> (B, dim, dim), such as TurnSet().train."""
    chosen = turns(rng, B) if callable(turns) else np.asarray(turns)[rng.integers(0, len(turns), size=B)]
    starts = rng.integers(0, size, size=(B, dim))
    v0 = rng.integers(0, size, size=(B, dim))
    return dt.encode(mixing.trajectory(starts, v0, chosen, steps, size, advance_fn)), chosen

def ideal_accuracy_by_step(positions, group, size):
    """The accuracy per coordinate of an ideal learner on steps 1 to T - 1 of positions (B, T, d): an array of T - 1.

    The ideal learner knows the world: the velocity turns by one of the matrices in `group`, all equally
    likely, and the first velocity is any. At step t it has seen the velocities v_0 .. v_(t-2). It keeps
    the turns that fit every pair so far (R v_i = v_(i+1)). For each such turn the next velocity is R v_(t-2).
    It writes the coordinates in order, as the model does, each time keeping the turns that agree with
    the coordinates already written, and picks the value most turns give (a tie is broken at random, so a
    tie counts as 1 / the number of values tied). Step 1 is a draw of the whole first velocity: 1 / size."""
    P = np.asarray(positions)
    B, T, d = P.shape
    V = (P[:, 1:] - P[:, :-1]) % size                                       # (B, T - 1, d): v_0 .. v_(T-2)
    out = [1 / size]
    for t in range(2, T):
        ok = np.ones((B, len(group)), dtype=bool)
        for i in range(t - 2):
            ok &= ((V[:, i + 1][:, None] - np.einsum("rjk,bk->brj", group, V[:, i])) % size == 0).all(-1)
        nxt = np.einsum("rjk,bk->brj", group, V[:, t - 2]) % size               # (B, |G|, d): next velocity under each turn
        true_v = V[:, t - 1]
        acc = []
        for a in range(d):
            ok_a = ok.copy()
            for b in range(a):
                ok_a &= nxt[:, :, b] == true_v[:, b][:, None]
            counts = np.stack([(ok_a & (nxt[:, :, a] == value)).sum(-1) for value in range(size)], axis=-1)   # (B, size)
            top = counts.max(-1, keepdims=True)
            tied = counts == top
            hit = tied[np.arange(B), true_v[:, a]] / tied.sum(-1)
            acc.append(hit.mean())
        out.append(float(np.mean(acc)))
    return np.array(out)
