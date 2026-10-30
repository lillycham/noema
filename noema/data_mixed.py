"""Training data for rung 4: the mixing world in several dimensions at once, and one held out.

A sequence is a particle in d dimensions (see mixing.py). Its tokens are those of data_torus.py: d
coordinate tokens and a SEP for each step. No token says what d is. The model sees it from how far
apart the first two SEPs are. A batch has one d, so every sequence in it has the same length.

A run trains on some dimensions (1, 2 and 3) and tests on one it never saw (4). The d = 4 world has
384 turns, and the model has seen none of them, and no sequence with four coordinates in a record.
The question is whether what it learned about axes carries over to a fourth.

Every turn of the training dimensions is used in training: the held-out part here is a dimension, not
turns.
"""
import numpy as np

from noema import data_mixing as dm, data_torus as dt, mixing

SIZE, STEPS = dt.SIZE, dt.STEPS

def batch(rng, B, dim, size=SIZE, steps=STEPS, advance_fn=None):
    """tokens (B, steps * (dim + 1)): B sequences in `dim` dimensions, each with a turn drawn from all of them."""
    return dm.trajectory_batch(rng, B, mixing.signed_permutations(dim), size, dim, steps, advance_fn)[0]

def train_batch(rng, B, dims, size=SIZE, steps=STEPS, advance_fn=None):
    """(tokens, dim): a batch in a dimension drawn at random from dims."""
    dim = int(dims[rng.integers(0, len(dims))])
    return batch(rng, B, dim, size, steps, advance_fn), dim

def ideal(X, dim, size=SIZE):
    """The ideal learner's accuracy by step on a batch X of tokens in `dim` dimensions (see data_mixing)."""
    return dm.ideal_accuracy_by_step(dt.decode(X, dim), mixing.signed_permutations(dim), size)
