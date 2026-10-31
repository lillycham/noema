"""A world where the stride does not name the dimension.

In the mixing world a record is d coordinates and a SEP, so the stride is d + 1, and every dimension
has just one stride. A model can then get by with a fixed distance for each stride, and rung 4 found
that it does. Here a record has FILLER tokens between the coordinates and the SEP, so one dimension
comes in many strides. A run holds out a stride for every dimension, and separately holds out a
dimension, and the two questions come apart: can the model read a record length it never saw, in a
dimension it knows? Can it read a new dimension at a record length it knows?

The filler carries nothing and the loss does not count it, so the ideal learner of a padded sequence
is that of the plain one.
"""
import numpy as np

from noema import data_mixed as dx, data_torus as dt

SIZE, STEPS = dt.SIZE, dt.STEPS

def filler(size=SIZE):
    """The token of a filler, one above SEP."""
    return size + 1

def pad(X, dim, stride, size=SIZE):
    """tokens (B, steps * (dim + 1)) -> (B, steps * stride), with stride - dim - 1 fillers before each SEP."""
    B, L = X.shape
    steps = L // (dim + 1)
    records = X.reshape(B, steps, dim + 1)
    fill = np.full((B, steps, stride - dim - 1), filler(size), dtype=X.dtype)
    return np.concatenate([records[..., :dim], fill, records[..., dim:]], axis=-1).reshape(B, steps * stride)

def batch(rng, B, dim, stride, size=SIZE, steps=STEPS):
    """tokens (B, steps * stride): B sequences in `dim` dimensions, written at this stride."""
    return pad(dx.batch(rng, B, dim, size, steps), dim, stride, size)

def pairs(dims, strides):
    """The (dim, stride) pairs that exist: a record needs room for its coordinates and a SEP."""
    return [(d, s) for d in dims for s in strides if s >= d + 1]
