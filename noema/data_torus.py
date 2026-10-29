"""Training data for the torus world: a hidden velocity and a random start, rolled out, as tokens.

A position is d tokens, one per coordinate, each from 0 to SIZE - 1, and a SEP token ends the
step. Axis identity comes from place in the record and from nothing else: the same token means
"3" on every axis, and a world with a third axis needs no new token. (A token per axis, or one
token per position, would make the held-out dimension of rung 4 a symbol never trained.)

With SIZE = 11, DIM = 2 and STEPS = 12 a step is 3 tokens and a sequence 36. The cell the model
must predict, coordinate k of step t, has its history at fixed distances back: the same axis in the
previous step is d + 1 tokens back, and in the step before that 2 * (d + 1).

Held-out velocities are chosen by a hash of the velocity, about 1 in 16 as in data.RuleSet. A
model that has learned addition modulo SIZE does as well on them as on the others. A model that
has learned a table for each velocity it saw does not.
"""
import numpy as np

from noema import data, torus

SIZE, DIM, STEPS = 11, 2, 12
SEP = SIZE
VOCAB = SIZE + 1

def encode(positions):
    """positions (..., steps, d) -> tokens (..., steps * (d + 1)), int64."""
    positions = np.asarray(positions)
    sep = np.full(positions.shape[:-1] + (1,), SEP, dtype=positions.dtype)
    both = np.concatenate([positions, sep], axis=-1)
    return both.reshape(*positions.shape[:-2], -1).astype(np.int64)

def decode(tokens, dim):
    """tokens (..., steps * (d + 1)) -> positions (..., steps, d). The inverse of encode."""
    tokens = np.asarray(tokens)
    stride = dim + 1
    if tokens.shape[-1] % stride:
        raise ValueError(f"{tokens.shape[-1]} tokens is not a whole number of steps of {stride}")
    steps = tokens.reshape(*tokens.shape[:-1], -1, stride)
    if not (steps[..., dim] == SEP).all():
        raise ValueError("a step does not end in a separator")
    return steps[..., :dim].astype(np.int64)

def target_mask(length, dim):
    """Which targets count in the loss for a sequence of this length. The targets are tokens[1:], so
    entry i is the token at position i + 1. True for the coordinates of the second step and later.
    Not the first step, which is random, and not the separators."""
    position = np.arange(1, length)
    stride = dim + 1
    return (position >= stride) & (position % stride < dim)

class VelocitySet:
    """The velocities a run trains on and the ones it holds out. train() never returns a held-out
    velocity, and held() returns only held-out ones. A velocity is d numbers from 0 to size - 1."""

    def __init__(self, size=SIZE, dim=DIM, seed=0):
        self.size, self.dim, self.seed = size, dim, seed

    def is_held_out(self, velocity):
        number = 0
        for v in np.asarray(velocity).tolist():
            number = number * self.size + int(v)
        return data.held_out_by_hash(number, self.seed)

    def _draw(self, rng, B, held):
        out = []
        while len(out) < B:
            v = rng.integers(0, self.size, size=self.dim)
            if self.is_held_out(v) == held:
                out.append(v)
        return np.array(out, dtype=np.int64)

    def train(self, rng, B):
        return self._draw(rng, B, held=False)

    def held(self, rng, B):
        return self._draw(rng, B, held=True)

def trajectory_batch(rng, B, velocities, size=SIZE, dim=DIM, steps=STEPS, advance_fn=None):
    """(tokens (B, steps * (dim + 1)) int64, velocities (B, dim)): B sequences, each with its own
    velocity and random start. velocities is an array (n, dim) to choose from, or a function
    (rng, B) -> (B, dim), such as VelocitySet().train."""
    chosen = velocities(rng, B) if callable(velocities) else np.asarray(velocities)[rng.integers(0, len(velocities), size=B)]
    starts = rng.integers(0, size, size=(B, dim))
    return encode(torus.trajectory(starts, chosen, steps, size, advance_fn)), chosen

def ideal_accuracy_by_step(steps, size=SIZE):
    """The accuracy per coordinate of an ideal learner on steps 1 to steps - 1: an array of steps - 1.
    The first move is a draw from size velocities: chance, 1 / size. After that the velocity is the
    difference of the two positions before, on each axis, so the learner is certain. So the curve
    for the model to be held against is 1 / size, then 1.00."""
    return np.array([1 / size] + [1.0] * (steps - 2))
