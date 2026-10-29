"""A particle that turns: a torus world where the axes are no longer independent.

The world of torus.py adds a constant velocity to the position, axis by axis. Here the velocity
also turns every step. A turn R is a signed permutation of the axes: it moves each axis's velocity
onto another axis, and may flip its sign. In 2D there are 8 (the identity, three quarter turns,
and four reflections), in 3D there are 48, and in d dimensions 2^d * d!. One step is

    position  <-  position + velocity            (mod SIZE)
    velocity  <-  R @ velocity                   (mod SIZE)

With the identity for R this is the world of torus.py. With a quarter turn the particle goes round
in a loop: velocity (2, 0) becomes (0, 2), (-2, 0), (0, -2), and the x of the next step depends on
what the y velocity was. So a model that works with one copy of a circuit for each axis is not enough:
something has to carry a velocity from one axis to another.

R and the first velocity are hidden. Each sequence has its own. A model must work out R from the
first two velocities (v1 = R v0) and keep applying it.

advance() is the step, for whole batches at once. reference_advance() is the slow, obviously correct
one. tests/test_mixing.py checks advance() against it.

Positions and velocities are (..., d) arrays of integers in 0 .. SIZE - 1 (a velocity of -1 is
SIZE - 1), and a turn is a (..., d, d) array of -1, 0 and 1.
"""
import itertools

import numpy as np

def signed_permutations(dim):
    """All 2^d * d! signed permutation matrices of d axes: (n, d, d) int64, in a fixed order."""
    out = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product((1, -1), repeat=dim):
            R = np.zeros((dim, dim), dtype=np.int64)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                R[row, col] = sign
            out.append(R)
    return np.stack(out)

def reference_advance(positions, velocities, turns, size):
    """(next positions, next velocities), one coordinate at a time. positions and velocities (..., d),
    turns (..., d, d), broadcast against each other. The slow version that advance() is held against."""
    positions, velocities = np.broadcast_arrays(np.asarray(positions), np.asarray(velocities))
    turns = np.broadcast_to(np.asarray(turns), positions.shape + positions.shape[-1:])
    dim = positions.shape[-1]
    new_p = np.empty(positions.shape, dtype=np.int64)
    new_v = np.empty(positions.shape, dtype=np.int64)
    for index in np.ndindex(positions.shape):
        axis = index[-1]
        new_p[index] = (int(positions[index]) + int(velocities[index])) % size
        total = sum(int(turns[index + (k,)]) * int(velocities[index[:-1] + (k,)]) for k in range(dim))
        new_v[index] = total % size
    return new_p, new_v

def advance(positions, velocities, turns, size):
    """(next positions, next velocities), for whole batches at once: positions and velocities (..., d),
    turns (..., d, d), as for reference_advance. Both results are int64."""
    positions, velocities = np.broadcast_arrays(np.asarray(positions, dtype=np.int64), np.asarray(velocities, dtype=np.int64))
    turned = np.einsum("...jk,...k->...j", np.asarray(turns, dtype=np.int64), velocities)
    return (positions + velocities) % size, turned % size

def trajectory(starts, velocities, turns, steps, size, advance_fn=None):
    """The positions of each particle over time: (..., steps, d), the first of them the start.
    starts and velocities are (..., d) (the velocity at the start), turns (..., d, d)."""
    advance_fn = advance_fn or advance
    here, v = np.asarray(starts, dtype=np.int64), np.asarray(velocities, dtype=np.int64)
    out = []
    for _ in range(steps):
        out.append(here)
        here, v = advance_fn(here, v, turns, size)
    return np.stack(out, axis=-2)
