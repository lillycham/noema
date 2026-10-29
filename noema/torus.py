"""A particle on a torus: rung 3, a world of coordinates.

The world is a grid of SIZE points along each of d axes, with edges that wrap round (a torus). A
particle sits at a position, a tuple of d numbers from 0 to SIZE - 1, and has a velocity, another
such tuple. One step adds the velocity to the position, axis by axis, modulo SIZE:

    position (3, 9), velocity (4, 3), SIZE 11   ->   (7, 1)

So the world is a product of d independent copies of addition modulo SIZE. This is the "easy
version" of the plan: one circuit used on every axis would already be dimension-agnostic. The
velocity is never shown to a model. Each trajectory has its own, so a model must work it out from
the steps so far, as the rule is worked out from the rows in the 1D world.

advance() is the step, for whole batches at once. reference_advance() is the slow, obviously
correct one, a coordinate at a time. tests/test_torus.py checks advance() against it.

Arrays have the d axes of the world on the last axis, (..., d), and any axes in front are a batch.
In a batch, every particle can have its own velocity.
"""
import numpy as np

def reference_advance(positions, velocities, size):
    """The next positions, one coordinate at a time. positions and velocities (..., d) of integers
    in 0 .. size - 1, broadcast against each other. The slow version that advance() is held against."""
    positions, velocities = np.broadcast_arrays(np.asarray(positions), np.asarray(velocities))
    out = np.empty(positions.shape, dtype=np.int64)
    for index in np.ndindex(positions.shape):
        out[index] = (int(positions[index]) + int(velocities[index])) % size
    return out

def advance(positions, velocities, size):
    """The next positions, for whole batches at once: positions and velocities (..., d), as for
    reference_advance. The result is int64, shaped like the two broadcast together."""
    return (np.asarray(positions, dtype=np.int64)+ np.asarray(velocities, dtype=np.int64)) % size

def trajectory(starts, velocities, steps, size, advance_fn=None):
    """The positions of each particle over time: (..., steps, d), the first of them the start.
    starts and velocities are (..., d). advance_fn is advance unless a slower one is given."""
    advance_fn = advance_fn or advance
    here = np.asarray(starts, dtype=np.int64)
    out = []
    for _ in range(steps):
        out.append(here)
        here = advance_fn(here, velocities, size)
    return np.stack(out, axis=-2)
