"""The torus world. The first group checks reference_advance(). The last checks advance() against it.

    pytest tests/test_torus.py

The tests of advance() fail until Lilly writes it.
"""
import numpy as np

from noema import torus

SIZE = 7

def random_ints(shape, seed=0):
    return np.random.default_rng(seed).integers(0, SIZE, size=shape)

# ---------- the reference ----------
def test_reference_adds_axis_by_axis():
    assert torus.reference_advance([3, 9], [4, 3], 11).tolist() == [7, 1]

def test_reference_wraps_round_at_the_edge():
    assert torus.reference_advance([SIZE - 1], [1], SIZE).tolist() == [0]

def test_reference_zero_velocity_stays_put():
    p = random_ints((5, 3))
    assert np.array_equal(torus.reference_advance(p, np.zeros(3, dtype=int), SIZE), p)

def test_reference_velocity_broadcasts_over_a_batch():
    p = random_ints((5, 3))
    out = torus.reference_advance(p, [1, 0, 2], SIZE)
    assert out.shape == (5, 3) and np.array_equal(out, (p + [1, 0, 2]) % SIZE)

# ---------- advance ----------
def test_advance_matches_the_reference():
    p, v = random_ints((8, 4, 3), 1), random_ints((8, 4, 3), 2)
    assert np.array_equal(torus.advance(p, v, SIZE), torus.reference_advance(p, v, SIZE))

def test_advance_broadcasts_one_velocity_over_a_batch():
    p = random_ints((8, 2), 3)
    assert np.array_equal(torus.advance(p, [3, 5], SIZE), torus.reference_advance(p, [3, 5], SIZE))

def test_advance_is_int64_and_in_range():
    out = torus.advance(random_ints((6, 2)).astype(np.uint8), random_ints((6, 2)).astype(np.uint8), SIZE)
    assert out.dtype == np.int64 and out.min() >= 0 and out.max() < SIZE

def test_n_advances_by_a_velocity_are_one_advance_by_n_times_it():
    p, v = random_ints((6, 2), 4), random_ints((6, 2), 5)
    for n in range(1, 5):
        here = p
        for _ in range(n):
            here = torus.advance(here, v, SIZE)
        assert np.array_equal(here, torus.advance(p, n * v, SIZE))

# ---------- trajectory ----------
def test_trajectory_starts_at_the_start_and_adds_the_velocity_each_step():
    p, v = random_ints((6, 2), 6), random_ints((6, 2), 7)
    t = torus.trajectory(p, v, 5, SIZE, torus.reference_advance)
    assert t.shape == (6, 5, 2) and np.array_equal(t[:, 0], p)
    for k in range(5):
        assert np.array_equal(t[:, k], (p + k * v) % SIZE)
