"""The mixing world. The first groups check reference_advance() and the turns. The last checks
advance() against reference_advance().

    pytest tests/test_mixing.py

advance() is written by Claude (73750d0), at Lilly's word.
"""
import numpy as np

from noema import mixing

SIZE = 11
QUARTER = np.array([[0, -1], [1, 0]])
IDENTITY = np.eye(2, dtype=np.int64)

def random_ints(shape, seed=0):
    return np.random.default_rng(seed).integers(0, SIZE, size=shape)

# ---------- the turns ----------
def test_there_are_2_to_the_d_times_d_factorial_turns():
    for d, n in ((1, 2), (2, 8), (3, 48), (4, 384)):
        assert len(mixing.signed_permutations(d)) == n

def test_the_turns_are_distinct_and_each_axis_goes_to_one_axis_with_a_sign():
    G = mixing.signed_permutations(3)
    assert len({R.tobytes() for R in G}) == 48
    assert (np.abs(G).sum(-1) == 1).all() and (np.abs(G).sum(-2) == 1).all()

def test_the_identity_is_among_the_turns_and_the_turns_are_closed_under_composition():
    G = mixing.signed_permutations(2)
    keys = {R.tobytes() for R in G}
    assert IDENTITY.tobytes() in keys
    assert all((A @ B).tobytes() in keys for A in G for B in G)

# ---------- the reference ----------
def test_reference_with_the_identity_is_the_world_of_the_torus():
    p, v = np.array([3, 9]), np.array([4, 3])
    new_p, new_v = mixing.reference_advance(p, v, IDENTITY, SIZE)
    assert new_p.tolist() == [7, 1] and new_v.tolist() == [4, 3]

def test_reference_a_quarter_turn_moves_the_velocity_onto_the_other_axis():
    new_p, new_v = mixing.reference_advance([3, 3], [2, 0], QUARTER, SIZE)
    assert new_p.tolist() == [5, 3] and new_v.tolist() == [0, 2]

def test_reference_a_negative_sign_wraps_the_velocity_round():
    _, new_v = mixing.reference_advance([0, 0], [2, 0], QUARTER.T, SIZE)       # the other way: (2, 0) -> (0, -2)
    assert new_v.tolist() == [0, SIZE - 2]

def test_four_quarter_turns_are_the_identity():
    v = random_ints((5, 2), 1)
    here = v
    for _ in range(4):
        _, here = mixing.reference_advance(np.zeros((5, 2), dtype=int), here, QUARTER, SIZE)
    assert np.array_equal(here, v)

def test_trajectory_follows_the_reference_step_by_step():
    p, v = random_ints((4, 2), 2), random_ints((4, 2), 3)
    t = mixing.trajectory(p, v, QUARTER, 6, SIZE, mixing.reference_advance)
    assert t.shape == (4, 6, 2) and np.array_equal(t[:, 0], p)
    here, vel = p, v
    for k in range(1, 6):
        here, vel = mixing.reference_advance(here, vel, QUARTER, SIZE)
        assert np.array_equal(t[:, k], here)

# ---------- advance ----------
def test_advance_matches_the_reference():
    G = mixing.signed_permutations(3)
    p, v = random_ints((8, 3), 4), random_ints((8, 3), 5)
    R = G[np.random.default_rng(6).integers(0, len(G), size=8)]
    for got, want in zip(mixing.advance(p, v, R, SIZE), mixing.reference_advance(p, v, R, SIZE)):
        assert np.array_equal(got, want)

def test_advance_takes_one_turn_for_a_whole_batch():
    p, v = random_ints((8, 2), 7), random_ints((8, 2), 8)
    for got, want in zip(mixing.advance(p, v, QUARTER, SIZE), mixing.reference_advance(p, v, QUARTER, SIZE)):
        assert np.array_equal(got, want)

def test_advance_is_int64_and_in_range():
    new_p, new_v = mixing.advance(random_ints((6, 2)).astype(np.uint8), random_ints((6, 2), 1).astype(np.uint8), QUARTER, SIZE)
    for a in (new_p, new_v):
        assert a.dtype == np.int64 and a.min() >= 0 and a.max() < SIZE

def test_advance_does_not_change_the_dimension_it_is_given():
    G = mixing.signed_permutations(4)
    p, v = random_ints((3, 4), 9), random_ints((3, 4), 10)
    new_p, new_v = mixing.advance(p, v, G[[5, 100, 300]], SIZE)
    assert new_p.shape == new_v.shape == (3, 4)
