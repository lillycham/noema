"""The rung 4 measures.

    pytest tests/test_probe_mixed.py
"""
import numpy as np
import torch

from noema import data_torus as dt, mixing, probe_mixed as pm

def test_the_turns_that_fix_the_last_axis_are_the_turns_of_one_axis_less():
    for d in (2, 3, 4):
        assert len(pm.fixing_last_axis(d)) == len(mixing.signed_permutations(d - 1))
        assert (pm.fixing_last_axis(d)[:, -1, -1] == 1).all()

def test_the_frozen_world_has_a_constant_last_axis():
    X = pm.graded_batch(np.random.default_rng(0), 16, 4, "frozen")
    P = dt.decode(X, 4)
    assert (P[:, :, 3] == P[:, :1, 3]).all()

def test_the_identity_world_keeps_one_velocity():
    P = dt.decode(pm.graded_batch(np.random.default_rng(1), 16, 3, "identity"), 3)
    V = (P[:, 1:] - P[:, :-1]) % dt.SIZE
    assert (V == V[:, :1]).all()

def test_live_ids_mark_coordinates_from_the_first_step_named():
    ids = pm.live_ids(2, 12, 2, first_step=2)
    assert ids.shape == (2, 12)
    assert ids[0].tolist() == [-1] * 6 + [0, 0, -1, 0, 0, -1]

def test_stride_count_takes_sharp_heads_a_whole_number_of_records_back():
    distance, share = np.array([[7, 19, 9, 3]]), np.array([[0.5, 0.5, 0.5, 0.1]])
    assert pm.stride_count(distance, share, 4) == 2             # 7 and 19; 3 is not sharp
    assert pm.stride_count(distance, share, 5) == 2             # 19 and 9

def test_copy_agreement_finds_the_period_of_a_guess():
    X = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3]])
    guess = X[:, 1:]
    assert pm.copy_agreement(guess, X, [3])[0] == 1.0
    assert pm.copy_agreement(guess, X, [1])[0] == 0.0
