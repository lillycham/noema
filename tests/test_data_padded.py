"""The world with free strides.

    pytest tests/test_data_padded.py
"""
import numpy as np

from noema import data_mixed as dx, data_padded as dp, data_torus as dt

def test_pad_with_no_room_changes_nothing():
    X = dx.batch(np.random.default_rng(0), 4, 3)
    assert (dp.pad(X, 3, 4) == X).all()

def test_padding_keeps_the_coordinates_and_the_separators():
    X = dx.batch(np.random.default_rng(1), 4, 2)
    Y = dp.pad(X, 2, 6)
    assert Y.shape == (4, dt.STEPS * 6)
    assert (Y[:, 0::6] == X[:, 0::3]).all() and (Y[:, 1::6] == X[:, 1::3]).all()
    assert (Y[:, 2::6] == dp.filler()).all() and (Y[:, 4::6] == dp.filler()).all() and (Y[:, 5::6] == dt.SEP).all()

def test_a_mask_at_a_stride_counts_the_coordinates_from_the_second_record():
    mask = dt.target_mask(dt.STEPS * 6, 2, 6)
    assert mask.sum() == 2 * (dt.STEPS - 1)
    assert mask[5] and mask[6] and not mask[7]          # targets are tokens[1:], so these are tokens 6, 7, 8

def test_pairs_leave_out_records_too_short():
    assert dp.pairs([1, 3], [2, 4]) == [(1, 2), (1, 4), (3, 4)]
