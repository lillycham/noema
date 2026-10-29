import numpy as np
import pytest

from noema import tokens

W, R = 15, 16

def rollout(seed=0, shape=(3, R, W)):
    return np.random.default_rng(seed).integers(0, 2, size=shape, dtype=np.uint8)

def test_encode_makes_whole_rows_with_a_separator_each():
    X = tokens.encode(rollout())
    assert X.shape == (3, R * (W + 1)) and X.dtype == np.int64
    assert (X[:, W::W + 1] == tokens.SEP).all()                 # every 16th token
    assert ((X == tokens.SEP).sum(axis=1) == R).all()           # and no others

def test_decode_is_the_inverse_of_encode():
    rows = rollout()
    assert np.array_equal(tokens.decode(tokens.encode(rows), W), rows)

def test_encode_a_single_rollout_without_a_batch():
    rows = rollout(shape=(R, W))
    assert tokens.encode(rows).shape == (R * (W + 1),)

def test_decode_refuses_a_partial_row():
    with pytest.raises(ValueError):
        tokens.decode(tokens.encode(rollout())[:, :-3], W)

def test_decode_refuses_a_missing_separator():
    X = tokens.encode(rollout())
    X[:, W] = tokens.ONE
    with pytest.raises(ValueError):
        tokens.decode(X, W)

def test_the_loss_counts_the_cells_of_rows_one_onwards():
    """15 rows of 15 cells: not the random first row, and not the separators."""
    mask = tokens.target_mask(R * (W + 1), W)
    assert mask.shape == (R * (W + 1) - 1,)
    assert mask.sum() == (R - 1) * W

def test_the_mask_lines_up_with_the_targets():
    """Entry i is for the token at position i + 1: the first counted one is cell 0 of row 1, position 16."""
    mask = tokens.target_mask(R * (W + 1), W)
    assert not mask[:W].any() and mask[W]                    # entry 15 is position 16
    X = tokens.encode(rollout())
    assert (X[:, 1:][:, mask] != tokens.SEP).all()           # no separator is ever counted
