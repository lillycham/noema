"""Data of the mixing world: held-out turns, batches, and the ideal learner.

    pytest tests/test_data_mixing.py
"""
import numpy as np

from noema import data_mixing as dm, data_torus as dt, mixing

SIZE, DIM, STEPS = dm.SIZE, dm.DIM, dm.STEPS
SLOW = mixing.reference_advance

def batch(seed=0, B=6, turns=None):
    turns = turns if turns is not None else dm.TurnSet().train
    return dm.trajectory_batch(np.random.default_rng(seed), B, turns, advance_fn=SLOW)

def test_a_quarter_of_the_turns_are_held_out_and_never_mixed_with_the_rest():
    ts = dm.TurnSet()
    assert len(ts.held_turns) == 2 and len(ts.train_turns) == 6
    rng = np.random.default_rng(0)
    assert not any(ts.is_held_out(R) for R in ts.train(rng, 100))
    assert all(ts.is_held_out(R) for R in ts.held(rng, 100))

def test_a_different_seed_holds_out_different_turns():
    a, b = dm.TurnSet(seed=0), dm.TurnSet(seed=1)
    assert {R.tobytes() for R in a.held_turns} != {R.tobytes() for R in b.held_turns}

def test_three_dimensions_hold_out_twelve_of_forty_eight():
    assert len(dm.TurnSet(3).held_turns) == 12

def test_a_batch_is_whole_trajectories_in_the_tokens_of_the_torus_world():
    X, R = batch()
    assert X.shape == (6, STEPS * (DIM + 1)) and R.shape == (6, DIM, DIM)
    assert (X.reshape(6, STEPS, DIM + 1)[:, :, DIM] == dt.SEP).all()
    assert X.min() >= 0 and X.max() < dt.VOCAB

def test_the_same_seed_gives_the_same_batch():
    (A, a), (B, b) = batch(3), batch(3)
    assert np.array_equal(A, B) and np.array_equal(a, b)

def test_the_velocity_turns_by_the_turn_of_the_sequence_each_step():
    X, R = batch()
    P = dt.decode(X, DIM)
    V = (P[:, 1:] - P[:, :-1]) % SIZE
    for t in range(STEPS - 2):
        assert np.array_equal(V[:, t + 1], np.einsum("bjk,bk->bj", R, V[:, t]) % SIZE)

def test_with_only_the_identity_the_ideal_learner_is_certain_from_step_two():
    X, _ = batch(B=200, turns=np.eye(DIM, dtype=np.int64)[None])
    c = dm.ideal_accuracy_by_step(dt.decode(X, DIM), np.eye(DIM, dtype=np.int64)[None], SIZE)
    assert c[0] == 1 / SIZE and np.allclose(c[1:], 1.0)

def test_the_ideal_learner_has_to_guess_at_first_and_is_nearly_certain_later():
    X, _ = batch(B=1000)
    c = dm.ideal_accuracy_by_step(dt.decode(X, DIM), mixing.signed_permutations(DIM), SIZE)
    assert c.shape == (STEPS - 1,) and c[0] == 1 / SIZE
    assert c[1] < 0.6 and c[-1] > 0.9
