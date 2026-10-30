"""The mixed-dimension data of rung 4.

    pytest tests/test_data_mixed.py
"""
import numpy as np

from noema import data_mixed as dx, data_torus as dt, mixing

SIZE, STEPS = dx.SIZE, dx.STEPS

def test_a_batch_has_steps_times_dim_plus_one_tokens():
    for d in (1, 2, 3, 4):
        X = dx.batch(np.random.default_rng(d), 5, d)
        assert X.shape == (5, STEPS * (d + 1)) and X.min() >= 0 and X.max() <= SIZE

def test_every_step_ends_in_a_separator_and_holds_no_other():
    for d in (1, 3, 4):
        X = dx.batch(np.random.default_rng(0), 6, d).reshape(6, STEPS, d + 1)
        assert (X[:, :, d] == dt.SEP).all() and (X[:, :, :d] < SIZE).all()

def test_a_one_dimensional_sequence_is_a_line_with_a_velocity_that_may_flip_sign():
    P = dt.decode(dx.batch(np.random.default_rng(1), 20, 1), 1)[..., 0]
    V = (P[:, 1:] - P[:, :-1]) % SIZE
    assert (np.minimum(V[:, 1:], SIZE - V[:, 1:]) == np.minimum(V[:, :-1], SIZE - V[:, :-1])).all()

def test_a_batch_follows_the_world_step_by_step():
    d = 3
    P = dt.decode(dx.batch(np.random.default_rng(2), 8, d), d)
    G = mixing.signed_permutations(d)
    for seq in P:
        V = (seq[1:] - seq[:-1]) % SIZE
        assert any(all(((R @ V[i]) % SIZE == V[i + 1]).all() for i in range(len(V) - 1)) for R in G)

def test_the_same_seed_gives_the_same_batch():
    assert np.array_equal(dx.batch(np.random.default_rng(3), 4, 2), dx.batch(np.random.default_rng(3), 4, 2))

def test_train_batch_draws_only_from_the_dims_it_is_given():
    rng, seen = np.random.default_rng(4), set()
    for _ in range(60):
        X, d = dx.train_batch(rng, 2, (1, 2, 3))
        assert X.shape[1] == STEPS * (d + 1)
        seen.add(d)
    assert seen == {1, 2, 3}

def test_the_ideal_learner_starts_at_one_over_size_and_reaches_one_in_every_dimension():
    for d in (1, 2, 3, 4):
        curve = dx.ideal(dx.batch(np.random.default_rng(5), 64, d), d)
        assert curve.shape == (STEPS - 1,) and abs(curve[0] - 1 / SIZE) < 1e-9 and curve[-1] > 0.99
