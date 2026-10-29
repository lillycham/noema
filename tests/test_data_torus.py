"""Tokens, held-out velocities and batches of the torus world.

    pytest tests/test_data_torus.py
"""
import numpy as np
import torch

from noema import data_torus as dt, gpt, torus
from noema.train_torus import score

SLOW = torus.reference_advance
SIZE, DIM, STEPS = dt.SIZE, dt.DIM, dt.STEPS

def batch(seed=0, B=6, velocities=None):
    velocities = velocities if velocities is not None else dt.VelocitySet().train
    return dt.trajectory_batch(np.random.default_rng(seed), B, velocities, advance_fn=SLOW)

def test_encode_and_decode_are_inverses():
    p = np.random.default_rng(0).integers(0, SIZE, size=(4, STEPS, 3))
    assert np.array_equal(dt.decode(dt.encode(p), 3), p)

def test_a_batch_is_whole_trajectories_with_one_separator_a_step():
    X, v = batch()
    assert X.shape == (6, STEPS * (DIM + 1)) and X.dtype == np.int64 and v.shape == (6, DIM)
    assert (X.reshape(6, STEPS, DIM + 1)[:, :, DIM] == dt.SEP).all()
    assert X.min() >= 0 and X.max() < dt.VOCAB

def test_the_same_seed_gives_the_same_batch():
    (A, a), (B, b) = batch(3), batch(3)
    assert np.array_equal(A, B) and np.array_equal(a, b)

def test_each_step_adds_the_velocity():
    X, v = batch()
    p = dt.decode(X, DIM)
    for t in range(STEPS - 1):
        assert np.array_equal(p[:, t + 1], (p[:, t] + v) % SIZE)

def test_the_same_token_means_the_same_number_on_every_axis():
    """No token belongs to an axis: a coordinate token is a number from 0 to SIZE - 1 and nothing else."""
    X, _ = batch(B=64)
    assert set(np.unique(X)) == set(range(dt.VOCAB))

def test_the_loss_counts_coordinates_of_the_second_step_on():
    m = dt.target_mask(STEPS * (DIM + 1), DIM)
    assert m.shape == (STEPS * (DIM + 1) - 1,) and m.sum() == (STEPS - 1) * DIM
    X = np.arange(STEPS * (DIM + 1))
    assert (X[1:][m] % (DIM + 1) < DIM).all() and (X[1:][m] >= DIM + 1).all()

def test_trained_and_held_out_velocities_never_mix():
    vs, rng = dt.VelocitySet(), np.random.default_rng(0)
    assert not any(vs.is_held_out(v) for v in vs.train(rng, 200))
    assert all(vs.is_held_out(v) for v in vs.held(rng, 200))

def test_about_one_velocity_in_sixteen_is_held_out():
    vs = dt.VelocitySet()
    n = sum(vs.is_held_out([a, b]) for a in range(SIZE) for b in range(SIZE))
    assert 3 <= n <= 20

def test_a_different_seed_holds_out_different_velocities():
    a, b = dt.VelocitySet(seed=0), dt.VelocitySet(seed=1)
    grid = [[x, y] for x in range(SIZE) for y in range(SIZE)]
    assert [a.is_held_out(v) for v in grid] != [b.is_held_out(v) for v in grid]

def test_the_ideal_curve_is_chance_then_certain():
    c = dt.ideal_accuracy_by_step(STEPS)
    assert c.shape == (STEPS - 1,) and c[0] == 1 / SIZE and (c[1:] == 1).all()

def test_a_model_scores_chance_before_training_and_score_has_the_right_shape():
    X, _ = batch(B=8)
    model = gpt.GPT(gpt.Config(vocab=dt.VOCAB, block=X.shape[1], emb=32, heads=2, layers=2))
    loss, acc = score(model, torch.from_numpy(X), torch.from_numpy(dt.target_mask(X.shape[1], DIM)), DIM, SIZE)
    assert acc.shape == (STEPS - 1,) and abs(loss - np.log(dt.VOCAB)) < 0.5
