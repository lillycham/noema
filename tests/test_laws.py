"""The law checks, tested with an oracle that keeps every law and predictors that each break one.

    pytest tests/test_laws.py
"""
import numpy as np
import torch

from noema import data_torus as dt, gpt, laws

SIZE, DIM = dt.SIZE, dt.DIM
VELS = dt.VelocitySet()

def oracle(P):
    """The true next position, from the last two positions: what a model that has learned the world does."""
    return (2 * P[:, -1] - P[:, -2]) % SIZE

def anchored(P):
    """Breaks translation: the answer is always the same wherever the prompt is."""
    return np.zeros_like(P[:, -1])

def coupled(P):
    """Breaks independence: the second axis moves by the first axis's velocity."""
    out = oracle(P)
    out[:, 1] = (P[:, -1, 1] + (P[:, -1, 0] - P[:, -2, 0])) % SIZE
    return out

def forward_only(P):
    """Breaks reversal: assumes the velocity is the one a forward-running trajectory has, whatever the order."""
    return (P[:, -1] + (P[:, 2] - P[:, 1]) * np.sign(P[:, 2] - P[:, 0]) + 1) % SIZE

def results(predict, **kw):
    return laws.run(predict, np.random.default_rng(0), VELS, SIZE, DIM, B=256, **kw)

def test_the_oracle_keeps_every_law():
    for name, r in results(oracle).items():
        assert r.all(), name

def test_translation_catches_a_predictor_that_ignores_where_it_is():
    r = results(anchored)
    assert r["translation/seen"].mean() < 0.2

def test_independence_catches_a_predictor_that_mixes_the_axes():
    r = results(coupled)
    assert r["independence axis 1/seen"].mean() < 0.5
    assert r["independence axis 0/seen"].all()

def test_truth_catches_a_wrong_answer():
    r = results(forward_only)
    assert r["truth/seen"].mean() < 0.5

def test_reversal_is_split_by_whether_the_reversed_velocity_was_held_out():
    r = results(oracle)
    assert len(r["reversal/-v held-out"]) > 0 and len(r["reversal/-v seen"]) > 0

def test_rollout_has_one_column_for_each_step_after_the_prompt():
    r = results(oracle, m=3)
    assert r["rollout/seen"].shape == (256, 12 - 3)

def test_the_model_predictor_writes_a_position_in_range():
    P, _ = laws.cases(np.random.default_rng(0), 5, VELS.train)
    model = gpt.GPT(gpt.Config(vocab=dt.VOCAB, block=36, emb=16, heads=2, layers=2))
    out = laws.model_predictor(model, SIZE, DIM, "cpu")(P)
    assert out.shape == (5, DIM) and out.min() >= 0 and out.max() < SIZE
