"""The laws of the mixing world, held against an oracle that keeps them and predictors that break them.

    pytest tests/test_laws_mixing.py
"""
import numpy as np

from noema import laws_mixing as lm, mixing

SIZE, DIM = 11, 3
G = mixing.signed_permutations(DIM)

def scored(predict, seed=0):
    return lm.run(predict, np.random.default_rng(seed), G, SIZE, DIM, B=512)

def test_cases_are_trajectories_with_a_fixed_answer():
    P, R, v0 = lm.cases(np.random.default_rng(1), 256, G, SIZE, DIM)
    assert len(P) > 128
    assert lm.is_fixed(P, G, SIZE).all()
    assert ((P[:, 1] - P[:, 0] - v0) % SIZE == 0).all()

def test_the_oracle_gives_the_true_next_position():
    rng = np.random.default_rng(2)
    P, R, v0 = lm.cases(rng, 256, G, SIZE, DIM)
    full = mixing.trajectory(P[:, 0], v0, R, P.shape[1] + 1, SIZE)
    assert (lm.oracle(G, SIZE)(P) == full[:, -1]).all()

def test_the_oracle_keeps_every_law():
    for name, r in scored(lm.oracle(G, SIZE)).items():
        assert r.mean() == 1.0, name

def test_a_predictor_anchored_to_a_fixed_point_breaks_translation():
    oracle = lm.oracle(G, SIZE)
    def anchored(P):
        out = oracle(P)
        return np.where(P[:, -1:, :].sum((1, 2), keepdims=False)[:, None] % 2 == 0, out, (out + 1) % SIZE)
    assert scored(anchored)["translation"].mean() < 0.8

def test_a_predictor_with_no_turn_breaks_truth_and_reversal():
    def straight(P):
        return (2 * P[:, -1] - P[:, -2]) % SIZE
    r = scored(straight)
    assert r["truth"].mean() < 0.3 and r["reversal"].mean() < 0.3

def test_a_predictor_that_treats_one_axis_differently_breaks_relabel():
    oracle = lm.oracle(G, SIZE)
    def lopsided(P):
        out = oracle(P).copy()
        out[:, 0] = (out[:, 0] + 1) % SIZE
        return out
    r = scored(lopsided)
    assert r["relabel"].mean() < 0.5 and r["translation"].mean() == 1.0
