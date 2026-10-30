"""The probes of the mixing world: the targets are the right ones.

    pytest tests/test_probe_mixing.py
"""
import numpy as np

from noema import data_mixing as dm, data_torus as dt, gpt, mixing, probe_mixing

SIZE, DIM, STEPS = dm.SIZE, dm.DIM, dm.STEPS
TS = dm.TurnSet()

def setup(B=6, seed=0):
    X, R = dm.trajectory_batch(np.random.default_rng(seed), B, TS.train, advance_fn=mixing.reference_advance)
    model = gpt.GPT(gpt.Config(vocab=dt.VOCAB, block=X.shape[1], emb=16, heads=2, layers=2))
    return model, X, R

def test_turn_index_finds_each_turn_in_the_group():
    G = TS.group
    assert probe_mixing.turn_index(G, G).tolist() == list(range(len(G)))

def test_targets_are_consistent_with_the_world():
    model, X, R = setup()
    for axis in range(DIM):
        F, tg = probe_mixing.collect(model, X, R, TS.group, DIM, SIZE, STEPS, range(2, STEPS - 1), axis)
        assert F.shape[:2] == (3, 6 * (STEPS - 3))
        assert np.array_equal((tg["position"] + tg["next velocity"]) % SIZE, tg["next"])

def test_the_next_velocity_is_the_turn_applied_to_the_last_one():
    model, X, R = setup()
    ts = list(range(2, STEPS - 1))
    got = {a: probe_mixing.collect(model, X, R, TS.group, DIM, SIZE, STEPS, ts, a)[1] for a in range(DIM)}
    last = np.stack([got[a]["last velocity"] for a in range(DIM)], -1).reshape(6, len(ts), DIM)
    nxt = np.stack([got[a]["next velocity"] for a in range(DIM)], -1).reshape(6, len(ts), DIM)
    assert np.array_equal(nxt, np.einsum("bjk,btk->btj", R, last) % SIZE)

def test_the_turn_target_is_the_same_for_both_axes_and_in_range():
    model, X, R = setup()
    t0 = probe_mixing.collect(model, X, R, TS.group, DIM, SIZE, STEPS, [2], 0)[1]["turn"]
    t1 = probe_mixing.collect(model, X, R, TS.group, DIM, SIZE, STEPS, [2], 1)[1]["turn"]
    assert np.array_equal(t0, t1) and t0.min() >= 0 and t0.max() < len(TS.group)

def test_turn_recall_is_one_for_a_readable_turn_and_zero_for_a_probe_that_names_one_turn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=2000)
    F = np.eye(4)[y] + 0.05 * rng.normal(size=(2000, 4))
    assert probe_mixing.turn_recall(F[:1000], y[:1000], F[1000:], y[1000:], 4).min() > 0.95
    noise = rng.normal(size=(2000, 4))
    r = probe_mixing.turn_recall(noise[:1000], y[:1000], noise[1000:], y[1000:], 4)
    assert abs(r.mean() - 0.25) < 0.1

def test_row_class_names_the_axis_that_feeds_this_one_and_the_sign():
    quarter = np.array([[[0, -1], [1, 0]]])
    assert probe_mixing.row_class(quarter, 0).tolist() == [3]       # x gets -y: source 1, minus
    assert probe_mixing.row_class(quarter, 1).tolist() == [0]       # y gets +x: source 0, plus
    identity = np.eye(2, dtype=np.int64)[None]
    assert probe_mixing.row_class(identity, 0).tolist() == [0] and probe_mixing.row_class(identity, 1).tolist() == [2]

def test_the_row_target_gives_the_next_velocity_from_the_last_one():
    model, X, R = setup()
    ts = list(range(2, STEPS - 1))
    for axis in range(DIM):
        _, tg = probe_mixing.collect(model, X, R, TS.group, DIM, SIZE, STEPS, ts, axis)
        source, minus = tg["row"] // 2, tg["row"] % 2
        others = [probe_mixing.collect(model, X, R, TS.group, DIM, SIZE, STEPS, ts, a)[1]["last velocity"] for a in range(DIM)]
        fed = np.choose(source, others)
        assert np.array_equal(tg["next velocity"], np.where(minus == 1, -fed, fed) % SIZE)
