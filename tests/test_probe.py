"""The probes. A probe on features that hold the answer finds it; on noise it does not.

    pytest tests/test_probe.py
"""
import numpy as np
import torch

from noema import data_torus as dt, gpt, probe

def model_and_batch(B=8):
    X, _ = dt.trajectory_batch(np.random.default_rng(0), B, dt.VelocitySet().train)
    model = gpt.GPT(gpt.Config(vocab=dt.VOCAB, block=X.shape[1], emb=16, heads=2, layers=3))
    return model, X

def test_residuals_has_one_entry_before_the_blocks_and_one_after_each():
    model, X = model_and_batch()
    s = probe.residuals(model, torch.from_numpy(X))
    assert s.shape == (4, 8, X.shape[1], 16)

def test_the_last_entry_is_what_the_output_layer_reads():
    model, X = model_and_batch()
    s = probe.residuals(model, torch.from_numpy(X))
    logits = model.out(model.norm(torch.from_numpy(s[-1]))).detach().numpy()
    assert np.allclose(logits, model(torch.from_numpy(X)).detach().numpy(), atol=1e-5)

def test_axis_0_is_read_at_the_separator_and_axis_1_one_token_later():
    s = np.arange(2 * 1 * 12 * 1).reshape(2, 1, 12, 1)
    got = probe.at_queries(s[:, :, :11], dim=2, ts=[0, 1, 2], axis=0)
    assert got.shape == (2, 1, 3, 1) and got[0, 0, :, 0].tolist() == [2, 5, 8]
    assert probe.at_queries(s[:, :, :12], dim=2, ts=[0, 1, 2], axis=1)[0, 0, :, 0].tolist() == [3, 6, 9]

def test_a_probe_reads_a_label_that_is_in_the_features_and_not_one_that_is_not():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=3000)
    F = np.eye(11)[y] + 0.1 * rng.normal(size=(3000, 11))
    assert probe.probe_accuracy(F[:2000], y[:2000], F[2000:], y[2000:], 11) > 0.95
    noise = rng.normal(size=(3000, 11))
    assert probe.probe_accuracy(noise[:2000], y[:2000], noise[2000:], y[2000:], 11) < 0.2

def test_fourier_share_is_flat_for_a_one_hot_code_and_peaked_for_a_circular_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 11, size=4000)
    one_hot = np.eye(11)[y] + 0.05 * rng.normal(size=(4000, 11))
    s = probe.fourier_share(one_hot, y, 11)
    assert s.shape == (5,) and abs(s.sum() - 1) < 1e-9 and s.max() < 0.3
    angle = 2 * np.pi * 2 * y / 11
    circle = np.stack([np.cos(angle), np.sin(angle)], -1) + 0.05 * rng.normal(size=(4000, 2))
    c = probe.fourier_share(circle, y, 11)
    assert c.argmax() == 1 and c[1] > 0.95

def test_targets_are_the_position_the_velocity_and_the_next_position_on_each_axis():
    model, X = model_and_batch(B=4)
    for axis in range(dt.DIM):
        F, tg = probe.collect(model, X, dt.DIM, dt.SIZE, dt.STEPS, axis)
        assert F.shape[0] == 4 and F.shape[1] == 4 * (dt.STEPS - 3)
        assert np.array_equal((tg["position"] + tg["velocity"]) % dt.SIZE, tg["next"])
