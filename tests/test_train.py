import numpy as np
import torch

from noema import ca, data, gpt, tokens
from noema.train_eca import lr_at, train_step

W, R = data.WIDTH, data.ROWS
L = R * (W + 1)

def test_the_learning_rate_warms_up_then_falls_to_a_tenth():
    peak, steps, warmup = 1.0, 1000, 100
    assert lr_at(0, steps, peak, warmup) < 0.02
    assert lr_at(99, steps, peak, warmup) == peak
    assert lr_at(steps, steps, peak, warmup) == 0.1 * peak
    assert 0.1 < lr_at(550, steps, peak, warmup) < 1.0

def test_a_few_steps_on_easy_rules_bring_the_loss_down():
    """Rules 204 (copy the cell above), 170 and 240 (shift): any working model learns these fast."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = gpt.GPT(gpt.Config(vocab=tokens.VOCAB, block=L, emb=48, heads=2, layers=2))
    opt = torch.optim.AdamW(gpt.param_groups(model, 0.1), lr=3e-3, betas=(0.9, 0.95))
    mask = torch.from_numpy(tokens.target_mask(L, W))
    losses = []
    for step in range(60):
        X, _ = data.rollout_batch(rng, 32, [204, 170, 240], step_fn=ca.reference_step)
        losses.append(train_step(model, opt, torch.from_numpy(X), mask, 3e-3))
    assert np.mean(losses[:5]) > 0.6                      # starts near ln 2 = 0.69
    assert np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])
