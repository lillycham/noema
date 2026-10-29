import numpy as np
import torch

from noema import ca, data, evaluate, gpt, tokens

W, R = data.WIDTH, data.ROWS
L = R * (W + 1)

def batch(B=6, rules=(30, 90, 110, 184)):
    X, _ = data.rollout_batch(np.random.default_rng(0), B, list(rules), step_fn=ca.reference_step)
    return torch.from_numpy(X)

def untrained():
    torch.manual_seed(0)
    return gpt.GPT(gpt.Config(vocab=tokens.VOCAB, block=L, emb=32, heads=2, layers=2)).eval()

class Oracle(torch.nn.Module):
    """A model that always knows the next token: its logits are the answer, one-hot."""

    def __init__(self, X):
        super().__init__()
        self.X = X
        self.cfg = gpt.Config(vocab=tokens.VOCAB, block=L)

    def forward(self, tokens_so_far):
        T = tokens_so_far.shape[1]
        answer = self.X[:, 1:T + 1] if T < L else torch.cat([self.X[:, 1:], self.X[:, :1]], dim=1)
        return torch.nn.functional.one_hot(answer, tokens.VOCAB).float() * 50

def test_score_gives_a_loss_and_an_accuracy_for_each_row_after_the_first():
    X, mask = batch(), torch.from_numpy(tokens.target_mask(L, W))
    loss, by_row = evaluate.score(untrained(), X, mask, W)
    assert loss > 0 and by_row.shape == (R - 1,)
    assert ((0 <= by_row) & (by_row <= 1)).all()

def test_a_model_that_knows_the_answer_scores_perfectly():
    X, mask = batch(), torch.from_numpy(tokens.target_mask(L, W))
    loss, by_row = evaluate.score(Oracle(X), X, mask, W)
    assert loss < 1e-3 and np.allclose(by_row, 1.0)

def test_complete_keeps_the_prefix_and_fills_the_block_with_whole_rows():
    X = batch()
    out = evaluate.complete(untrained(), X[:, :3 * (W + 1)], W)
    assert out.shape == X.shape
    assert torch.equal(out[:, :3 * (W + 1)], X[:, :3 * (W + 1)])
    assert tokens.decode(out.numpy(), W).shape == (6, R, W)      # separators all in place: decode would raise

def test_complete_only_writes_cells_and_is_the_same_every_time_when_greedy():
    X = batch()
    a = evaluate.complete(untrained(), X[:, :W + 1], W)
    b = evaluate.complete(untrained(), X[:, :W + 1], W)
    assert torch.equal(a, b)
    assert set(a.unique().tolist()) <= {tokens.ZERO, tokens.ONE, tokens.SEP}

def test_a_model_that_knows_the_answer_completes_the_true_rollout():
    X = batch()
    assert torch.equal(evaluate.complete(Oracle(X), X[:, :2 * (W + 1)], W), X)
