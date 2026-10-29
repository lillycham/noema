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

# ---------- the ideal table-lookup learner ----------
def test_neighbourhood_numbers_are_the_ones_the_reference_step_reads():
    row = np.array([[1, 0, 0, 1, 1, 0, 1, 0]])
    n1 = evaluate.neighbourhoods(row, 1)
    assert n1[0, 0] == 4 * row[0, -1] + 2 * row[0, 0] + row[0, 1]               # wraps round the edge
    n2 = evaluate.neighbourhoods(row, 2)
    assert n2.min() >= 0 and n2.max() < 32
    assert n2[0, 3] == 16 * row[0, 1] + 8 * row[0, 2] + 4 * row[0, 3] + 2 * row[0, 4] + row[0, 5]

def test_neighbourhood_numbers_reproduce_the_rule():
    """Reading bit n of the rule at the neighbourhood number is the step itself."""
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2, size=(5, W), dtype=np.uint8)
    for rule in (0x9E3779B9, 0x1234ABCD):
        want = ca.reference_step(rows, rule, 2)
        assert np.array_equal((rule >> evaluate.neighbourhoods(rows, 2)) & 1, want)

def test_ideal_learner_on_an_all_zero_world_has_one_unknown_cell_then_none():
    """Every neighbourhood is 0. In the first row it knows the cell after the first one: 14 of 15 cells."""
    grid = np.zeros((4, R, W), dtype=np.uint8)
    ideal = evaluate.ideal_accuracy_by_row(grid, 1)
    assert ideal.shape == (R - 1,)
    assert ideal[0] == np.float64(0.5 + 0.5 * 14 / 15)
    assert (ideal[1:] == 1.0).all()

def test_ideal_learner_on_random_radius_2_rules_climbs_with_the_rows():
    rs = data.RuleSet(2)
    X, _ = data.rollout_batch(np.random.default_rng(0), 64, rs.train, step_fn=ca.reference_step, radius=2)
    ideal = evaluate.ideal_accuracy_by_row(tokens.decode(X, W), 2)
    assert (ideal >= 0.5).all() and (ideal <= 1.0).all()
    assert ideal[0] < 0.75 and ideal[-1] > 0.9 and ideal[-1] > ideal[0] + 0.2
    assert (np.diff(ideal) > -0.02).all()                       # the rows never tell it less than before
