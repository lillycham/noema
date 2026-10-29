import numpy as np
import torch

from noema import ca2d, data2d, evaluate, gpt, render, tokens
from noema.train_eca import train_step

H, W, F = data2d.HEIGHT, data2d.WIDTH, data2d.FRAMES
CELLS = H * W
SLOW = ca2d.reference_step

def batch(seed=0, B=6, rules=None):
    rules = rules if rules is not None else data2d.RuleSet().train
    return data2d.rollout_batch(np.random.default_rng(seed), B, rules, step_fn=SLOW)

def test_a_batch_is_whole_rollouts_of_tokens_with_one_separator_a_frame():
    X, rules = batch()
    assert X.shape == (6, F * (CELLS + 1)) and X.dtype == np.int64 and rules.shape == (6,)
    assert (X.reshape(6, F, CELLS + 1)[:, :, CELLS] == tokens.SEP).all()
    assert X.min() >= 0 and X.max() < tokens.VOCAB

def test_the_same_seed_gives_the_same_batch():
    (A, a), (B, b) = batch(3), batch(3)
    assert np.array_equal(A, B) and np.array_equal(a, b)

def test_each_frame_follows_from_the_one_before_by_its_rule():
    X, rules = batch()
    grids = data2d.to_frames(X)
    assert grids.shape == (6, F, H, W)
    for i in range(6):
        for t in range(F - 1):
            assert np.array_equal(grids[i, t + 1], SLOW(grids[i, t], int(rules[i])))

def test_a_rule_that_copies_the_cell_above_moves_every_row_down_one():
    X, _ = batch(rules=[0xFFFF0000], B=2)                       # new cell = the cell above
    g = data2d.to_frames(X)
    assert np.array_equal(g[:, 1], np.roll(g[:, 0], 1, axis=1))

def test_first_grids_are_random_and_different():
    X, _ = batch(B=16)
    assert len({tuple(row[:CELLS]) for row in X}) > 12

# ---------- the rules ----------
def test_train_rules_are_never_held_out_and_held_ones_always_are():
    rs, rng = data2d.RuleSet(), np.random.default_rng(0)
    assert not any(rs.is_held_out(r) for r in rs.train(rng, 200).tolist())
    assert all(rs.is_held_out(r) for r in rs.held(rng, 30).tolist())

def test_rules_are_32_bit_numbers():
    rules = data2d.RuleSet().train(np.random.default_rng(0), 300)
    assert rules.min() >= 0 and rules.max() < 2 ** 32 and rules.max() > 2 ** 30

def test_a_class_is_held_out_together_whatever_turn_or_reflection():
    rs, rng = data2d.RuleSet(), np.random.default_rng(1)
    for rule in rng.integers(0, 2 ** 32, size=60).tolist():
        assert len({rs.is_held_out(twin) for twin in ca2d.equivalent_rules(rule)}) == 1

def test_about_one_rule_in_sixteen_is_held_out():
    rs = data2d.RuleSet()
    rules = np.random.default_rng(2).integers(0, 2 ** 32, size=1500).tolist()
    assert 0.03 < sum(rs.is_held_out(r) for r in rules) / len(rules) < 0.10

# ---------- the ideal learner ----------
def test_neighbourhood_numbers_reproduce_the_rule():
    grids = np.random.default_rng(0).integers(0, 2, size=(3, H, W), dtype=np.uint8)
    for rule in (0x9E3779B9, 0x1234ABCD):
        nb = ca2d.neighbourhoods(grids)
        assert nb.min() >= 0 and nb.max() < 32
        assert np.array_equal((rule >> nb) & 1, SLOW(grids, rule))

def test_ideal_learner_on_an_all_zero_world_has_one_unknown_cell_then_none():
    ideal = data2d.ideal_accuracy_by_frame(np.zeros((4, F, H, W), dtype=np.uint8))
    assert ideal.shape == (F - 1,)
    assert ideal[0] == np.float64(0.5 + 0.5 * (CELLS - 1) / CELLS) and (ideal[1:] == 1.0).all()

def test_ideal_learner_on_random_rules_climbs_with_the_frames():
    X, _ = batch(B=48)
    ideal = data2d.ideal_accuracy_by_frame(data2d.to_frames(X))
    assert ideal[0] < 0.75 and ideal[-1] > 0.95 and (np.diff(ideal) > -0.02).all()

def test_the_1d_ideal_learner_is_unchanged_by_the_refactor():
    grid = np.zeros((4, 16, 15), dtype=np.uint8)
    assert evaluate.ideal_accuracy_by_row(grid, 1)[0] == np.float64(0.5 + 0.5 * 14 / 15)

# ---------- the model, the picture, the script ----------
def test_a_few_steps_on_a_moving_world_bring_the_loss_down():
    torch.manual_seed(0)
    rng, side, frames = np.random.default_rng(0), 3, 4
    L = frames * (side * side + 1)
    model = gpt.GPT(gpt.Config(vocab=tokens.VOCAB, block=L, emb=48, heads=2, layers=2))
    opt = torch.optim.AdamW(gpt.param_groups(model, 0.1), lr=3e-3, betas=(0.9, 0.95))
    mask = torch.from_numpy(tokens.target_mask(L, side * side))
    losses = []
    for _ in range(150):                                      # two rules: it has to tell which one from the frames
        X, _ = data2d.rollout_batch(rng, 32, [0xF0F0F0F0, 0xFFFF0000], side, side, frames, step_fn=SLOW)
        losses.append(train_step(model, opt, torch.from_numpy(X), mask, 3e-3))
    assert np.mean(losses[:5]) > 0.6 and np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])

def test_the_picture_of_worlds_is_written(tmp_path):
    g = np.random.default_rng(0).integers(0, 2, size=(F, H, W), dtype=np.uint8)
    render.save_worlds([(1, g, g, "seen"), (2, g, 1 - g, "held out")], tmp_path / "w.png", 2)
    assert (tmp_path / "w.png").stat().st_size > 1000

def test_the_training_script_runs_and_saves_its_outputs(tmp_path, monkeypatch):
    from noema import train_2d
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_2d", "--steps", "4", "--log-every", "2", "--emb", "32", "--heads", "2",
                                     "--layers", "2", "--side", "3", "--frames", "4", "--device", "cpu", "--tag", "t"])
    train_2d.main()
    assert (tmp_path / "runs/world2d_t_rollouts.png").exists() and (tmp_path / "checkpoints/world2d_t.pt").exists()
