import numpy as np

from noema import ca, data, tokens

W, R = data.WIDTH, data.ROWS
SLOW = ca.reference_step

def batch(seed=0, B=8, rules=range(256)):
    return data.rollout_batch(np.random.default_rng(seed), B, rules, step_fn=SLOW)

def test_a_batch_is_whole_rollouts_of_tokens():
    X, rules = batch()
    assert X.shape == (8, R * (W + 1)) and X.dtype == np.int64
    assert rules.shape == (8,)
    assert X.min() >= 0 and X.max() < tokens.VOCAB

def test_the_same_seed_gives_the_same_batch():
    (A, a), (B, b) = batch(3), batch(3)
    assert np.array_equal(A, B) and np.array_equal(a, b)

def test_rules_come_only_from_the_allowed_ones():
    _, rules = batch(B=64, rules=[30, 90])
    assert set(rules.tolist()) <= {30, 90}

def test_each_row_follows_from_the_one_before_by_its_rule():
    """The data really is the automaton: decode, and step each row by hand."""
    X, rules = batch()
    rows = tokens.decode(X, W)
    for i in range(len(X)):
        for r in range(R - 1):
            assert np.array_equal(rows[i, r + 1], ca.reference_step(rows[i, r], int(rules[i])))

def test_first_rows_are_random_and_different():
    X, _ = batch(B=16)
    firsts = {tuple(row[:W]) for row in X}
    assert len(firsts) > 12

def test_the_held_out_rules_are_never_trained_on():
    train, held = data.split_rules(32)
    assert 32 <= len(held) < 36 and len(train) + len(held) == 256
    assert not set(train.tolist()) & set(held.tolist())
    assert sorted(train.tolist() + held.tolist()) == list(range(256))

def test_no_held_out_rule_has_a_mirror_or_complement_in_training():
    """Rule 110 and its twins 124, 137 and 193 are one automaton in four disguises. A split by rule
    number would hold out 110 and train on 124, and test nothing."""
    train, held = data.split_rules(32)
    for rule in held.tolist():
        assert not ca.equivalent_rules(rule) & set(train.tolist()), f"rule {rule}"

def test_the_split_is_the_same_every_time():
    (a, b), (c, d) = data.split_rules(32), data.split_rules(32)
    assert np.array_equal(a, c) and np.array_equal(b, d)
