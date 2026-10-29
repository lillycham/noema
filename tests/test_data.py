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

# ---------- radius 2: too many rules to list ----------
def test_radius_1_rule_set_is_the_explicit_split():
    rs = data.RuleSet(1, 32)
    train, held = data.split_rules(32)
    assert set(rs.train(np.random.default_rng(0), 500).tolist()) <= set(train.tolist())
    assert set(rs.held(np.random.default_rng(0), 500).tolist()) <= set(held.tolist())

def test_radius_2_train_rules_are_never_held_out_and_held_ones_always_are():
    rs, rng = data.RuleSet(2), np.random.default_rng(0)
    assert not any(rs.is_held_out(r) for r in rs.train(rng, 300).tolist())
    assert all(rs.is_held_out(r) for r in rs.held(rng, 40).tolist())

def test_radius_2_rules_are_32_bit_numbers():
    rs, rng = data.RuleSet(2), np.random.default_rng(0)
    rules = rs.train(rng, 300)
    assert rules.min() >= 0 and rules.max() < 2 ** 32 and rules.max() > 2 ** 30      # they use the whole range

def test_radius_2_twins_are_held_out_together():
    """The hash is of the class, so a rule and its mirror and complement are all in or all out."""
    rs, rng = data.RuleSet(2), np.random.default_rng(1)
    for rule in rng.integers(0, 2 ** 32, size=200).tolist():
        assert len({rs.is_held_out(twin) for twin in ca.equivalent_rules(rule, 2)}) == 1

def test_about_one_radius_2_rule_in_sixteen_is_held_out():
    rs = data.RuleSet(2)
    rules = np.random.default_rng(2).integers(0, 2 ** 32, size=3000).tolist()
    share = sum(rs.is_held_out(r) for r in rules) / len(rules)
    assert 0.03 < share < 0.10

def test_the_seed_changes_which_radius_2_rules_are_held_out():
    a, b = data.RuleSet(2, seed=0), data.RuleSet(2, seed=1)
    rules = np.random.default_rng(3).integers(0, 2 ** 32, size=500).tolist()
    assert [a.is_held_out(r) for r in rules] != [b.is_held_out(r) for r in rules]

def test_a_radius_2_batch_follows_the_radius_2_rule():
    rs = data.RuleSet(2)
    X, rules = data.rollout_batch(np.random.default_rng(0), 3, rs.train, step_fn=ca.reference_step, radius=2)
    rows = tokens.decode(X, W)
    for i in range(3):
        for r in range(R - 1):
            assert np.array_equal(rows[i, r + 1], ca.reference_step(rows[i, r], int(rules[i]), 2))
