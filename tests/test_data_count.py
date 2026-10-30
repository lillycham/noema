"""Rules that count.

    pytest tests/test_data_count.py
"""
import numpy as np

from noema import ca, data, data_count as dc, evaluate, tokens

def test_a_counting_rule_gives_the_same_answer_for_every_neighbourhood_with_one_count():
    table = dc.table_of(0b101101)
    number = dc.rule_number(table)
    for n in range(32):
        assert (number >> n) & 1 == table[bin(n).count("1")]

def test_there_are_64_different_rules():
    assert len({dc.rule_number(dc.table_of(i)) for i in range(dc.RULES)}) == 64

def test_a_rule_and_its_complement_twin_are_true_twins():
    i = 0b011010
    rule, twin = dc.rule_number(dc.table_of(i)), dc.rule_number(dc.table_of(dc.complement_index(i)))
    first = np.random.default_rng(0).integers(0, 2, size=(1, 15), dtype=np.uint8)
    next_row = ca.reference_step(first, np.array([rule]), 2)
    other = ca.reference_step(1 - first, np.array([twin]), 2)
    assert (next_row == 1 - other).all()

def test_the_split_is_disjoint_and_holds_whole_classes():
    s = dc.CountRuleSet()
    assert not set(s.train_rules.tolist()) & set(s.held_rules.tolist())
    assert len(s.train_rules) + len(s.held_rules) == 64
    for i in range(dc.RULES):
        r, t = dc.rule_number(dc.table_of(i)), dc.rule_number(dc.table_of(dc.complement_index(i)))
        assert s.is_held_out(r) == s.is_held_out(t)

def test_the_count_ideal_is_above_the_table_ideal_and_reaches_one_sooner():
    s = dc.CountRuleSet()
    X, _ = data.rollout_batch(np.random.default_rng(1), 64, s.train, radius=2)
    grid = tokens.decode(X, data.WIDTH)
    table, count = evaluate.ideal_accuracy_by_row(grid, 2), dc.count_ideal(grid)
    assert (count >= table - 1e-9).all() and count[0] > table[0] + 0.1 and count[2] > 0.98

def test_counting_rules_run_on_the_2d_grid_as_the_same_table():
    from noema import ca2d
    i = 0b101100
    number = dc.rule_number(dc.table_of(i))
    grid = np.random.default_rng(2).integers(0, 2, size=(4, 5, 5), dtype=np.uint8)
    out = ca2d.reference_step(grid, np.full(4, number))
    counts = np.vectorize(lambda n: bin(n).count("1"))(ca2d.neighbourhoods(grid))
    assert (out == np.array(dc.table_of(i))[counts]).all()
