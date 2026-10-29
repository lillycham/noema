"""The cellular automaton. The first group checks reference_step() against rules whose behaviour is
known, and needs nothing from you. The second checks your step() against reference_step().

    pytest tests/test_ca.py
"""
import numpy as np

from noema import ca

W = 15

def random_rows(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=shape, dtype=np.uint8)

def single_one(width=W):
    row = np.zeros(width, dtype=np.uint8)
    row[width // 2] = 1
    return row

def ones_at(row):
    return set(np.flatnonzero(row).tolist())

# ---------- the reference ----------
def test_reference_rule_204_changes_nothing():
    """204 = 0b11001100: the new cell is the cell itself."""
    row = random_rows(W)
    assert np.array_equal(ca.reference_step(row, 204), row)

def test_reference_rule_170_shifts_left():
    """170 = 0b10101010: the new cell is its right neighbour, so the pattern moves left."""
    row = random_rows(W)
    assert np.array_equal(ca.reference_step(row, 170), np.roll(row, -1))

def test_reference_rule_240_shifts_right():
    """240 = 0b11110000: the new cell is its left neighbour, so the pattern moves right."""
    row = random_rows(W)
    assert np.array_equal(ca.reference_step(row, 240), np.roll(row, 1))

def test_reference_rule_0_kills_and_255_fills():
    row = random_rows(W)
    assert not ca.reference_step(row, 0).any()
    assert ca.reference_step(row, 255).all()

def test_reference_rule_90_draws_a_sierpinski_triangle():
    """90: left XOR right. From one 1 in the middle (cell 7), rows 2 and 3 are known by hand."""
    rows = ca.rollout(single_one(), 90, 4, ca.reference_step)
    assert ones_at(rows[0]) == {7}
    assert ones_at(rows[1]) == {6, 8}
    assert ones_at(rows[2]) == {5, 9}
    assert ones_at(rows[3]) == {4, 6, 8, 10}

def test_reference_gives_every_row_its_own_rule():
    rows = np.stack([random_rows(W, 1)] * 3)
    out = ca.reference_step(rows, np.array([0, 255, 204]))
    assert not out[0].any() and out[1].all() and np.array_equal(out[2], rows[2])

def test_rule_110_and_its_three_twins():
    assert ca.mirror(110) == 124 and ca.complement(110) == 137
    assert ca.equivalent_rules(110) == {110, 124, 137, 193}

def test_mirror_and_complement_undo_themselves():
    for rule in range(256):
        assert ca.mirror(ca.mirror(rule)) == rule
        assert ca.complement(ca.complement(rule)) == rule

def test_the_256_rules_fall_into_88_classes():
    assert len({frozenset(ca.equivalent_rules(r)) for r in range(256)}) == 88

def test_a_mirror_rule_makes_the_mirror_world():
    """Run rule 110 on a row, and rule 124 on the reversed row: the results are reversed copies."""
    row = random_rows(W)
    assert np.array_equal(ca.reference_step(row, 110)[::-1], ca.reference_step(row[::-1], 124))

def test_a_complement_rule_makes_the_inverted_world():
    row = random_rows(W)
    assert np.array_equal(1 - ca.reference_step(row, 110), ca.reference_step(1 - row, 137))

def test_rollout_shape_and_first_row():
    first = random_rows((4, W))
    rows = ca.rollout(first, np.array([30, 90, 110, 184]), 6, ca.reference_step)
    assert rows.shape == (4, 6, W)
    assert np.array_equal(rows[:, 0], first)

# ---------- your step() ----------
def test_step_matches_reference_for_every_rule():
    """All 256 rules, a batch of random rows for each."""
    for rule in range(256):
        rows = random_rows((5, W), seed=rule)
        assert np.array_equal(ca.step(rows, rule), ca.reference_step(rows, rule)), f"rule {rule}"

def test_step_with_one_rule_for_each_row_of_a_batch():
    rows = random_rows((4, 3, W))
    rules = np.random.default_rng(1).integers(0, 256, size=(4, 3))
    assert np.array_equal(ca.step(rows, rules), ca.reference_step(rows, rules))

def test_step_wraps_round_the_edge():
    """A 1 in cell 0 under rule 170 (shift left) comes out in the last cell."""
    row = np.zeros(W, dtype=np.uint8)
    row[0] = 1
    assert ones_at(ca.step(row, 170)) == {W - 1}

def test_step_draws_a_sierpinski_triangle():
    rows = ca.rollout(single_one(), 90, 4)
    assert ones_at(rows[3]) == {4, 6, 8, 10}
