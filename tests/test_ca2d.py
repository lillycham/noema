"""The 2D automaton. The first groups check reference_step() and the symmetries. The last group
checks step() against reference_step().

    pytest tests/test_ca2d.py
"""
import numpy as np

from noema import ca, ca2d

H, W = 6, 7                         # not square, so a mix-up of rows and columns shows

def random_grids(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=shape, dtype=np.uint8)

def copy_of(weight):
    """The rule whose new cell is the neighbour with this weight in n: 16 up, 8 left, 4 itself, 2 right, 1 down."""
    return sum(1 << n for n in range(32) if n & weight)

# ---------- the reference ----------
def test_reference_rule_that_copies_itself_changes_nothing():
    g = random_grids((H, W))
    assert np.array_equal(ca2d.reference_step(g, copy_of(4)), g)

def test_reference_copying_a_neighbour_moves_the_picture_the_other_way():
    """New cell = the cell above it: everything moves down a row, and the edge wraps."""
    g = random_grids((H, W))
    assert np.array_equal(ca2d.reference_step(g, copy_of(16)), np.roll(g, 1, axis=0))
    assert np.array_equal(ca2d.reference_step(g, copy_of(1)), np.roll(g, -1, axis=0))
    assert np.array_equal(ca2d.reference_step(g, copy_of(8)), np.roll(g, 1, axis=1))
    assert np.array_equal(ca2d.reference_step(g, copy_of(2)), np.roll(g, -1, axis=1))

def test_reference_rule_0_kills_and_all_ones_fills():
    g = random_grids((H, W))
    assert not ca2d.reference_step(g, 0).any()
    assert ca2d.reference_step(g, 2 ** 32 - 1).all()

def test_reference_neighbourhood_is_read_in_the_order_of_a_page():
    """One cell set in a grid of zeros. Its right neighbour sees it as 'left' (weight 8): n = 8."""
    g = np.zeros((H, W), dtype=np.uint8)
    g[2, 3] = 1
    out = ca2d.reference_step(g, 1 << 8)                      # only n = 8 gives 1
    assert np.argwhere(out).tolist() == [[2, 4]]
    out = ca2d.reference_step(g, 1 << 1)                      # only n = 1 (down is set): the cell above it
    assert np.argwhere(out).tolist() == [[1, 3]]

def test_reference_gives_every_grid_its_own_rule():
    g = np.stack([random_grids((H, W), 1)] * 3)
    out = ca2d.reference_step(g, np.array([0, 2 ** 32 - 1, copy_of(4)]))
    assert not out[0].any() and out[1].all() and np.array_equal(out[2], g[2])

def test_rollout_stacks_the_frames():
    frames = ca2d.rollout(random_grids((2, H, W)), np.array([copy_of(16), copy_of(2)]), 5, ca2d.reference_step)
    assert frames.shape == (2, 5, H, W)
    assert np.array_equal(frames[0, 3], np.roll(frames[0, 0], 3, axis=0))

# ---------- the symmetries ----------
def test_there_are_eight_different_symmetries():
    assert len({tuple(m) for m in ca2d.NEIGHBOURHOOD_MAPS}) == 8
    assert all(sorted(m) == list(range(32)) for m in ca2d.NEIGHBOURHOOD_MAPS)       # each only shuffles the 32 numbers

def test_turn_back_undoes_turn():
    g = random_grids((3, H, W))
    for k, flip in ca2d.SYMMETRIES:
        assert np.array_equal(ca2d.turn_back(ca2d.turn(g, k, flip), k, flip), g)

def test_a_turned_world_follows_the_turned_rule():
    """The definition of transform(): step, then turn, is turn, then step with the turned rule."""
    rng = np.random.default_rng(0)
    for rule in rng.integers(0, 2 ** 32, size=4).tolist():
        g = random_grids((H, H), seed=rule)                        # square, so a quarter turn fits
        for s, (k, flip) in enumerate(ca2d.SYMMETRIES):
            want = ca2d.turn(ca2d.reference_step(g, rule), k, flip)
            got = ca2d.reference_step(ca2d.turn(g, k, flip), ca2d.transform(rule, s))
            assert np.array_equal(got, want), f"rule {rule}, symmetry {k, flip}"

def test_the_identity_symmetry_leaves_the_rule_alone():
    assert ca2d.SYMMETRIES[0] == (0, False)
    assert ca2d.transform(0x9E3779B9, 0) == 0x9E3779B9

def test_copying_the_cell_above_becomes_copying_another_neighbour_when_turned():
    """The four neighbours are interchangeable by turning: all of 'copy a neighbour' are one class."""
    cls = ca2d.equivalent_rules(copy_of(16))
    assert {copy_of(16), copy_of(8), copy_of(2), copy_of(1)} <= cls

def test_a_class_has_up_to_16_rules_and_contains_its_own_complement():
    rng = np.random.default_rng(1)
    for rule in rng.integers(0, 2 ** 32, size=20).tolist():
        cls = ca2d.equivalent_rules(rule)
        assert rule in cls and ca.complement(rule, 2) in cls and len(cls) <= 16
        assert all(ca2d.equivalent_rules(other) == cls for other in cls)       # every member gives the same class

# ---------- step() ----------
def test_step_matches_the_reference_on_a_batch_with_a_rule_each():
    g = random_grids((5, H, W))
    rules = np.random.default_rng(2).integers(0, 2 ** 32, size=5)
    assert np.array_equal(ca2d.step(g, rules), ca2d.reference_step(g, rules))

def test_step_matches_the_reference_on_one_grid_with_one_rule():
    g = random_grids((H, W), 3)
    assert np.array_equal(ca2d.step(g, 0x9E3779B9), ca2d.reference_step(g, 0x9E3779B9))

def test_step_moves_a_picture_when_the_rule_copies_a_neighbour():
    g = random_grids((H, W), 4)
    for weight, shift, axis in ((16, 1, 0), (1, -1, 0), (8, 1, 1), (2, -1, 1)):
        assert np.array_equal(ca2d.step(g, copy_of(weight)), np.roll(g, shift, axis=axis))

def test_step_gives_a_whole_batch_of_frames():
    g = random_grids((2, 3, H, W), 5)                          # two batch axes
    rules = np.arange(6, dtype=np.int64).reshape(2, 3) * 123456789
    assert ca2d.step(g, rules).shape == g.shape
    assert np.array_equal(ca2d.step(g, rules), ca2d.reference_step(g, rules))
