"""Rules that count: a task where the ideal table-lookup learner is not the ceiling.

A rule at radius 2 is a table of 32 entries, and on random tables no learner can beat
evaluate.ideal_accuracy_by_row: an entry it has not met is a coin flip. A totalistic rule looks only at
how many of the 5 cells are alive, so it is a table of 6 entries, one for each count, spread over the 32
neighbourhoods. There are 64 such rules. A learner that knows this has met an entry once it has met any
neighbourhood with the same count, so it knows more than the table learner does, and sooner. A model
above the table curve has learned to count. The ceiling is count_ideal below.

Whole classes are held out, 1 rule in 4 about: a rule and its complement twin are in the same class
(the mirror of a totalistic rule is the rule itself).
"""
import numpy as np

from noema import evaluate

RADIUS = 2
CELLS = 2 * RADIUS + 1
RULES = 2 ** (CELLS + 1)

def rule_number(table):
    """The radius-2 rule number of a table: table[k] (0 or 1) is the new cell when k of the 5 cells are alive."""
    return sum(int(table[bin(n).count("1")]) << n for n in range(2 ** CELLS))

def table_of(index):
    """The table (CELLS + 1 bits) of the rule with this index in 0 .. RULES - 1."""
    return [(index >> k) & 1 for k in range(CELLS + 1)]

def complement_index(index):
    """The rule with 0 and 1 swapped: alive cells become dead, and so does the answer."""
    return sum((1 - ((index >> (CELLS - k)) & 1)) << k for k in range(CELLS + 1))

class CountRuleSet:
    """Like data.RuleSet: train() and held() give rule numbers, is_held_out() says which is which."""

    def __init__(self, seed=0):
        classes = sorted({tuple(sorted((i, complement_index(i)))) for i in range(RULES)})
        order = np.random.default_rng(seed).permutation(len(classes))
        held, train, want = [], [], RULES // 4
        for c in order:
            (held if len(held) < want else train).extend(set(classes[c]))     # a rule can be its own twin
        self.train_rules = np.array([rule_number(table_of(i)) for i in train], dtype=np.int64)
        self.held_rules = np.array([rule_number(table_of(i)) for i in held], dtype=np.int64)

    def is_held_out(self, rule):
        return int(rule) in set(self.held_rules.tolist())

    def train(self, rng, B):
        return rng.choice(self.train_rules, size=B)

    def held(self, rng, B):
        return rng.choice(self.held_rules, size=B)

def count_ideal(grid):
    """The accuracy by row of a learner that knows rules count: certain once it has met this cell's count
    in the sequence, else a coin flip. grid (B, n, W). The ceiling for a table of 6 fair coins."""
    return count_ideal_from(evaluate.neighbourhoods(np.asarray(grid)[:, :-1], RADIUS))

def count_ideal_from(numbers):
    """count_ideal for any world, from the neighbourhood numbers (B, T, N), as evaluate.ideal_accuracy_from."""
    return evaluate.ideal_accuracy_from(np.vectorize(lambda n: bin(n).count("1"))(numbers))
