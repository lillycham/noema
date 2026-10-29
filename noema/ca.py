"""Elementary cellular automata: a row of cells, a rule number from 0 to 255, and edges that wrap round.

Wolfram's numbering. A cell's next state depends on its neighbourhood (left, itself, right), which
is a 3-bit number n = 4*left + 2*itself + right, from 0 to 7. A rule is 8 bits, one for each n:
the next state of a cell with neighbourhood n is bit n of the rule number.

    rule 204 = 0b11001100   n = 2, 3, 6, 7 give 1 (the cell itself is 1): nothing ever changes
    rule 170 = 0b10101010   n odd gives 1 (the right neighbour is 1): the row shifts left
    rule 90  = 0b01011010   left XOR right: from a single 1, a Sierpinski triangle

The step is the one function in this file that is yours to write. reference_step() below is the
slow, obviously correct version, one cell at a time. tests/test_ca.py checks yours against it for
all 256 rules.

Rows are numpy arrays of 0 and 1 with the cells on the last axis, and any axes in front are a
batch. In a batch, every row can have its own rule.
"""
import numpy as np

def step(rows, rules):
    left = np.roll(rows, 1, axis=-1)
    right = np.roll(rows, -1, axis=-1)
    n = 4 * left + 2 * rows + right
    return (np.asarray(rules)[..., None] >> n) & 1

def reference_step(rows, rules):
    """The slow version of step(): one cell at a time, with the definition written out."""
    rows = np.asarray(rows)
    rules = np.broadcast_to(np.asarray(rules), rows.shape[:-1])
    out = np.empty_like(rows)
    for at in np.ndindex(rows.shape[:-1]):
        row, rule = rows[at], int(rules[at])
        W = len(row)
        for i in range(W):
            n = 4 * int(row[i - 1]) + 2 * int(row[i]) + int(row[(i + 1) % W])
            out[at][i] = (rule >> n) & 1
    return out

def mirror(rule):
    """The rule for the left-right reflected world: the bit for neighbourhood (l, c, r) moves to the
    one for (r, c, l). Rule 110 becomes 124. Doing it twice gives the rule back."""
    out = 0
    for n in range(8):
        left, centre, right = n >> 2, (n >> 1) & 1, n & 1
        out |= ((rule >> n) & 1) << (4 * right + 2 * centre + left)
    return out

def complement(rule):
    """The rule for the world with 0 and 1 swapped: neighbourhood n becomes 7 - n, and the new
    state is flipped. Rule 110 becomes 137."""
    return sum((1 - ((rule >> (7 - n)) & 1)) << n for n in range(8))

def equivalent_rules(rule):
    """The rules with the same dynamics as this one, seen in a mirror, with 0 and 1 swapped, or
    both: at most four, this one included. 110 gives {110, 124, 137, 193}. The 256 rules fall into
    88 such classes."""
    return {rule, mirror(rule), complement(rule), mirror(complement(rule))}

def rollout(first, rules, n_rows, step_fn=None):
    """first (..., W) and its n_rows - 1 successors, stacked: (..., n_rows, W).
    step_fn is step by default. Pass reference_step to use the slow version."""
    step_fn = step_fn or step
    rows = [np.asarray(first)]
    for _ in range(n_rows - 1):
        rows.append(step_fn(rows[-1], rules))
    return np.stack(rows, axis=-2)
