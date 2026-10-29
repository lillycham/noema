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

def step(rows, rules, radius=1):
    n = np.zeros_like(rows, dtype=np.int64)

    for offset in range(-radius, radius + 1):
        n = (n << 1) | np.roll(rows, -offset, axis=-1)

    return (np.asarray(rules)[..., None] >> n) & 1

def reference_step(rows, rules, radius=1):
    """The slow version of step(): one cell at a time, with the definition written out.

    The radius is how far a cell looks each way: a neighbourhood is 2*radius + 1 cells, and a rule
    has 2^(2*radius + 1) bits. Radius 1 is the elementary automaton above, with 8 bits. Radius 2 has
    32 bits, so 2^32 rules: about four billion, too many for a model to know them all.

    The cell at offset o from the one being updated (o from -radius to radius) counts for
    2^(radius - o) in the neighbourhood number n. For radius 1 that is 4*left + 2*itself + right.
    """
    rows = np.asarray(rows)
    rules = np.broadcast_to(np.asarray(rules), rows.shape[:-1])
    out = np.empty_like(rows)
    for at in np.ndindex(rows.shape[:-1]):
        row, rule = rows[at], int(rules[at])
        W = len(row)
        for i in range(W):
            n = sum(int(row[(i + o) % W]) << (radius - o) for o in range(-radius, radius + 1))
            out[at][i] = (rule >> n) & 1
    return out

def mirror(rule, radius=1):
    """The rule for the left-right reflected world: the bit for a neighbourhood moves to the one with
    its cells in the opposite order, which is the same number with its bits reversed. Rule 110
    becomes 124. Doing it twice gives the rule back."""
    k = 2 * radius + 1
    out = 0
    for n in range(2 ** k):
        out |= ((rule >> n) & 1) << int(format(n, f"0{k}b")[::-1], 2)
    return out

def complement(rule, radius=1):
    """The rule for the world with 0 and 1 swapped: neighbourhood n becomes 2^k - 1 - n, and the new
    state is flipped. Rule 110 becomes 137."""
    top = 2 ** (2 * radius + 1) - 1
    return sum((1 - ((rule >> (top - n)) & 1)) << n for n in range(top + 1))

def equivalent_rules(rule, radius=1):
    """The rules with the same dynamics as this one, seen in a mirror, with 0 and 1 swapped, or
    both: at most four, this one included. 110 gives {110, 124, 137, 193}. The 256 rules of radius 1
    fall into 88 such classes."""
    other = complement(rule, radius)
    return {rule, mirror(rule, radius), other, mirror(other, radius)}

def rollout(first, rules, n_rows, step_fn=None, radius=1):
    """first (..., W) and its n_rows - 1 successors, stacked: (..., n_rows, W).
    step_fn is step by default. Pass reference_step to use the slow version.
    From radius 2 on, step_fn gets the radius as a third argument."""
    step_fn = step_fn or step
    extra = () if radius == 1 else (radius,)
    rows = [np.asarray(first)]
    for _ in range(n_rows - 1):
        rows.append(step_fn(rows[-1], rules, *extra))
    return np.stack(rows, axis=-2)
