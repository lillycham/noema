"""Two-dimensional automata: a grid of cells, a 32-bit rule, and edges that wrap round (a torus).

A cell looks at itself and its four neighbours, up, left, right and down. Read in the order of a
page, top to bottom and left to right, they make a 5-bit neighbourhood number:

    n = 16*up + 8*left + 4*itself + 2*right + 1*down          from 0 to 31

The rule is 32 bits, one for each n: the next state of a cell with neighbourhood n is bit n of the
rule number. So there are 2^32 rules, as in the 1D automaton of radius 2, and the ideal learner of
evaluate.py applies unchanged. Only the geometry is new.

    rule 0xFFFF0000   n >= 16 gives 1 (the cell above is 1): the picture moves down one row
    rule 0xF0F0F0F0   n with bit 2 set gives 1 (the cell itself is 1): nothing ever changes

step() is the fast version, for whole batches of grids at once. reference_step() is the slow,
obviously correct one, a cell at a time. tests/test_ca2d.py checks step() against it.

Grids are numpy arrays of 0 and 1 with the rows and columns on the last two axes, (H, W), and any
axes in front are a batch. In a batch, every grid can have its own rule.

A world can be turned and reflected, and 0 and 1 swapped, without changing what it does. The eight
turns and reflections of a square (the dihedral group) move the four neighbours among themselves,
and complement() swaps 0 and 1, so a rule comes in a class of up to 16 with the same dynamics.
The symmetries are found from reference_step() itself, so there is no formula to get wrong.
"""
import numpy as np

from noema import ca

CELLS = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))       # (row, column) offsets: up, left, itself, right, down

def step(grids, rules):
    """The next grids, for whole batches at once: grids (..., H, W), and one rule for each grid or one
    for all. The neighbourhood number is built a bit at a time, one rolled copy of the grid for each
    cell in CELLS, and the new cell is bit n of the rule."""
    grids = np.asarray(grids)
    rules = np.broadcast_to(
        np.asarray(rules, dtype=np.int64),
        grids.shape[:-2],
    )

    n = np.zeros(grids.shape, dtype=np.int64)

    for di, dj in CELLS:
        n = (n << 1) | np.roll(
            grids, shift=(-di, -dj), axis=(-2, -1)
        ).astype(np.int64)

    return (
        np.right_shift(rules[..., None, None], n) & 1
    ).astype(grids.dtype)

def reference_step(grids, rules):
    """The slow version of step(): one cell at a time, with the definition written out."""
    grids = np.asarray(grids)
    rules = np.broadcast_to(np.asarray(rules), grids.shape[:-2])
    out = np.empty_like(grids)
    for at in np.ndindex(grids.shape[:-2]):
        grid, rule = grids[at], int(rules[at])
        H, W = grid.shape
        for i in range(H):
            for j in range(W):
                n = 0
                for di, dj in CELLS:
                    n = (n << 1) | int(grid[(i + di) % H, (j + dj) % W])
                out[at][i, j] = (rule >> n) & 1
    return out

def rollout(first, rules, n_frames, step_fn=None):
    """first (..., H, W) and its n_frames - 1 successors, stacked: (..., n_frames, H, W).
    step_fn is step by default. Pass reference_step to use the slow version."""
    step_fn = step_fn or step
    frames = [np.asarray(first)]
    for _ in range(n_frames - 1):
        frames.append(step_fn(frames[-1], rules))
    return np.stack(frames, axis=-3)

# ---------- the symmetries ----------
def turn(grids, k, flip):
    """The k-th quarter turn of grids (..., H, W), after a left-right reflection if flip. The eight
    pairs (k from 0 to 3, flip) are the symmetries of a square."""
    grids = np.asarray(grids)
    return np.rot90(grids[..., ::-1] if flip else grids, k, axes=(-2, -1))

def turn_back(grids, k, flip):
    """The inverse of turn."""
    back = np.rot90(np.asarray(grids), -k, axes=(-2, -1))
    return back[..., ::-1] if flip else back

SYMMETRIES = tuple((k, flip) for flip in (False, True) for k in range(4))

def _neighbourhood_maps():
    """For each symmetry, a table old[n]: the rule seen through the symmetry is the rule whose bit n
    is bit old[n] of the original. Found by turning a 5x5 grid with a cross of cells set to n round
    its centre, taking one reference step with the plain rule, and turning back."""
    maps = []
    for k, flip in SYMMETRIES:
        old = np.empty(32, dtype=np.int64)
        for n in range(32):
            grid = np.zeros((5, 5), dtype=np.uint8)
            for bit, (di, dj) in enumerate(reversed(CELLS)):
                grid[2 + di, 2 + dj] = (n >> bit) & 1
            seen = turn_back(grid, k, flip)
            old[n] = sum(int(seen[2 + di, 2 + dj]) << bit for bit, (di, dj) in enumerate(reversed(CELLS)))
        maps.append(old)
    return maps

NEIGHBOURHOOD_MAPS = _neighbourhood_maps()

def transform(rule, symmetry):
    """The rule for the world turned or reflected by SYMMETRIES[symmetry]: if grids follow rule,
    then turn(grids, *SYMMETRIES[symmetry]) follows transform(rule, symmetry)."""
    old = NEIGHBOURHOOD_MAPS[symmetry]
    return sum(((rule >> int(old[n])) & 1) << n for n in range(32))

def equivalent_rules(rule):
    """The rules with the same dynamics as this one, turned, reflected, with 0 and 1 swapped, or
    any mix: at most 16, this one included."""
    others = {transform(rule, s) for s in range(len(SYMMETRIES))}
    return others | {ca.complement(r, 2) for r in others}

def neighbourhoods(grids):
    """The neighbourhood number of every cell of grids (..., H, W), as reference_step counts it.
    The new grid is bit n of the rule at each of these, so this is step() before the rule is read."""
    grids = np.asarray(grids).astype(np.int64)
    n = np.zeros_like(grids)
    for di, dj in CELLS:
        n = (n << 1) | np.roll(grids, shift=(-di, -dj), axis=(-2, -1))
    return n
