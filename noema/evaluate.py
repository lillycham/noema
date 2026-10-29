"""Judging a model: the loss, the accuracy row by row, and a rollout it writes itself.

Accuracy by row is the main picture. The first row is random. The second is predictable only if
the model has guessed the rule from the first pair of rows, and so on. A model that works out the
rule in context starts near chance and climbs towards 1 over the rows. A model that has only
learned the rules it saw does that only for those rules, and the held-out rules show it.
"""
import numpy as np
import torch

from noema import gpt, tokens

@torch.no_grad()
def score(model, X, mask, width):
    """(loss, accuracy by row) on a batch X (B, L) of token IDs. Accuracy is the share of cells
    in row r that the model's best guess gets right, for r = 1 to rows - 1: an array of rows - 1."""
    model.eval()
    logits = model(X[:, :-1])
    targets = X[:, 1:]
    loss = gpt.masked_cross_entropy(logits, targets, mask).item()
    right = (logits[..., :2].argmax(-1) == targets).float()                        # cells only: 0 or 1
    stride = width + 1
    position = torch.arange(1, X.shape[1], device=X.device)
    by_row = [right[:, (position // stride == r) & (position % stride < width)].mean().item()
              for r in range(1, X.shape[1] // stride)]
    return loss, np.array(by_row)

@torch.no_grad()
def complete(model, prefix, width, temperature=0.0, generator=None):
    """Carry on a batch of rollouts to the model's full block. prefix (B, t) is whole rows.
    A separator goes where a row ends, so the model only chooses cells: 0 or 1.
    temperature 0 takes the likeliest cell every time."""
    model.eval()
    X, stride = prefix.clone(), width + 1
    while X.shape[1] < model.cfg.block:
        if X.shape[1] % stride == width:
            nxt = torch.full((X.shape[0], 1), tokens.SEP, device=X.device)
        else:
            logits = model(X)[:, -1, :2]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator)
        X = torch.cat([X, nxt], dim=1)
    return X

def neighbourhoods(rows, radius=1):
    """The neighbourhood number of every cell of rows (..., W), as ca.reference_step counts it: the
    cell at offset o from the one in question counts for 2^(radius - o)."""
    rows = np.asarray(rows).astype(np.int64)
    return sum(np.roll(rows, -o, axis=-1) << (radius - o) for o in range(-radius, radius + 1))

def ideal_accuracy_by_row(grid, radius=1):
    """The accuracy of an ideal table-lookup learner on rows 1 to n - 1 of grid (B, n, W) of 0 and 1,
    as an array of n - 1: the curve for the model to be held against.

    The learner knows only that a rule is a table, from neighbourhood number to new cell, and that
    its entries are fair coins. When it has to predict a cell, it has seen every transition in the rows
    above and in the cells to the left in the row it is writing. If it has met this cell's neighbourhood
    number in those, it is certain: 1.0. If not, it can only guess: 0.5. So the curve is how fast a
    sequence reveals its own rule, and no model can beat it on random tables, which radius 2 has.
    (At radius 1 there are only 256 rules, and a model that knows them can do better, because a few
    entries of the table then give away the rest.)
    """
    return ideal_accuracy_from(neighbourhoods(np.asarray(grid)[:, :-1], radius))

def ideal_accuracy_from(nb):
    """ideal_accuracy_by_row for any world, from nb (B, T, N): the neighbourhood number of each of the
    N cells of a row (or frame) t, in the order the model writes them, as read to make row t + 1."""
    B, T, W = nb.shape
    known = np.zeros((T, B, W))
    for b in range(B):
        for t in range(T):
            met = set(nb[b, :t].ravel().tolist())                      # all the transitions of the rows above
            for i in range(W):
                known[t, b, i] = nb[b, t, i] in met
                met.add(nb[b, t, i])                                    # the cell just written
    return (0.5 + 0.5 * known).mean(axis=(1, 2))

def ideal_free_running(grid, prompt_rows, radius=1, rng=None):
    """What an ideal table-lookup learner writes when it carries on alone from the first prompt_rows
    rows of grid (B, n, W), and nothing else: an array shaped like grid, the prompt then its rows.

    It builds its table from the transitions between the prompt rows. It writes a row cell by cell
    from the row it wrote before. A cell whose neighbourhood number is in the table is certain. Any
    other is a fair coin, and the coin is kept in the table so the learner agrees with itself. It is
    never shown the truth again, so it cannot learn from its mistakes. That is the model's position
    too, so this is the curve for a free-running model to be held against, as ideal_accuracy_by_row
    is for a teacher-forced one.
    """
    rng = rng or np.random.default_rng(0)
    grid = np.asarray(grid)
    B, n, W = grid.shape
    out = grid.copy()
    nb = neighbourhoods(grid[:, :prompt_rows - 1], radius)             # the prompt's transitions, as read
    for b in range(B):
        table = {int(k): int(v) for k, v in zip(nb[b].ravel(), grid[b, 1:prompt_rows].ravel())}
        for t in range(prompt_rows, n):
            for i, k in enumerate(neighbourhoods(out[b, t - 1], radius).tolist()):
                if k not in table:
                    table[k] = int(rng.integers(2))
                out[b, t, i] = table[k]
    return out

def wrong_by_row(written, truth, first=0):
    """The share of cells that differ from the truth, row by row from row first: (B, n, W) twice."""
    return (np.asarray(written) != np.asarray(truth))[:, first:].mean(axis=(0, 2))
