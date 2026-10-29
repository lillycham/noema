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
