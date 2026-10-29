"""Training data: a random rule and a random first row, rolled out, as tokens.

The model is never told the rule. It has to work it out from the rows it has seen so far, so
the later rows of a sequence are easier to predict than the early ones. That curve, accuracy
against row, is what the evaluation plots.

Some rules are held out of training. A model that only learned the rules it saw will do well on
those and badly on the held-out ones. A model that works out the rule in context does well on both.
"""
import numpy as np

from noema import ca, tokens

WIDTH, ROWS = 15, 16      # 16 tokens a row with its separator, so 256 tokens in all

def split_rules(n_heldout=32, seed=0):
    """(train rules, held-out rules), sorted, the same every time.

    Whole classes of equivalent rules are held out (see ca.equivalent_rules). Holding out rule 110
    while training on its mirror image 124 would not test anything: the same dynamics, in a
    disguise the model may well see through. Classes are taken in shuffled order until there are at
    least n_heldout rules, so a few more than n_heldout can come back.
    """
    classes = sorted({frozenset(ca.equivalent_rules(r)) for r in range(256)}, key=min)
    held = set()
    for i in np.random.default_rng(seed).permutation(len(classes)):
        if len(held) >= n_heldout:
            break
        held |= classes[i]
    return np.array(sorted(set(range(256)) - held)), np.array(sorted(held))

def rollout_batch(rng, B, rules, width=WIDTH, rows=ROWS, step_fn=None):
    """(tokens (B, rows * (width + 1)) int64, rules (B,)): B sequences, each with its own rule
    chosen from rules and its own random first row. rng is a numpy Generator."""
    chosen = rng.choice(np.asarray(rules), size=B)
    first = rng.integers(0, 2, size=(B, width), dtype=np.uint8)
    return tokens.encode(ca.rollout(first, chosen, rows, step_fn)), chosen
