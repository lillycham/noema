"""Training data for the 2D world: a random rule and a random first grid, rolled out, as tokens.

A frame of H x W cells is laid out as one long row, line by line, with a single separator at its end.
So the tokens are the same as in 1D, with a row of H * W cells, and the model, the loss and the
scoring of evaluate.py all work as they are. The only sign of the grid is position: the cell above
a cell is W tokens back, the cell to its left is 1 back (or W - 1 forward at the edge, where the
grid wraps), and the same cell in the frame before is H * W + 1 back.

Held-out rules are whole classes (ca2d.equivalent_rules), about 1 in 16, chosen by a hash as in data.RuleSet.
"""
import numpy as np

from noema import ca2d, data, evaluate, tokens

HEIGHT, WIDTH, FRAMES = 5, 5, 8       # 26 tokens a frame with its separator, so 208 tokens in all
# 5 x 5, because a bigger grid gives the rule away at once: the ideal learner is at 0.78 after one
# frame at 8 x 8, but at 0.64 at 5 x 5, where it climbs to 1.00 over the frames, as it does in 1D at radius 2.

class RuleSet:
    """The 2D rules a run trains on and the ones it holds out. There are 2^32 rules, too many to list,
    so a rule is held out when a hash of its class says so. train() never returns a held-out rule,
    and held() returns only held-out ones."""

    def __init__(self, seed=0):
        self.seed = seed

    def is_held_out(self, rule):
        return data.held_out_by_hash(min(ca2d.equivalent_rules(int(rule))), self.seed)

    def _draw(self, rng, B, held):
        out = []
        while len(out) < B:
            rule = int(rng.integers(0, 2 ** 32))
            if self.is_held_out(rule) == held:
                out.append(rule)
        return np.array(out, dtype=np.int64)

    def train(self, rng, B):
        """B rule numbers to train on."""
        return self._draw(rng, B, held=False)

    def held(self, rng, B):
        """B held-out rule numbers."""
        return self._draw(rng, B, held=True)

def rollout_batch(rng, B, rules, height=HEIGHT, width=WIDTH, frames=FRAMES, step_fn=None):
    """(tokens (B, frames * (height * width + 1)) int64, rules (B,)): B sequences, each with its own
    rule and its own random first grid. rules is an array to choose from, or a function (rng, B) ->
    B rule numbers, such as RuleSet().train."""
    chosen = rules(rng, B) if callable(rules) else rng.choice(np.asarray(rules), size=B)
    first = rng.integers(0, 2, size=(B, height, width), dtype=np.uint8)
    grids = ca2d.rollout(first, chosen, frames, step_fn)                         # (B, frames, height, width)
    return tokens.encode(grids.reshape(B, frames, height * width)), chosen

def to_frames(X, height=HEIGHT, width=WIDTH):
    """Tokens (..., frames * (height * width + 1)) -> grids (..., frames, height, width). The inverse of rollout_batch."""
    flat = tokens.decode(X, height * width)
    return flat.reshape(*flat.shape[:-1], height, width)

def ideal_accuracy_by_frame(grids):
    """The curve of the ideal table-lookup learner (evaluate.ideal_accuracy_by_row) for grids
    (B, frames, H, W): cells are written line by line, as the model writes them."""
    nb = ca2d.neighbourhoods(np.asarray(grids)[:, :-1])
    return evaluate.ideal_accuracy_from(nb.reshape(*nb.shape[:2], -1))
