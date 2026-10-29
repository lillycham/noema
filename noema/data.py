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

class RuleSet:
    """The rules a run trains on and the ones it holds out, at one radius.

    Radius 1 has 256 rules, so the split is a list (split_rules). From radius 2 on there are too
    many to list: 2^32 at radius 2. A rule is held out when a hash of its class says so, about 1
    class in 16. A class is the rule with its mirror and complement (ca.equivalent_rules), so no
    twin of a held-out rule is ever trained on. train() never returns a held-out rule, and held()
    returns only held-out ones.

    From radius 2 on, a training run sees about steps * batch rules out of billions, so the held-out
    rules are not just unseen: no model could have learned the table of any rule beforehand.
    """

    def __init__(self, radius=1, n_heldout=32, seed=0):
        self.radius, self.seed = radius, seed
        if radius == 1:
            self._train, self._held = split_rules(n_heldout, seed)
            self._held_set = set(self._held.tolist())

    def is_held_out(self, rule):
        if self.radius == 1:
            return int(rule) in self._held_set
        rep = min(ca.equivalent_rules(int(rule), self.radius))      # the same for every twin
        mixed = ((rep ^ (self.seed * 0x9E3779B1)) * 0x9E3779B97F4A7C15) % 2 ** 64
        return mixed >> 60 == 0                                      # the top 4 bits are 0: 1 in 16

    def _draw(self, rng, B, held):
        out = []
        while len(out) < B:
            rule = int(rng.integers(0, 2 ** 2 ** (2 * self.radius + 1)))      # a uniform random rule
            if self.is_held_out(rule) == held:
                out.append(rule)
        return np.array(out, dtype=np.int64)

    def train(self, rng, B):
        """B rule numbers to train on."""
        return rng.choice(self._train, size=B) if self.radius == 1 else self._draw(rng, B, held=False)

    def held(self, rng, B):
        """B held-out rule numbers."""
        return rng.choice(self._held, size=B) if self.radius == 1 else self._draw(rng, B, held=True)

def rollout_batch(rng, B, rules, width=WIDTH, rows=ROWS, step_fn=None, radius=1):
    """(tokens (B, rows * (width + 1)) int64, rules (B,)): B sequences, each with its own rule
    and its own random first row. rules is an array of rule numbers to choose from, or a function
    (rng, B) -> B rule numbers, such as RuleSet(...).train. rng is a numpy Generator."""
    chosen = rules(rng, B) if callable(rules) else rng.choice(np.asarray(rules), size=B)
    first = rng.integers(0, 2, size=(B, width), dtype=np.uint8)
    return tokens.encode(ca.rollout(first, chosen, rows, step_fn, radius)), chosen
