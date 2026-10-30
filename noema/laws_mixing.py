"""Law checks on the mixing models: the laws of laws.py that survive a world where the velocity turns.

    python -m noema.laws_mixing checkpoints/mixing_d3Lrope0.pt
    python -m noema.laws_mixing checkpoints/mixed_rope0.pt

Independence fails in this world by design, so it is left out. The others take a new form.

    truth          the next position is the last one plus the turned velocity
    translation    shift the whole prompt by c, and the answer shifts by c
    reversal       run the prompt backwards: it is a trajectory of the turn R^-1, velocity -v, so the
                   answer is the first position minus R^-1 v_0
    relabel        apply a signed permutation S of the axes to the whole prompt: it is a trajectory of
                   the turn S R S^-1, so the answer is S of the old answer. This is the law that is
                   about the algebra: it says the axes are interchangeable.

relabel is held against the model's own answer, as translation is, so it needs no knowledge of the
world, and a model that is wrong in a consistent way passes it. Each law is scored only on prompts
that fix the answer: a prompt that leaves two next velocities open has no one right answer.
"""
import argparse

import numpy as np

from noema import mixing
from noema.laws import model_predictor
from noema.probe import load
from noema.train_eca import default_device

def turns_that_fit(P, group, size):
    """(B, |G|) booleans: which turns fit the velocities of the prompts P (B, m, d)."""
    V = (P[:, 1:] - P[:, :-1]) % size
    ok = np.ones((len(P), len(group)), dtype=bool)
    for i in range(V.shape[1] - 1):
        ok &= ((V[:, i + 1][:, None] - np.einsum("rjk,bk->brj", group, V[:, i])) % size == 0).all(-1)
    return ok

def answers(P, group, size):
    """(the next position under each turn (B, |G|, d), which turns fit (B, |G|))."""
    v = np.einsum("rjk,bk->brj", group, (P[:, -1] - P[:, -2]) % size)
    return (P[:, -1, None] + v) % size, turns_that_fit(P, group, size)

def is_fixed(P, group, size):
    """(B,) booleans: every turn that fits the prompt gives the same next position."""
    next_p, ok = answers(P, group, size)
    first = next_p[np.arange(len(P)), ok.argmax(-1)]
    return ((next_p == first[:, None]).all(-1) | ~ok).all(-1)

def oracle(group, size):
    """A predict function that keeps every law on prompts that fix the answer."""
    def predict(P):
        next_p, ok = answers(P, group, size)
        return next_p[np.arange(len(P)), ok.argmax(-1)]
    return predict

def cases(rng, B, group, size, dim, m=5, turns=None):
    """(prompts (N, m, d), turns (N, d, d), first velocities (N, d)), N <= B: trajectories of m positions
    from `turns` (default the whole group), kept if the prompt, run forwards and backwards, fixes the answer."""
    turns = group if turns is None else turns
    R = turns[rng.integers(0, len(turns), size=B)]
    v0 = rng.integers(0, size, size=(B, dim))
    P = mixing.trajectory(rng.integers(0, size, size=(B, dim)), v0, R, m, size)
    keep = is_fixed(P, group, size) & is_fixed(P[:, ::-1], group, size)
    return P[keep], R[keep], v0[keep]

def truth(predict, P, group, size):
    return (predict(P) == oracle(group, size)(P)).all(-1)

def translation(predict, P, rng, size):
    c = rng.integers(0, size, size=(len(P), P.shape[-1]))
    return (predict((P + c[:, None]) % size) == (predict(P) + c) % size).all(-1)

def reversal(predict, P, R, v0, size):
    """R is orthogonal, so R^-1 is its transpose."""
    back = np.einsum("bkj,bk->bj", R, v0)
    return (predict(P[:, ::-1]) == (P[:, 0] - back) % size).all(-1)

def relabel(predict, P, group, rng, size):
    S = group[rng.integers(0, len(group), size=len(P))]
    moved = np.einsum("bjk,bmk->bmj", S, P) % size
    return (predict(moved) == np.einsum("bjk,bk->bj", S, predict(P)) % size).all(-1)

def run(predict, rng, group, size, dim, B=1024, turns=None):
    """All four laws on the cases that fix their answer: a dict of (N,) boolean arrays."""
    P, R, v0 = cases(rng, B, group, size, dim, turns=turns)
    return {"truth": truth(predict, P, group, size), "translation": translation(predict, P, rng, size),
            "reversal": reversal(predict, P, R, v0, size), "relabel": relabel(predict, P, group, rng, size)}

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=1024, help="cases to draw for each dimension")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    model, ck = load(args.checkpoint, args.device)
    size = ck["size"]
    dims = [*ck["dims"], ck["held_dim"]] if "dims" in ck else [ck["dim"]]
    for dim in sorted(dims):
        group = mixing.signed_permutations(dim)
        results = run(model_predictor(model, size, dim, args.device), np.random.default_rng(7), group, size, dim, args.n)
        mark = "*" if dim == ck.get("held_dim") else ""
        print(f"d={dim}{mark}  " + "  ".join(f"{k} {r.mean():.3f}" for k, r in results.items()) + f"   ({len(results['truth'])} cases)")

if __name__ == "__main__":
    main()
