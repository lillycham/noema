"""What did a rung 4 model learn across dimensions, and why does the held-out one fail or not?

    python -m noema.probe_mixed checkpoints/mixed_rope0.pt

Six pictures, cheapest first.

    graded       accuracy on the held-out dimension in easier and easier versions of it: all turns,
                 only turns that leave the last axis alone, the identity, and those with the last
                 axis frozen as well. The last is a world of one dimension less with a constant column.
    tokens       what the model writes at each place of a record: a model that still reads records of
                 the training lengths writes a separator in the wrong place.
    copies       how often its best guess is the token k places back. Reading the world in the wrong
                 period shows here as a peak at the wrong k.
    heads        for each head, the distance back it looks at most, in each dimension, as a multiple of
                 the stride d + 1. A head that follows the stride is one circuit used in every
                 dimension. A head with a fixed distance is not.
    zeroing      what zeroing one head costs in each dimension, and how those costs agree between
                 dimensions: shared heads cost in all of them.
    probes       is the last move readable on the held-out dimension at all (fitted there), and does a
                 probe fitted on one axis read another axis? (Are the axes copies of one another?)

A probe shows that information is linearly readable, not that the model uses it.
"""
import argparse

import numpy as np
import torch

from noema import circuit, data_torus as dt, mixing
from noema.probe import at_queries, load, probe_accuracy, residuals
from noema.train_eca import row_curve
from noema.train_torus import score

def fixing_last_axis(dim):
    """The signed permutations of `dim` axes that leave the last axis alone: (2^(d-1) (d-1)!, d, d)."""
    G = mixing.signed_permutations(dim)
    return G[G[:, -1, -1] == 1]

def graded_batch(rng, B, dim, kind, size=dt.SIZE, steps=dt.STEPS):
    """Tokens (B, steps * (dim + 1)) of one of the worlds of the graded test: "all", "fix last", "identity",
    or "frozen", which is "fix last" with no velocity on the last axis."""
    turns = {"all": mixing.signed_permutations(dim), "identity": np.eye(dim, dtype=np.int64)[None]}.get(kind, fixing_last_axis(dim))
    chosen = turns[rng.integers(0, len(turns), size=B)]
    v0 = rng.integers(0, size, size=(B, dim))
    if kind == "frozen":
        v0[:, -1] = 0
    return dt.encode(mixing.trajectory(rng.integers(0, size, size=(B, dim)), v0, chosen, steps, size))

def live_ids(B, L, dim, first_step=3):
    """ids (B, L) for circuit.offset_profile: 0 at the coordinates of steps from first_step on, else -1."""
    position = np.arange(L)
    return torch.from_numpy(np.where((position // (dim + 1) >= first_step) & (position % (dim + 1) < dim), 0, -1)[None].repeat(B, 0))

def copy_agreement(guess, X, distances):
    """For each k in distances, the share of the coordinate places where guess (B, L - 1), the best guess for
    X[:, 1:], is the token k places before the token it guesses. Places within k of the start don't count."""
    return np.array([(guess[:, k - 1:] == X[:, :X.shape[1] - k]).float().mean().item() for k in distances])

def top_distances(model, X, dim, max_offset):
    """(top distance (layers, heads), share of attention there (layers, heads)), averaged over X."""
    ids = live_ids(len(X), X.shape[1], dim)
    profile = torch.stack([circuit.offset_profile(w, ids, max_offset) for w in circuit.all_patterns(model, X)])
    share, distance = profile[..., 1:].max(-1)
    return (distance + 1).numpy(), share.numpy()

def stride_count(distance, share, stride, sharp=0.3):
    """How many heads with at least `sharp` of their attention at one distance are at a distance k where
    k + 1 is a multiple of stride: the key is a whole number of records back, in that stride. Strides that
    share a factor overlap (k = 23 counts for 2, 3 and 4), so read 4 against 5, which only k = 19 confuses."""
    return int((((distance + 1) % stride == 0) & (share >= sharp)).sum())

def zero_costs(model, X, mask, dim, size):
    """Accuracy lost when one head is zeroed: (layers, heads)."""
    base = score(model, X, mask, dim, size)[1].mean()
    table = np.zeros((model.cfg.layers, model.cfg.heads))
    for l in range(model.cfg.layers):
        for h in range(model.cfg.heads):
            with circuit.zeroed(model, l, h):
                table[l, h] = base - score(model, X, mask, dim, size)[1].mean()
    return table

def last_moves(model, X, dim, size, ts):
    """{axis: (features (L, N, emb), last move on that axis (N,))} at the query for each axis, steps ts."""
    P = dt.decode(X, dim)
    stream = residuals(model, torch.from_numpy(X))
    out = {}
    for a in range(dim):
        F = at_queries(stream, dim, ts, a)
        L, B, n, E = F.shape
        out[a] = F.reshape(L, B * n, E), ((P[:, ts, a] - P[:, np.asarray(ts) - 1, a]) % size).reshape(B * n)
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=256, help="sequences for each measurement")
    p.add_argument("--probe-n", type=int, default=1024, help="sequences to fit a probe on")
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    model, ck = load(args.checkpoint, "cpu")
    model.eval()
    size, steps, dims, held = ck["size"], ck["steps"], ck["dims"], ck["held_dim"]
    rng = np.random.default_rng(args.seed)
    every = sorted([*dims, held])
    make = lambda d, kind="all", B=args.n: torch.from_numpy(graded_batch(rng, B, d, kind, size, steps))
    mask = lambda d: torch.from_numpy(dt.target_mask(steps * (d + 1), d))

    print(f"trained on {dims}, held out {held}. Chance {1 / size:.2f}.\n\ngraded test, accuracy by step")
    for d in every:
        for kind in ("all", "fix last", "identity", "frozen"):
            print(f"  d={d}{'*' if d == held else ' '} {kind:9s} {row_curve(score(model, make(d, kind), mask(d), d, size)[1])}")

    print("\nshare of best guesses that are a separator, by place in the record (steps 3 on)")
    for d in every:
        X = make(d)
        guess = model(X[:, :-1]).argmax(-1)
        place = (torch.arange(1, X.shape[1]) % (d + 1))
        late = torch.arange(1, X.shape[1]) // (d + 1) >= 3
        print(f"  d={d}: " + " ".join(f"{(guess[:, (place == q) & late] == dt.SEP).float().mean():.2f}" for q in range(d + 1)))

    print("\nshare of best guesses equal to the token k back")
    ks = list(range(1, 13))
    print("  k:   " + " ".join(f"{k:4d}" for k in ks))
    for d in every:
        X = make(d)
        print(f"  d={d}: " + " ".join(f"{x:4.2f}" for x in copy_agreement(model(X[:, :-1]).argmax(-1), X, ks)))

    print("\nheads: top distance back / stride, with its share of attention (* = at least 0.3 in that dimension)")
    tops = {d: top_distances(model, make(d, B=64), d, 24) for d in every}
    strides = [d + 1 for d in every]
    tracking = 0
    for l in range(model.cfg.layers):
        for h in range(model.cfg.heads):
            cells = []
            for d in every:
                k, share = tops[d][0][l, h], tops[d][1][l, h]
                cells.append(f"d={d} {k:2d}={k / (d + 1):4.2f}s {share:.2f}{'*' if share >= 0.3 else ' '}")
            ratios = {round(tops[d][0][l, h] / (d + 1), 2) for d in dims}
            sharp = all(tops[d][1][l, h] >= 0.3 for d in dims)
            tracking += sharp and len(ratios) == 1
            print(f"  L{l}H{h}  " + "  ".join(cells) + ("  <- follows the stride" if sharp and len(ratios) == 1 else ""))
    print(f"  heads sharp in every training dimension that keep one multiple of the stride: {tracking}")
    print("  which stride the sharp heads of each dimension read: heads with distance + 1 a multiple of that stride")
    for d in every:
        print(f"    d={d} ({int((tops[d][1] >= 0.3).sum())} sharp): " + "  ".join(f"stride {s}: {stride_count(*tops[d], s)}" for s in strides))

    print("\nzeroing one head: points of accuracy lost")
    costs = {d: zero_costs(model, make(d), mask(d), d, size) for d in dims}
    for d in dims:
        circuit.print_table(f"d={d}", 100 * costs[d])
    print("correlation of the costs between dimensions")
    for i, a in enumerate(dims):
        for b in dims[i + 1:]:
            print(f"  d={a} and d={b}: {np.corrcoef(costs[a].ravel(), costs[b].ravel())[0, 1]:.2f}")

    print("\nprobes on the last move (chance about 0.09), by layer. d=N a->b: fitted on axis a, scored on axis b")
    ts = list(range(3, steps - 1))
    for d in every:
        fit, test = (last_moves(model, graded_batch(rng, args.probe_n, d, "all", size, steps), d, size, ts) for _ in range(2))
        print(f"  d={d}{'*' if d == held else ' '} same axis, " + "  ".join(
            f"axis {a}: " + " ".join(f"{probe_accuracy(fit[a][0][l], fit[a][1], test[a][0][l], test[a][1], size):.2f}" for l in (2, 3, 4, 6)) for a in range(d))
            + "   (layers 2 3 4 6)")
        for b in range(1, d):
            print(f"       0->{b}: " + " ".join(f"{probe_accuracy(fit[0][0][l], fit[0][1], test[b][0][l], test[b][1], size):.2f}" for l in (2, 3, 4, 6)))

if __name__ == "__main__":
    with torch.no_grad():
        main()
