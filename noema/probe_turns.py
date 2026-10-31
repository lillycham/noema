"""Two questions about the 3D mixing models.

    python -m noema.probe_turns checkpoints/mixing_d3Lrope0.pt checkpoints/mixing_d3Lpos1.pt ...

Does a model use the coordinates of the step it is writing? The ideal learner does (the coordinates
written so far rule turns out). So there are two ideal curves, with and without that, and a model that
does not look can reach only the second. The attention of the last blocks is a direct check: the share
that falls on the coordinates written so far in this step, from queries that have some.

Where does accuracy on held-out turns dip? Accuracy by step for each held-out turn, with its order (the
smallest n with R^n = I). A turn of order 6 shows a new velocity for longer than a turn of order 2.
"""
import argparse

import numpy as np
import torch

from noema import circuit, data_mixing as dm, data_torus as dt
from noema.probe import load
from noema.train_eca import row_curve
from noema.train_torus import score

def order(R):
    """The smallest n >= 1 with R^n = I, for a square integer matrix of finite order."""
    M, n = R.copy(), 1
    while not (M == np.eye(len(R), dtype=R.dtype)).all():
        M, n = M @ R, n + 1
    return n

def current_record_share(weights, dim, first_step=3):
    """Per head, the mean share of attention on the coordinates of the step being written, over queries that
    predict a coordinate after the first of a step, from first_step on. weights (B, heads, T, T). The query
    for axis a of step t sits on coordinate a - 1, so the written coordinates are the query and a - 1 before it."""
    T = weights.shape[-1]
    stride = dim + 1
    total, count = 0, 0
    for q in range(first_step * stride, T):
        a = (q + 1) % stride                                       # the axis this query predicts
        if a in range(1, dim):
            total, count = total + weights[:, :, q, q - a + 1:q + 1].sum(-1).mean(0), count + 1
    return total / count

def shared_rows(R, turns):
    """The most rows that R has in common with any one of turns (n, d, d), not counting R itself."""
    same = (turns == R).all(-1).sum(-1)
    return int(same[same < len(R)].max())

def by_turn(model, pieces, rng, n):
    """For each turn of order 4, seen and held out: accuracy at steps 4, 5, 6, and how it relates to the training turns."""
    train = pieces.train_turns
    for label, turns in (("held", pieces.held_turns), ("seen", train)):
        for R in turns:
            if order(R) == 4:
                X = torch.from_numpy(dm.trajectory_batch(rng, n, R[None], dt.SIZE, len(R), dt.STEPS)[0])
                mask = torch.from_numpy(dt.target_mask(X.shape[1], len(R)))
                acc = score(model, X, mask, len(R), dt.SIZE)[1]
                square_seen = any((R @ R == T).all() for T in train)
                inverse_seen = any((R.T == T).all() for T in train)
                print(f"    {label} {' '.join(''.join(f'{x:+d}' for x in row) for row in R)}  steps 4-6 {acc[3]:.2f} {acc[4]:.2f} {acc[5]:.2f}"
                      f"  shares {shared_rows(R, train)} rows, inverse seen {inverse_seen}, square seen {square_seen}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--xy-pair", action="store_true", help="for checkpoints trained with --xy-pair")
    args = p.parse_args()
    dim = 3
    pieces = dm.TurnSet(dim, also_train=dm.XY_PAIR if args.xy_pair else None)
    rng = np.random.default_rng(11)
    X, R = dm.trajectory_batch(rng, 4 * args.n, pieces.group, dt.SIZE, dim, dt.STEPS)
    P = dt.decode(X, dim)
    print("ideal, using the written coordinates:     ", row_curve(dm.ideal_accuracy_by_step(P, pieces.group, dt.SIZE)))
    print("ideal, not using the written coordinates: ", row_curve(dm.ideal_accuracy_by_step(P, pieces.group, dt.SIZE, condition=False)))
    orders = [order(H) for H in pieces.held_turns]
    print(f"orders of the held-out turns: {orders}; of all turns: {sorted(order(H) for H in pieces.group)}")
    mask = torch.from_numpy(dt.target_mask(X.shape[1], dim))
    for path in args.checkpoints:
        model, ck = load(path, "cpu")
        model.eval()
        print(f"\n{path}")
        seen = torch.from_numpy(dm.trajectory_batch(rng, args.n, pieces.train, dt.SIZE, dim, dt.STEPS)[0])
        print("  seen turns      ", row_curve(score(model, seen, mask, dim, dt.SIZE)[1]))
        weights = circuit.all_patterns(model, seen[:64])
        print("  attention on the coordinates written, by layer (mean over heads):",
              " ".join(f"{current_record_share(w, dim).mean():.2f}" for w in weights))
        for n in sorted(set(orders)):
            turns = pieces.held_turns[np.array(orders) == n]
            held = torch.from_numpy(dm.trajectory_batch(rng, args.n, turns, dt.SIZE, dim, dt.STEPS)[0])
            print(f"  held-out order {n} ({len(turns):2d}) ", row_curve(score(model, held, mask, dim, dt.SIZE)[1]))
        by_turn(model, pieces, rng, 256)

if __name__ == "__main__":
    with torch.no_grad():
        main()
