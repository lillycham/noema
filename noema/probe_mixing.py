"""Linear probes on the mixing models: where is the hidden turn read, and does a new turn get read?

    python -m noema.probe_mixing checkpoints/mixing_rope0.pt

The same probes as probe.py (ridge regression to one-hot labels, argmax), on the world where the
velocity turns every step. At the token that predicts coordinate a of position p_(t+1), the model
has read p_0 .. p_t. The targets are

    last velocity   v_(t-1)[a], the last move it has seen (p_t - p_(t-1))
    next velocity   v_t[a], the move it is about to make: R v_(t-1)
    next            p_(t+1)[a]
    turn            which of the signed permutations R is: 8 classes in 2D, the same for both axes
    row             the row of R this query needs: which axis feeds this one, and with which sign
                    (4 classes in 2D). The query for x needs only row 0, so two turns with the same
                    row 0 look alike to it: this is the target that the stream can hold.

The turn can first be read at t = 2 (two velocities, v_1 = R v_0), so that is the step reported
first, and it is where a new turn is a real test. After it the turn is also readable from v_(t-1) and
v_(t-2), and, in 2D, v_t = +-v_(t-2) always, because every turn squares to +I or -I: later steps can be
answered without ever naming R.

Velocity and next probes are fitted on trajectories with seen turns and scored on seen and on
held-out turns. The turn probe has to name a turn the other probes never saw, so it is fitted on
new trajectories with every turn equally often, and scored on a second such set: it asks whether the
stream holds a turn code even for a turn that was never trained, and not whether the model uses it.
It is reported as the recall of each turn (the share of its cases named right), seen turns first
and the held-out ones marked H. A probe that names one turn every time scores 1 on it and 0 on the
rest, and the chance level of each is 1/8.

The row probe is fitted on seen turns only and scored on a set with every turn, as recall of each
row class. Both held-out turns are made of rows that seen turns have, so a row code carries over to
them: that is a way for a new turn to be read without being new.
"""
import argparse

import numpy as np
import torch

from noema import data_mixing as dm, data_torus as dt
from noema.probe import at_queries, load, probe_accuracy, residuals, ridge_fit
from noema.train_eca import default_device

def turn_index(turns, group):
    """The index in group of each turn in turns (B, d, d)."""
    return (np.asarray(turns)[:, None] == group[None]).all((-1, -2)).argmax(-1)

def row_class(turns, axis):
    """Which axis feeds `axis` under each turn in turns (B, d, d), and with which sign: the class
    2 * source + (1 if the sign is minus else 0), one of 2 * d. This is the row of the turn that the
    query for coordinate `axis` needs: coordinate `axis` of v_t is the source coordinate of v_(t-1), signed."""
    row = np.asarray(turns)[:, axis]
    source = np.abs(row).argmax(-1)
    return 2 * source + (row[np.arange(len(row)), source] < 0)

def turn_recall(train_F, train_y, test_F, test_y, classes):
    """The share of the test cases of each turn that a ridge probe fitted on train names correctly: an
    array of `classes`. A probe that names one turn every time scores 1 on that turn and 0 on the rest,
    which a single accuracy over a lopsided set would hide."""
    predict = ridge_fit(train_F, np.eye(classes)[train_y])
    guess = predict(test_F).argmax(-1)
    return np.array([(guess[test_y == c] == c).mean() for c in range(classes)])

def collect(model, X, turns, group, dim, size, steps, ts, axis):
    """Features (L, N, emb) at the queries for coordinate `axis` of the position after step t, for t in ts,
    and a dict of int targets (N,): position, last velocity, next velocity, next, turn, row."""
    P = dt.decode(X, dim)
    V = (P[:, 1:] - P[:, :-1]) % size                                     # V[:, i] is v_i
    ts = np.asarray(ts)
    stream = residuals(model, torch.from_numpy(X).to(next(model.parameters()).device))
    feats = at_queries(stream, dim, ts, axis)
    L, B, n, E = feats.shape
    index = turn_index(turns, group)
    tg = {"position": P[:, ts, axis], "last velocity": V[:, ts - 1, axis], "next velocity": V[:, ts, axis],
          "next": P[:, ts + 1, axis], "turn": np.repeat(index[:, None], n, 1),
          "row": np.repeat(row_class(turns, axis)[:, None], n, 1)}
    return feats.reshape(L, B * n, E), {k: x.reshape(B * n) for k, x in tg.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=2048, help="sequences in each set")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    model, ck = load(args.checkpoint, args.device)
    size, dim, steps = ck["size"], ck["dim"], ck["steps"]
    ts_set = dm.TurnSet(dim)
    G = ts_set.group
    rng = np.random.default_rng(100)

    def make(turns):
        X, R = dm.trajectory_batch(rng, args.n, turns, size, dim, steps)
        return X, R

    sets = {"fit": make(ts_set.train), "seen": make(ts_set.train), "held-out": make(ts_set.held),
            "fit turns": make(G), "test turns": make(G)}
    held = [i for i, R in enumerate(G) if ts_set.is_held_out(R)]
    order = [i for i in range(len(G)) if i not in held] + held
    row_names = [f"{'+-'[k % 2]}{'xyz'[k // 2]}" for k in range(2 * dim)]
    mark = " ".join(f"{'H' if i in held else 's':>4}" for i in order)
    plans = {"t = 2 (first step the turn can be read)": [2], f"t = 3 .. {steps - 2}": list(range(3, steps - 1))}
    quantities = ("position", "last velocity", "next velocity", "next")
    for title, ts in plans.items():
        for axis in range(dim):
            C = {name: collect(model, X, R, G, dim, size, steps, ts, axis) for name, (X, R) in sets.items()}
            Ftr, Ttr = C["fit"]
            Fall, Tturn = C["fit turns"][0], C["fit turns"][1]["turn"]
            print(f"\n{title}, axis {axis}. Chance {1 / size:.2f} for a coordinate, {1 / len(G):.3f} for the turn.")
            print("layer  " + "  ".join(f"{q:>14}" for q in quantities) + "  |  row recall: " + " ".join(f"{c:>5}" for c in row_names) + "  |  turn recall by turn  " + mark)
            for layer in range(len(Ftr)):
                row = []
                for q in quantities:
                    for name in ("seen", "held-out"):
                        F, T = C[name]
                        row.append(probe_accuracy(Ftr[layer], Ttr[q], F[layer], T[q], size))
                recall = turn_recall(Fall[layer], Tturn, C["test turns"][0][layer], C["test turns"][1]["turn"], len(G))
                rows = turn_recall(Ftr[layer], Ttr["row"], C["test turns"][0][layer], C["test turns"][1]["row"], 2 * dim)
                cells = "  ".join(f"{row[2 * i]:6.2f}/{row[2 * i + 1]:<6.2f}" for i in range(len(quantities)))
                print(f"{layer:5d}  {cells}  |  {'':13s}" + " ".join(f"{r:5.2f}" for r in rows) + "  |  " + " ".join(f"{recall[i]:4.2f}" for i in order))
    print("\nEach quantity shows: accuracy on seen turns / on held-out turns.")

if __name__ == "__main__":
    main()
