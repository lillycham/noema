"""Linear probes on the torus models: what does the residual stream hold, layer by layer?

    python -m noema.probe checkpoints/torus_s0.pt

Each axis is probed where the model writes it. Coordinate a of step t + 1 is predicted from the
token before it: for axis 0 the separator that ends step t, for axis 1 the token for coordinate 0 of
step t + 1. There the model needs coordinate a of p_t and of the velocity, and it must hold the
position it is about to write, p_t + v. A probe is a linear map from the stream to a target, fitted
by ridge regression to one-hot labels. Accuracy is argmax.

The probe is fitted on trajectories with seen velocities and scored twice: on new trajectories
with seen velocities, and on ones with held-out velocities. Only about 8 velocity vectors are held
out, so few values of each coordinate occur there, and a constant guess can beat or lose to 1 / SIZE
by luck. The held-out scores are read next to the number of distinct values (printed).

The Fourier scores ask a sharper question. Regress cos and sin of 2 pi k n / SIZE on the stream,
for each frequency k, and report R^2. A model that adds mod SIZE with a circular code (as the
modular addition models of Nanda et al. do) should show a few frequencies much higher than the rest.

A probe says the information is there and linearly readable. It does not say the model uses it.
"""
import argparse

import numpy as np
import torch

from noema import data_torus as dt, gpt
from noema.train_eca import default_device

@torch.no_grad()
def residuals(model, X):
    """The residual stream before the first block and after each block: (layers + 1, B, T, emb)."""
    model.eval()
    got = []
    block = model.blocks[0]
    hooks = [block.register_forward_pre_hook(lambda m, args: got.append(args[0].detach().cpu()))]
    hooks += [b.register_forward_hook(lambda m, args, out: got.append(out.detach().cpu())) for b in model.blocks]
    try:
        model(X)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(got).numpy()

def at_queries(stream, dim, ts, axis):
    """Stream (L, B, T, emb) -> (L, B, len(ts), emb): the vector at the token that predicts coordinate
    `axis` of the step after t, for each t in ts: position t * (dim + 1) + dim + axis. (Axis 0: the separator.)"""
    return stream[:, :, np.asarray(ts) * (dim + 1) + dim + axis]

def ridge_fit(F, Y, lam=1e-2):
    """Weights W with F1 @ W close to Y, where F1 is F centred, scaled and with a constant column.
    Returns a function that applies the fit to new features."""
    mean, std = F.mean(0), F.std(0) + 1e-6
    A = np.hstack([(F - mean) / std, np.ones((len(F), 1))])
    W = np.linalg.solve(A.T @ A + lam * len(F) * np.eye(A.shape[1]), A.T @ Y)
    return lambda G: np.hstack([(G - mean) / std, np.ones((len(G), 1))]) @ W

def probe_accuracy(train_F, train_y, test_F, test_y, classes):
    """Accuracy of a ridge probe to one-hot labels in 0 .. classes - 1, fitted on train and scored on test."""
    predict = ridge_fit(train_F, np.eye(classes)[train_y])
    return float((predict(test_F).argmax(-1) == test_y).mean())

def fourier_r2(train_F, train_y, test_F, test_y, size):
    """R^2 on test of a ridge probe to (cos, sin) of 2 pi k y / size, for k = 1 .. size // 2: an array of them."""
    out = []
    for k in range(1, size // 2 + 1):
        angle = lambda y: 2 * np.pi * k * y / size
        predict = ridge_fit(train_F, np.stack([np.cos(angle(train_y)), np.sin(angle(train_y))], -1))
        target = np.stack([np.cos(angle(test_y)), np.sin(angle(test_y))], -1)
        err = ((predict(test_F) - target) ** 2).sum()
        out.append(1 - err / ((target - target.mean(0)) ** 2).sum())
    return np.array(out)

def collect(model, X, dim, size, steps, axis, first=2):
    """Features (L, N, emb) at the query for coordinate `axis` of steps first + 1 .. steps - 1, and the
    targets for each: a dict of int arrays (N,): this coordinate of the position, of the velocity, and
    of the next position."""
    p = dt.decode(X, dim)
    ts = np.arange(first, steps - 1)
    stream = residuals(model, torch.from_numpy(X).to(next(model.parameters()).device))
    feats = at_queries(stream, dim, ts, axis)
    L, B, n, E = feats.shape
    v = (p[:, 1, axis] - p[:, 0, axis]) % size
    tg = {"position": p[:, ts, axis], "velocity": np.repeat(v[:, None], n, 1), "next": p[:, ts + 1, axis]}
    return feats.reshape(L, B * n, E), {k: x.reshape(B * n) for k, x in tg.items()}

def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = gpt.GPT(gpt.Config(**ck["config"])).to(device)
    model.load_state_dict(ck["weights"])
    return model, ck

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=2048, help="sequences to fit the probes on")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    model, ck = load(args.checkpoint, args.device)
    size, dim, steps = ck["size"], ck["dim"], ck["steps"]
    vels = dt.VelocitySet(size, dim)
    rng = np.random.default_rng(100)
    Xtr, _ = dt.trajectory_batch(rng, args.n, vels.train, size, dim, steps)
    Xseen, _ = dt.trajectory_batch(rng, args.n, vels.train, size, dim, steps)
    Xheld, _ = dt.trajectory_batch(rng, args.n, vels.held, size, dim, steps)
    quantities = ("position", "velocity", "next")
    for axis in range(dim):
        Ftr, Ttr = collect(model, Xtr, dim, size, steps, axis)
        sets = {"seen": collect(model, Xseen, dim, size, steps, axis), "held-out": collect(model, Xheld, dim, size, steps, axis)}
        for name, (F, T) in sets.items():
            distinct = {q: len(np.unique(T[q])) for q in quantities}
            print(f"\naxis {axis}, probe accuracy on {name} velocities (chance {1 / size:.2f}; distinct values {distinct})")
            print("layer  " + "  ".join(f"{q:>9}" for q in quantities) + "   | Fourier R^2 of next, k = 1 .. " + str(size // 2))
            for layer in range(len(Ftr)):
                acc = [probe_accuracy(Ftr[layer], Ttr[q], F[layer], T[q], size) for q in quantities]
                r = fourier_r2(Ftr[layer], Ttr["next"], F[layer], T["next"], size)
                print(f"{layer:5d}  " + "  ".join(f"{x:9.2f}" for x in acc) + "   | " + " ".join(f"{x:5.2f}" for x in r))

if __name__ == "__main__":
    main()
