"""Law checks on the torus models: does the model behave like the world it was trained on?

    python -m noema.laws checkpoints/torus_s0.pt

The model is treated as a function from a prompt of m positions to the next position, and checked
like a QuickCheck property. The world has laws, and a model that has learned addition mod SIZE per
axis must keep them on prompts it has not seen. Each law returns one True or False for each case,
so a result can be split any way (seen against held-out velocities, one axis against the other).

    truth          the next position is the last one plus the velocity
    translation    shift the whole prompt by c, and the answer shifts by c
    independence   change the history of the other axes, and this axis's answer does not change
    reversal       run the prompt backwards (velocity -v), and the answer is the first position minus v
    rollout        feed the model its own answers: it should keep to the line for many steps

Every check takes a predict function, a map from prompts (B, m, d) to the next position (B, d), so
the laws are tested with an oracle that keeps them and with predictors that break them (tests/test_laws.py).
Two of the laws (translation, independence) do not need to know the true answer: they ask whether the
model agrees with itself.
"""
import argparse

import numpy as np
import torch

from noema import data_torus as dt, torus
from noema.probe import load
from noema.train_eca import default_device

def model_predictor(model, size, dim, device):
    """A predict function from a model: writes the d coordinates of the next step greedily, each
    one fed back in before the next, as the model is trained."""
    @torch.no_grad()
    def predict(P):
        model.eval()
        X = torch.from_numpy(dt.encode(P)).to(device)
        for _ in range(dim):
            X = torch.cat([X, model(X)[:, -1, :size].argmax(-1, keepdim=True)], dim=1)
        return X[:, -dim:].cpu().numpy()
    return predict

def cases(rng, B, velocities, size=dt.SIZE, dim=dt.DIM, m=3):
    """(prompts (B, m, d), velocities (B, d)): trajectories of m positions with a velocity from `velocities`
    (a function (rng, B) -> (B, d), such as VelocitySet().train) and a random start."""
    v = velocities(rng, B)
    return torus.trajectory(rng.integers(0, size, size=(B, dim)), v, m, size), v

def truth(predict, P, v, size):
    """The model's answer is the last position plus the velocity."""
    return (predict(P) == (P[:, -1] + v) % size).all(-1)

def translation(predict, P, rng, size):
    """Shift the prompt by a random c on each axis: the answer shifts by c, whatever the answer was."""
    c = rng.integers(0, size, size=(len(P), P.shape[-1]))
    return (predict((P + c[:, None]) % size) == (predict(P) + c) % size).all(-1)

def independence(predict, P, rng, size, axis):
    """Give the other axes the history of a different trajectory (the same velocity is not kept):
    the answer on this axis does not change. (B,) booleans."""
    other = P[rng.permutation(len(P))]
    mixed = other.copy()
    mixed[..., axis] = P[..., axis]
    return predict(mixed)[:, axis] == predict(P)[:, axis]

def reversal(predict, P, v, size):
    """Run the prompt backwards. It is a trajectory with velocity -v, so the next position is the
    first one minus v."""
    return (predict(P[:, ::-1]) == (P[:, 0] - v) % size).all(-1)

def rollout(predict, P, v, size, n):
    """Feed the model its own answers for n steps. Returns (B, n) booleans: is the step right, given
    that all the steps before it were the model's own?"""
    here, out = P.copy(), []
    truth_now = P[:, -1]
    for _ in range(n):
        here = np.concatenate([here, predict(here)[:, None]], axis=1)
        truth_now = (truth_now + v) % size
        out.append((here[:, -1] == truth_now).all(-1))
    return np.stack(out, axis=-1)

def run(predict, rng, vels, size, dim, B=1024, m=3, steps=12):
    """All the laws on B cases each, for seen and for held-out velocities. A dict of arrays."""
    out = {}
    for name, draw in (("seen", vels.train), ("held-out", vels.held)):
        P, v = cases(rng, B, draw, size, dim, m)
        out[f"truth/{name}"] = truth(predict, P, v, size)
        out[f"translation/{name}"] = translation(predict, P, rng, size)
        for a in range(dim):
            out[f"independence axis {a}/{name}"] = independence(predict, P, rng, size, a)
        out[f"reversal/{name}"] = reversal(predict, P, v, size)
        out[f"rollout/{name}"] = rollout(predict, P, v, size, steps - m)
    # Reversal where the reversed velocity is the held-out one (and the forward one is not).
    P, v = cases(rng, 8 * B, vels.train, size, dim, m)
    back = np.array([vels.is_held_out((-x) % size) for x in v])
    out["reversal/-v held-out"] = reversal(predict, P[back], v[back], size)
    out["reversal/-v seen"] = reversal(predict, P[~back], v[~back], size)
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=1024, help="cases for each law")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    model, ck = load(args.checkpoint, args.device)
    size, dim, steps = ck["size"], ck["dim"], ck["steps"]
    results = run(model_predictor(model, size, dim, args.device), np.random.default_rng(7),
                  dt.VelocitySet(size, dim), size, dim, args.n, steps=steps)
    for name, r in results.items():
        if name.startswith("rollout"):
            print(f"{name:34s}" + " ".join(f"{x:.2f}" for x in r.mean(0)) + "   (by step written)")
        else:
            print(f"{name:34s}{r.mean():.3f}   ({len(r)} cases)")

if __name__ == "__main__":
    main()
