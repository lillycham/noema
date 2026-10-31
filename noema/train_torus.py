"""Train a small transformer on a particle moving on a torus, from raw coordinates.

    python -m noema.train_torus --steps 3000 --tag first

Every sequence has its own hidden velocity and random start (see torus.py). The model sees only
coordinate tokens and a separator after each step, and has to work out the velocity from the steps
so far. Whole velocities are held out of training, about 1 in 16.

With --mixing the velocity also turns every step by a hidden turn, a signed permutation of the axes
(see mixing.py), and whole turns are held out, 1 in 4. The loop is the one in train_eca.py: only
the data and the scoring are different. Saves the weights in checkpoints/ for the probes of probe.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noema import data_mixing as dm, data_torus as dt, gpt, mixing, torus
from noema.train_eca import default_device, lr_at, row_curve, train_step

@torch.no_grad()
def score(model, X, mask, dim, size, stride=None):
    """(loss, accuracy by step) on a batch X (B, L) of token IDs: the share of coordinates in step t
    that the model's best guess gets right, for t = 1 to steps - 1. stride defaults to dim + 1."""
    model.eval()
    logits = model(X[:, :-1])
    targets = X[:, 1:]
    loss = gpt.masked_cross_entropy(logits, targets, mask).item()
    right = (logits[..., :size].argmax(-1) == targets).float()
    stride = stride or dim + 1
    position = torch.arange(1, X.shape[1], device=X.device)
    by_step = [right[:, (position // stride == t) & (position % stride < dim)].mean().item()
               for t in range(1, X.shape[1] // stride)]
    return loss, np.array(by_step)

@torch.no_grad()
def accuracy_by_axis(model, X, dim, size, steps_at=(2, 3, 4)):
    """Accuracy of the model's best guess for each axis of the positions p_t, for the steps in steps_at:
    an array (len(steps_at), dim). Teacher-forced: coordinate a of p_t is predicted with the true earlier
    coordinates of p_t in front of it. (The later axes are helped by the earlier ones: in 2D, the
    x of a new turn fixes the y.)"""
    model.eval()
    pred = model(X[:, :-1]).argmax(-1)
    stride = dim + 1
    return np.array([[(pred[:, t * stride + a - 1] == X[:, t * stride + a]).float().mean().item()
                      for a in range(dim)] for t in steps_at])

class World:
    """What a run is trained on: batch(rng, B, held) -> tokens (B, L), the held-out part if held, and
    ideal(X) -> the ideal learner's accuracy by step on a batch of tokens X."""

    def __init__(self, args):
        size, dim, steps = args.size, args.dim, args.steps_per_seq
        if args.mixing:
            self.pieces = dm.TurnSet(dim, also_train=dm.XY_PAIR if args.xy_pair else None)
            advance_fn = mixing.reference_advance if args.slow_world else None
            self.batch = lambda rng, B, held: dm.trajectory_batch(
                rng, B, self.pieces.held if held else self.pieces.train, size, dim, steps, advance_fn)[0]
            self.ideal = lambda X: dm.ideal_accuracy_by_step(dt.decode(X, dim), self.pieces.group, size)
        else:
            self.pieces = dt.VelocitySet(size, dim)
            advance_fn = torus.reference_advance if args.slow_world else None
            self.batch = lambda rng, B, held: dt.trajectory_batch(
                rng, B, self.pieces.held if held else self.pieces.train, size, dim, steps, advance_fn)[0]
            self.ideal = lambda X: dt.ideal_accuracy_by_step(steps, size)

def fixed_batch(world, seed, B, args, held):
    return torch.from_numpy(world.batch(np.random.default_rng(seed), B, held)).to(args.device)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3000, help="training steps")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    for name, default in (("emb", 192), ("heads", 6), ("layers", 6)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--size", type=int, default=dt.SIZE, help="points along each axis (default 11)")
    p.add_argument("--dim", type=int, default=dt.DIM, help="axes (default 2)")
    p.add_argument("--steps-per-seq", type=int, default=dt.STEPS, help="positions in a sequence (default 12)")
    p.add_argument("--learned-positions", action="store_true", help="a position table in place of RoPE")
    p.add_argument("--no-qk-norm", action="store_true")
    p.add_argument("--mixing", action="store_true", help="the world of mixing.py: the velocity turns every step")
    p.add_argument("--xy-pair", action="store_true", help="3D: train on dm.XY_PAIR, which the usual split holds out")
    p.add_argument("--slow-world", action="store_true", help="use the reference step of the world (torus.reference_advance, or mixing.reference_advance)")
    p.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="torus")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    world = World(args)
    L = args.steps_per_seq * (args.dim + 1)
    cfg = gpt.Config(vocab=args.size + 1, block=L, emb=args.emb, heads=args.heads, layers=args.layers,
                     rope=not args.learned_positions, qk_norm=not args.no_qk_norm)
    model = gpt.GPT(cfg).to(args.device)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(gpt.param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95))
    mask = torch.from_numpy(dt.target_mask(L, args.dim)).to(args.device)
    seen = fixed_batch(world, 1, 512, args, held=False)
    unseen = fixed_batch(world, 2, 512, args, held=True)
    ideal = world.ideal(seen.cpu().numpy())
    print(f"{gpt.n_parameters(model):,} parameters on {args.device}; {'mixing' if args.mixing else 'torus'} {args.size}^{args.dim}, "
          f"{args.steps_per_seq} steps, {L} tokens; {cfg}", flush=True)
    print(f"ideal learner, by step:  {row_curve(ideal)}  (mean {ideal.mean():.3f})", flush=True)

    t = time.time()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            loss, acc = score(net, seen, mask, args.dim, args.size)
            loss_h, acc_h = score(net, unseen, mask, args.dim, args.size)
            print(f"step {step:5d}  val loss {loss:.4f}  acc {acc.mean():.3f}  | held-out: "
                  f"loss {loss_h:.4f}  acc {acc_h.mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
        if step == args.steps:
            break
        X = world.batch(rng, args.batch, held=False)
        train_step(net, opt, torch.from_numpy(X).to(args.device), mask, lr_at(step, args.steps, args.lr, args.warmup))

    print(f"\nideal learner, by step:         {row_curve(ideal)}")
    print(f"accuracy by axis at steps 2, 3, 4 (rows), seen:     " + " | ".join(row_curve(r) for r in accuracy_by_axis(net, seen, args.dim, args.size)))
    print(f"accuracy by axis at steps 2, 3, 4 (rows), held out: " + " | ".join(row_curve(r) for r in accuracy_by_axis(net, unseen, args.dim, args.size)))
    print(f"accuracy by step, seen:         {row_curve(acc)}")
    print(f"accuracy by step, held out:     {row_curve(acc_h)}")
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"config": cfg.__dict__, "weights": model.state_dict(), "size": args.size, "dim": args.dim,
                "steps": args.steps_per_seq, "mixing": args.mixing},
               Path("checkpoints") / f"{'mixing' if args.mixing else 'torus'}_{args.tag}.pt")
    print(f"\nSaved checkpoints/{'mixing' if args.mixing else 'torus'}_{args.tag}.pt")

if __name__ == "__main__":
    main()
