"""Train a small transformer on a particle moving on a torus, from raw coordinates.

    python -m noema.train_torus --steps 3000 --tag first

Every sequence has its own hidden velocity and random start (see torus.py). The model sees only
coordinate tokens and a separator after each step, and has to work out the velocity from the
steps so far. Whole velocities are held out of training, about 1 in 16. The loop is the one in
train_eca.py: only the data and the scoring are different. Saves the weights in checkpoints/ for
the probes of probe.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noema import data_torus as dt, gpt, torus
from noema.train_eca import default_device, lr_at, row_curve, train_step

@torch.no_grad()
def score(model, X, mask, dim, size):
    """(loss, accuracy by step) on a batch X (B, L) of token IDs: the share of coordinates in step t
    that the model's best guess gets right, for t = 1 to steps - 1."""
    model.eval()
    logits = model(X[:, :-1])
    targets = X[:, 1:]
    loss = gpt.masked_cross_entropy(logits, targets, mask).item()
    right = (logits[..., :size].argmax(-1) == targets).float()
    stride = dim + 1
    position = torch.arange(1, X.shape[1], device=X.device)
    by_step = [right[:, (position // stride == t) & (position % stride < dim)].mean().item()
               for t in range(1, X.shape[1] // stride)]
    return loss, np.array(by_step)

def fixed_batch(velocities, seed, B, args, advance_fn=None):
    X, _ = dt.trajectory_batch(np.random.default_rng(seed), B, velocities, args.size, args.dim, args.steps_per_seq, advance_fn)
    return torch.from_numpy(X).to(args.device)

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
    p.add_argument("--slow-world", action="store_true", help="use torus.reference_advance")
    p.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="torus")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    advance_fn = torus.reference_advance if args.slow_world else None

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vels = dt.VelocitySet(args.size, args.dim)
    L = args.steps_per_seq * (args.dim + 1)
    cfg = gpt.Config(vocab=args.size + 1, block=L, emb=args.emb, heads=args.heads, layers=args.layers,
                     rope=not args.learned_positions, qk_norm=not args.no_qk_norm)
    model = gpt.GPT(cfg).to(args.device)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(gpt.param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95))
    mask = torch.from_numpy(dt.target_mask(L, args.dim)).to(args.device)
    seen = fixed_batch(vels.train, 1, 512, args, advance_fn)
    unseen = fixed_batch(vels.held, 2, 512, args, advance_fn)
    ideal = dt.ideal_accuracy_by_step(args.steps_per_seq, args.size)
    print(f"{gpt.n_parameters(model):,} parameters on {args.device}; torus {args.size}^{args.dim}, "
          f"{args.steps_per_seq} steps, {L} tokens; {cfg}", flush=True)
    print(f"ideal learner, by step:  {row_curve(ideal)}  (mean {ideal.mean():.3f})", flush=True)

    t = time.time()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            loss, acc = score(net, seen, mask, args.dim, args.size)
            loss_h, acc_h = score(net, unseen, mask, args.dim, args.size)
            print(f"step {step:5d}  val loss {loss:.4f}  acc {acc.mean():.3f}  | held-out velocities: "
                  f"loss {loss_h:.4f}  acc {acc_h.mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
        if step == args.steps:
            break
        X, _ = dt.trajectory_batch(rng, args.batch, vels.train, args.size, args.dim, args.steps_per_seq, advance_fn)
        train_step(net, opt, torch.from_numpy(X).to(args.device), mask, lr_at(step, args.steps, args.lr, args.warmup))

    print(f"\nideal learner, by step:         {row_curve(ideal)}")
    print(f"accuracy by step, seen:         {row_curve(acc)}")
    print(f"accuracy by step, held out:     {row_curve(acc_h)}")
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"config": cfg.__dict__, "weights": model.state_dict(), "size": args.size, "dim": args.dim,
                "steps": args.steps_per_seq}, Path("checkpoints") / f"torus_{args.tag}.pt")
    print(f"\nSaved checkpoints/torus_{args.tag}.pt")

if __name__ == "__main__":
    main()
