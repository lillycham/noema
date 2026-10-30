"""Rung 4: train one transformer on the mixing world in several dimensions, and test it on one it never saw.

    python -m noema.train_mixed --dims 1 2 3 --held-dim 4 --steps 12000 --tag first

Each training batch is in one dimension, drawn at random from --dims (see data_mixed.py). The model
has one set of weights and no token that names the dimension. Every so often it is scored on fixed
batches of each training dimension and of the held-out one, next to the ideal learner of each. The
block is long enough for the held-out dimension, so a learned position table has rows that no training
sequence reached (a sequence in d = 3 has 48 tokens, one in d = 4 has 60). RoPE has no such rows.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noema import data_mixed as dx, data_torus as dt, gpt
from noema.train_eca import default_device, lr_at, row_curve, train_step
from noema.train_torus import accuracy_by_axis, score

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3000, help="training steps")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    for name, default in (("emb", 192), ("heads", 6), ("layers", 6)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--size", type=int, default=dt.SIZE)
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3], help="dimensions to train on (default 1 2 3)")
    p.add_argument("--held-dim", type=int, default=4, help="the dimension held out (default 4)")
    p.add_argument("--steps-per-seq", type=int, default=dt.STEPS)
    p.add_argument("--learned-positions", action="store_true", help="a position table in place of RoPE")
    p.add_argument("--no-qk-norm", action="store_true")
    p.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    p.add_argument("--init", help="start from the weights of this checkpoint, to compare with a run from scratch")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="mixed")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    assert args.held_dim not in args.dims, "the held-out dimension is in the training dimensions"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    all_dims = [*args.dims, args.held_dim]
    L = args.steps_per_seq * (max(all_dims) + 1)
    cfg = gpt.Config(vocab=args.size + 1, block=L, emb=args.emb, heads=args.heads, layers=args.layers,
                     rope=not args.learned_positions, qk_norm=not args.no_qk_norm)
    model = gpt.GPT(cfg).to(args.device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=args.device)["weights"])
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(gpt.param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95))
    masks = {d: torch.from_numpy(dt.target_mask(args.steps_per_seq * (d + 1), d)).to(args.device) for d in all_dims}
    fixed = {d: torch.from_numpy(dx.batch(np.random.default_rng(100 + d), 512, d, args.size, args.steps_per_seq)).to(args.device)
             for d in all_dims}
    ideal = {d: dx.ideal(fixed[d].cpu().numpy(), d, args.size) for d in all_dims}
    print(f"{gpt.n_parameters(model):,} parameters on {args.device}; mixing {args.size}^d, d in {args.dims}, held out {args.held_dim}, "
          f"{args.steps_per_seq} steps, up to {L} tokens; {cfg}", flush=True)
    for d in all_dims:
        print(f"ideal learner d={d}, by step:  {row_curve(ideal[d])}  (mean {ideal[d].mean():.3f})", flush=True)

    t = time.time()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            scored = {d: score(net, fixed[d], masks[d], d, args.size) for d in all_dims}
            line = "  ".join(f"d{d}{'*' if d == args.held_dim else ''} {scored[d][1].mean():.3f}/{ideal[d].mean():.3f}" for d in all_dims)
            print(f"step {step:5d}  acc/ideal  {line}  ({time.time() - t:.0f}s)", flush=True)
        if step == args.steps:
            break
        X, d = dx.train_batch(rng, args.batch, args.dims, args.size, args.steps_per_seq)
        train_step(net, opt, torch.from_numpy(X).to(args.device), masks[d], lr_at(step, args.steps, args.lr, args.warmup))

    print("\n(* = held-out dimension)")
    for d in all_dims:
        print(f"d={d}{'*' if d == args.held_dim else ' '} ideal:     {row_curve(ideal[d])}")
        print(f"     accuracy:  {row_curve(scored[d][1])}")
        print(f"     by axis at steps 2, 3, 4: " + " | ".join(row_curve(r) for r in accuracy_by_axis(net, fixed[d], d, args.size)))
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"config": cfg.__dict__, "weights": model.state_dict(), "size": args.size, "dims": args.dims,
                "held_dim": args.held_dim, "steps": args.steps_per_seq, "mixing": True},
               Path("checkpoints") / f"mixed_{args.tag}.pt")
    print(f"\nSaved checkpoints/mixed_{args.tag}.pt")

if __name__ == "__main__":
    main()
