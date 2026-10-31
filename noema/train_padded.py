"""Free strides: train on some record lengths, test on another, and on a dimension never seen.

    python -m noema.train_padded --dims 1 2 3 --strides 2 3 4 6 7 8 --held-stride 5 --steps 16000 --tag first

Each training batch has one dimension and one stride (see data_padded.py). At the end every pair of a
dimension from 1 to --test-dim and a stride up to the longest is scored:

    new stride     a trained dimension at --held-stride
    new dimension  --test-dim at a trained stride
    both           --test-dim at --held-stride

"early" is the mean accuracy of records 2 to 4. A later record can come from arithmetic on a period:
every 2D turn has R^4 = I, so p(t) = p(t-1) + p(t-4) - p(t-5) holds with no knowledge of R, from record
5 on. Early records can't use it, so on a held-out pair early is the score to trust.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noema import data_mixed as dx, data_padded as dp, data_torus as dt, gpt
from noema.train_eca import default_device, lr_at, row_curve, train_step
from noema.train_torus import score

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
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3], help="dimensions to train on")
    p.add_argument("--strides", type=int, nargs="+", default=[2, 3, 4, 6, 7, 8], help="strides to train on")
    p.add_argument("--held-stride", type=int, default=5, help="the stride held out in every dimension")
    p.add_argument("--test-dim", type=int, default=4, help="the dimension held out at every stride")
    p.add_argument("--steps-per-seq", type=int, default=dt.STEPS)
    p.add_argument("--learned-positions", action="store_true", help="a position table in place of RoPE")
    p.add_argument("--no-qk-norm", action="store_true")
    p.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="padded")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    assert args.held_stride not in args.strides and args.test_dim not in args.dims

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    size, steps = args.size, args.steps_per_seq
    train = dp.pairs(args.dims, args.strides)
    tested = dp.pairs([*args.dims, args.test_dim], sorted({*args.strides, args.held_stride}))
    cfg = gpt.Config(vocab=dp.filler(size) + 1, block=steps * max(s for _, s in tested), emb=args.emb, heads=args.heads,
                     layers=args.layers, rope=not args.learned_positions, qk_norm=not args.no_qk_norm)
    model = gpt.GPT(cfg).to(args.device)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(gpt.param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95))
    masks = {(d, s): torch.from_numpy(dt.target_mask(steps * s, d, s)).to(args.device) for d, s in tested}
    fixed = {(d, s): torch.from_numpy(dp.batch(np.random.default_rng(100 + 10 * d + s), 256, d, s, size, steps)).to(args.device)
             for d, s in tested}
    ideal = {d: dx.ideal(dx.batch(np.random.default_rng(100 + d), 256, d, size, steps), d, size)
             for d in {*args.dims, args.test_dim}}
    print(f"{gpt.n_parameters(model):,} parameters on {args.device}; train on {train}, hold stride {args.held_stride} "
          f"and d={args.test_dim}; {cfg}", flush=True)

    def marked(d, s):
        return f"d{d}s{s}" + ("*" if d == args.test_dim or s == args.held_stride else "")

    def acc(d, s):
        return score(net, fixed[d, s], masks[d, s], d, size, s)[1]

    t = time.time()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            line = "  ".join(f"{marked(d, s)} {acc(d, s).mean():.3f}" for d, s in tested if d == args.test_dim or s == args.held_stride)
            print(f"step {step:5d}  {line}  ({time.time() - t:.0f}s)", flush=True)
        if step == args.steps:
            break
        d, s = train[rng.integers(0, len(train))]
        X = dp.batch(rng, args.batch, d, s, size, steps)
        train_step(net, opt, torch.from_numpy(X).to(args.device), masks[d, s], lr_at(step, args.steps, args.lr, args.warmup))

    print("\n(* = held out)   mean   early  (ideal mean, early)")
    for d, s in tested:
        curve = acc(d, s)
        print(f"{marked(d, s):8s} {curve.mean():.3f}  {curve[1:4].mean():.3f}  ({ideal[d].mean():.3f}, {ideal[d][1:4].mean():.3f})  {row_curve(curve)}")
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"config": cfg.__dict__, "weights": model.state_dict(), "size": size, "dims": args.dims, "strides": args.strides,
                "held_stride": args.held_stride, "test_dim": args.test_dim, "steps": steps},
               Path("checkpoints") / f"padded_{args.tag}.pt")
    print(f"\nSaved checkpoints/padded_{args.tag}.pt")

if __name__ == "__main__":
    main()
