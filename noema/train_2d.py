"""Train a small transformer on 2D cellular automata, and look at what it writes.

    python -m noema.train_2d --steps 3000 --tag first

Every sequence has its own random rule (a 32-bit table, see ca2d.py) and first grid, on a torus. A
frame of H x W cells is written line by line, with one separator after it. Whole classes of rules
are held out of training, about 1 in 16. The training loop is the one in train_eca.py: only the data
and the pictures are different. After training it writes 6 rollouts from the first frame or two of
each and saves a picture of them next to the truth in runs/.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noema import ca2d, data2d, data_count, evaluate, gpt, render, tokens
from noema.train_eca import default_device, lr_at, row_curve, train_step

def fixed_batch(rules, seed, B, args):
    X, _ = data2d.rollout_batch(np.random.default_rng(seed), B, rules, args.side, args.side, args.frames)
    return torch.from_numpy(X).to(args.device)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    for name, default in (("emb", 192), ("heads", 6), ("layers", 6)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--side", type=int, default=data2d.WIDTH, help="the grid is side x side (default 5)")
    p.add_argument("--frames", type=int, default=data2d.FRAMES, help="frames in a sequence (default 8)")
    p.add_argument("--prompt-frames", type=int, default=2, help="frames given before the model writes (default 2)")
    p.add_argument("--learned-positions", action="store_true", help="a position table in place of RoPE")
    p.add_argument("--no-qk-norm", action="store_true")
    p.add_argument("--count-rules", action="store_true", help="only rules that count (see data_count.py)")
    p.add_argument("--init", help="start from the weights of this checkpoint: the score at step 0 is the zero-shot one")
    p.add_argument("--save-at", type=int, nargs="*", default=[], help="also save the weights at these steps, to look along the curve")
    p.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="world2d")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rules = data_count.CountRuleSet() if args.count_rules else data2d.RuleSet()
    cells = args.side * args.side
    L = args.frames * (cells + 1)
    cfg = gpt.Config(vocab=tokens.VOCAB, block=L, emb=args.emb, heads=args.heads, layers=args.layers,
                     rope=not args.learned_positions, qk_norm=not args.no_qk_norm)
    model = gpt.GPT(cfg).to(args.device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=args.device)["weights"])
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(gpt.param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95))
    mask = torch.from_numpy(tokens.target_mask(L, cells)).to(args.device)
    seen = fixed_batch(rules.train, 1, 256, args)
    unseen = fixed_batch(rules.held, 2, 256, args)
    print(f"{gpt.n_parameters(model):,} parameters on {args.device}; {args.side}x{args.side} torus, "
          f"{args.frames} frames, {L} tokens; {cfg}", flush=True)
    ideal = data2d.ideal_accuracy_by_frame(data2d.to_frames(seen.cpu().numpy(), args.side, args.side))
    print(f"ideal table-lookup learner, by frame:  {row_curve(ideal)}  (mean {ideal.mean():.3f})", flush=True)
    if args.count_rules:
        grids = data2d.to_frames(seen.cpu().numpy(), args.side, args.side)
        numbers = ca2d.neighbourhoods(grids[:, :-1]).reshape(len(grids), args.frames - 1, cells)
        counting = data_count.count_ideal_from(numbers)
        print(f"ideal learner that knows rules count: {row_curve(counting)}  (mean {counting.mean():.3f})", flush=True)

    t = time.time()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            loss, acc = evaluate.score(net, seen, mask, cells)
            loss_h, acc_h = evaluate.score(net, unseen, mask, cells)
            print(f"step {step:5d}  val loss {loss:.4f}  acc {acc.mean():.3f}  | held-out rules: "
                  f"loss {loss_h:.4f}  acc {acc_h.mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
        if step in args.save_at:
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({"config": cfg.__dict__, "weights": model.state_dict(), "side": args.side, "frames": args.frames},
                       Path("checkpoints") / f"world2d_{args.tag}_at{step}.pt")
        if step == args.steps:
            break
        X, _ = data2d.rollout_batch(rng, args.batch, rules.train, args.side, args.side, args.frames)
        train_step(net, opt, torch.from_numpy(X).to(args.device), mask, lr_at(step, args.steps, args.lr, args.warmup))

    print(f"\nideal learner, by frame:      {row_curve(ideal)}")
    print(f"accuracy by frame, seen rules:  {row_curve(acc)}")
    print(f"accuracy by frame, held out:    {row_curve(acc_h)}")

    # Rollouts: the model gets the first frames and writes the rest. Three rules it trained on, three held out.
    draw = np.random.default_rng(args.seed + 5)
    shown = [int(r) for r in (*rules.train(draw, 3), *rules.held(draw, 3))]
    first = np.random.default_rng(args.seed + 7).integers(0, 2, size=(len(shown), args.side, args.side), dtype=np.uint8)
    truth = ca2d.rollout(first, np.array(shown), args.frames)
    prefix = torch.from_numpy(tokens.encode(truth[:, :args.prompt_frames].reshape(len(shown), args.prompt_frames, cells)))
    written = data2d.to_frames(evaluate.complete(model, prefix.to(args.device), cells).cpu().numpy(), args.side, args.side)
    items = [(r, truth[i], written[i], "held out" if rules.is_held_out(r) else "seen") for i, r in enumerate(shown)]
    out = Path("runs")
    out.mkdir(exist_ok=True)
    render.save_worlds(items, out / f"world2d_{args.tag}_rollouts.png", args.prompt_frames)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"config": cfg.__dict__, "weights": model.state_dict(), "side": args.side, "frames": args.frames},
               Path("checkpoints") / f"world2d_{args.tag}.pt")
    print(f"\nSaved runs/world2d_{args.tag}_rollouts.png and checkpoints/world2d_{args.tag}.pt")

if __name__ == "__main__":
    main()
