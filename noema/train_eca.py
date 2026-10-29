"""Train a small transformer on elementary cellular automata, and look at what it writes.

    python -m noema.train_eca --steps 3000 --slow-ca      # until step() in ca.py is written
    python -m noema.train_eca --steps 3000                # with your step()

Every sequence has its own random rule and first row. 32 of the 256 rules are held out of
training, so the evaluation can tell a model that works out the rule in context from one that
learned the rules it saw. After training it writes 6 rollouts from the first few rows of each and
saves a picture of them next to the truth in runs/.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from noema import ca, data, evaluate, gpt, render, tokens

SHOWN = [(30, "chaotic"), (90, "Sierpinski"), (110, "complex"), (184, "traffic"), (54, "complex"), (22, "chaotic")]

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"

def lr_at(step, steps, peak, warmup, floor=0.1):
    """Linear warm-up, then a cosine down to floor * peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))

def train_step(model, opt, X, mask, lr):
    for group in opt.param_groups:
        group["lr"] = lr
    loss = gpt.masked_cross_entropy(model(X[:, :-1]), X[:, 1:], mask)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()

def fixed_batch(rules, seed, B, step_fn, device):
    X, _ = data.rollout_batch(np.random.default_rng(seed), B, rules, step_fn=step_fn)
    return torch.from_numpy(X).to(device)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    for name, default in (("emb", 192), ("heads", 6), ("layers", 6)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--heldout", type=int, default=32, help="rules kept out of training (default 32)")
    p.add_argument("--prompt-rows", type=int, default=3, help="rows given before the model writes (default 3)")
    p.add_argument("--learned-positions", action="store_true", help="a position table in place of RoPE")
    p.add_argument("--no-qk-norm", action="store_true")
    p.add_argument("--slow-ca", action="store_true", help="use reference_step, until step() is written")
    p.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    step_fn = ca.reference_step if args.slow_ca else None
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_rules, held_rules = data.split_rules(args.heldout)
    width, L = data.WIDTH, data.ROWS * (data.WIDTH + 1)
    cfg = gpt.Config(vocab=tokens.VOCAB, block=L, emb=args.emb, heads=args.heads, layers=args.layers,
                     rope=not args.learned_positions, qk_norm=not args.no_qk_norm)
    model = gpt.GPT(cfg).to(args.device)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")      # TF32 for the matrix multiplications
    net = torch.compile(model) if args.compile else model      # same weights; save and sample from model
    opt =torch.optim.AdamW(gpt.param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95))
    mask = torch.from_numpy(tokens.target_mask(L, width)).to(args.device)
    seen = fixed_batch(train_rules, 1, 256, step_fn, args.device)
    unseen = fixed_batch(held_rules, 2, 256, step_fn, args.device) if args.heldout else None
    print(f"{gpt.n_parameters(model):,} parameters on {args.device}; {len(train_rules)} rules to train on,"
          f" {len(held_rules)} held out; {cfg}", flush=True)

    t = time.time()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            loss, acc = evaluate.score(net, seen, mask, width)
            line = f"step {step:5d}  val loss {loss:.4f}  acc {acc.mean():.3f}"
            if unseen is not None:
                loss_h, acc_h = evaluate.score(net, unseen, mask, width)
                line += f"  | held-out rules: loss {loss_h:.4f}  acc {acc_h.mean():.3f}"
            print(f"{line}  ({time.time() - t:.0f}s)", flush=True)
        if step == args.steps:
            break
        X, _ = data.rollout_batch(rng, args.batch, train_rules, step_fn=step_fn)
        train_step(net, opt, torch.from_numpy(X).to(args.device), mask, lr_at(step, args.steps, args.lr, args.warmup))

    print("\naccuracy by row, seen rules:  " + " ".join(f"{a:.2f}" for a in acc))
    if unseen is not None:
        print("accuracy by row, held out:    " + " ".join(f"{a:.2f}" for a in acc_h))

    # Rollouts: the model gets the first rows and writes the rest.
    rule_numbers = np.array([r for r, _ in SHOWN])
    first = np.random.default_rng(args.seed + 7).integers(0, 2, size=(len(SHOWN), width), dtype=np.uint8)
    truth = ca.rollout(first, rule_numbers, data.ROWS, step_fn)
    prefix = torch.from_numpy(tokens.encode(truth[:, :args.prompt_rows])).to(args.device)
    written = tokens.decode(evaluate.complete(model, prefix, width).cpu().numpy(), width)
    held = set(held_rules.tolist())
    items = [(r, truth[i], written[i], f"{name}, {'held out' if r in held else 'seen'}")
             for i, (r, name) in enumerate(SHOWN)]
    out = Path("runs")
    out.mkdir(exist_ok=True)
    render.save_rollouts(items, out / f"eca_{args.tag}_rollouts.png", args.prompt_rows)
    print(f"\nrule 110, truth then model:\n{render.show(truth[2])}\n\n{render.show(written[2])}")
    checkpoints = Path("checkpoints")
    checkpoints.mkdir(exist_ok=True)
    torch.save({"config": cfg.__dict__, "weights": model.state_dict()}, checkpoints / f"eca_{args.tag}.pt")
    print(f"\nSaved runs/eca_{args.tag}_rollouts.png and checkpoints/eca_{args.tag}.pt")

if __name__ == "__main__":
    main()
