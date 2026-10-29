"""Free-running: how far does a trained model drift from the truth, against an ideal learner?

    python -m noema.free_running checkpoints/eca_final_rope_s0.pt [more checkpoints] --radius 2

The model gets the first rows and writes all the rest from its own output. So does the ideal
learner of evaluate.ideal_free_running. Both see the same prompts, on rules the model trained on and
rules it never saw. Printed, row by row: the share of cells that differ from the truth.
"""
import argparse

import numpy as np
import torch

from noema import data, evaluate, gpt, tokens

def load(path, device):
    ck = torch.load(path, map_location=device)
    model = gpt.GPT(gpt.Config(**ck["config"])).to(device)
    model.load_state_dict(ck["weights"])
    return model.eval()

def row_curve(values):
    return " ".join(f"{a:.2f}" for a in values)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--prompt-rows", type=int, default=3)
    p.add_argument("--n", type=int, default=512, help="rollouts for each of seen and held-out rules")
    p.add_argument("--chunk", type=int, default=64, help="rollouts written at once: no KV cache, so memory grows with this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    width, k = data.WIDTH, args.prompt_rows
    rules = data.RuleSet(args.radius)
    for kind, pick in (("seen", rules.train), ("held out", rules.held)):
        X, _ = data.rollout_batch(np.random.default_rng(args.seed), args.n, pick, radius=args.radius)
        truth = tokens.decode(X, width)
        ideal = evaluate.ideal_free_running(truth, k, args.radius, np.random.default_rng(args.seed))
        print(f"\n== {kind} rules, {args.n} rollouts, wrong cells by row from row {k}")
        print(f"{'ideal learner':28s} {row_curve(evaluate.wrong_by_row(ideal, truth, k))}"
              f"   perfect to the end: {(ideal == truth).all(axis=(1, 2)).mean():.2f}")
        for path in args.checkpoints:
            model = load(path, args.device)
            prefix = torch.from_numpy(X[:, :k * (width + 1)])
            chunks = [evaluate.complete(model, part.to(args.device), width).cpu() for part in prefix.split(args.chunk)]
            written = tokens.decode(torch.cat(chunks).numpy(), width)
            name = path.split("eca_")[-1].removesuffix(".pt")
            print(f"{name:28s} {row_curve(evaluate.wrong_by_row(written, truth, k))}"
                  f"   perfect to the end: {(written == truth).all(axis=(1, 2)).mean():.2f}")

if __name__ == "__main__":
    main()
