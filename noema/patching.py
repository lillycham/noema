"""Activation patching on the 2D models: which heads carry which neighbour?

    python -m noema.patching checkpoints/world2d_rope_s0.pt

A query predicts cell (r, c) of a frame from the neighbourhood n of that cell in the frame before.
Flip one neighbour, say the cell above, and n becomes n' with one bit changed, so the true answer
changes too. Take only the cases where the answer really changes and the model has seen both n and
n' earlier in the sequence, so that it could look either up. The model should now follow the flip.

Two runs, the clean one and the corrupted one, give two scores. The score is how far the model's
logits favour the clean answer: positive for the clean run, negative if the corrupted run follows
the flip. Then, one head at a time, the clean run's output of that head at the query position is
written into the corrupted run (denoising: does that one head bring the clean answer back?), and
the corrupted run's output into the clean run (noising: does it lose the answer?). Both are shown as
a share of the whole gap between the clean and the corrupted score.

If the gather heads carry the neighbours, then for the flip of the cell above the head that looks
C + W back should do the work, and for the flip of the cell to the left the head C + 1 back.
"""
import argparse
import contextlib

import numpy as np
import torch

from noema import ca2d, circuit, data, data2d, tokens

WIDTH_1D, RADIUS_1D = data.WIDTH, 2
WEIGHT ={"up": 16, "left": 8, "self": 4, "right": 2, "down": 1}      # what the neighbour counts for in n
STEP = {"up": (-1, 0), "left": (0, -1), "self": (0, 0), "right": (0, 1), "down": (1, 0)}

def expected_distance(direction, side):
    """The distance back at which a gather head would find that neighbour (circuit.named_offsets_2d)."""
    C = side * side
    return {"up": C + side, "left": C + 1, "self": C, "right": C - 1, "down": C - side}[direction]

def wrapped_distance(direction, side):
    """The distance back at which a gather head would find that neighbour across the edge of the grid."""
    C = side * side
    return {"up": C - (side - 1) * side, "left": C - (side - 1), "right": C + (side - 1), "down": C + (side - 1) * side}[direction]

def make_cases(world, rng, n, direction, pool=4096, wrap=False):
    """n cases of a flip of the neighbour in that direction, on held-out rules, in the middle of the
    grid (so nothing wraps), or with wrap on the edge where that neighbour is across the wrap. A dict of
    tensors: clean and corrupted token batches (n, T), the query position q (n,) and the clean answer
    (n,), 0 or 1. Fewer than n if the pool runs out."""
    side, frames = world["side"], world["frames"]
    C = side * side
    stride = C + 1
    X, rules = data2d.rollout_batch(rng, pool, data2d.RuleSet().held, side, side, frames)
    ids = circuit.transition_ids_2d(data2d.to_frames(X, side, side))
    dr, dc = STEP[direction]
    keep = []
    for b in range(pool):
        f, r, c = int(rng.integers(2, frames)), int(rng.integers(1, side - 1)), int(rng.integers(1, side - 1))
        if wrap:
            r = {-1: 0, 1: side - 1}.get(dr, r)
            c = {-1: 0, 1: side - 1}.get(dc, c)
        q = f * stride + r * side + c - 1                           # the token before the cell it predicts
        n_old = int(ids[b, q + 1])
        n_new = n_old ^ WEIGHT[direction]
        answer = int(X[b, q + 1])
        if answer == (int(rules[b]) >> n_new) & 1:                  # the flip must change the answer
            continue
        seen = ids[b, :q + 1]
        if not ((seen == n_old).any() and (seen == n_new).any()):   # and both must be in the context to find
            continue
        keep.append((b, q, (f - 1) * stride + (r + dr) % side * side + (c + dc) % side, answer))
        if len(keep) == n:
            break
    rows = [b for b, _, _, _ in keep]
    clean = torch.from_numpy(X[rows])
    corrupt = clean.clone()
    for i, (_, _, flip, _) in enumerate(keep):
        corrupt[i, flip] = 1 - corrupt[i, flip]
    return dict(clean=clean, corrupt=corrupt, q=torch.tensor([k[1] for k in keep], dtype=torch.long),
                answer=torch.tensor([k[3] for k in keep], dtype=torch.long))

def make_cases_1d(rng, n, offset, pool=4096):
    """make_cases for the 1D radius-2 world: a flip of the cell at that offset in the row before, in the middle
    of the row (so nothing wraps), on held-out rules."""
    stride = WIDTH_1D + 1
    X, rules = data.rollout_batch(rng, pool, data.RuleSet(RADIUS_1D).held, radius=RADIUS_1D)
    ids = circuit.transition_ids(tokens.decode(X, WIDTH_1D), RADIUS_1D)
    rows = X.shape[1] // stride
    keep = []
    for b in range(pool):
        row, c = int(rng.integers(2, rows)), int(rng.integers(RADIUS_1D, WIDTH_1D - RADIUS_1D))
        q = row * stride + c - 1
        n_old = int(ids[b, q + 1])
        n_new = n_old ^ (1 << (RADIUS_1D - offset))
        answer = int(X[b, q + 1])
        if answer == (int(rules[b]) >> n_new) & 1:
            continue
        seen = ids[b, :q + 1]
        if not ((seen == n_old).any() and (seen == n_new).any()):
            continue
        keep.append((b, q, (row - 1) * stride + c + offset, answer))
        if len(keep) == n:
            break
    clean = torch.from_numpy(X[[k[0] for k in keep]])
    corrupt = clean.clone()
    for i, (_, _, flip, _) in enumerate(keep):
        corrupt[i, flip] = 1 - corrupt[i, flip]
    return dict(clean=clean, corrupt=corrupt, q=torch.tensor([k[1] for k in keep], dtype=torch.long),
                answer=torch.tensor([k[3] for k in keep], dtype=torch.long))

def score(logits, q, answer):
    """How far the logits at the query favour the clean answer: (logit of 1 - logit of 0), negated
    when the clean answer is 0. Positive for a model that gives the clean answer."""
    at = logits[torch.arange(len(q)), q, :2]
    return (at[:, 1] - at[:, 0]) * (2 * answer - 1)

@torch.no_grad()
def record(model, X, q):
    """(what every head writes at the query position, for each layer: (B, heads, head size), the logits)."""
    got, idx = {}, torch.arange(len(X))
    with contextlib.ExitStack() as stack:
        for layer in range(model.cfg.layers):
            def edit(out, layer=layer):
                got[layer] = out[idx, :, q].clone()
                return out
            stack.enter_context(circuit.edited(model, layer, edit))
        logits = model(X)
    return [got[layer] for layer in range(model.cfg.layers)], logits

@torch.no_grad()
def with_head_from(model, X, q, layer, head, source):
    """The logits on X when the head of that layer writes source (B, head size) at the query position."""
    idx = torch.arange(len(X))
    def edit(out):
        out = out.clone()
        out[idx, head, q] = source
        return out
    with circuit.edited(model, layer, edit):
        return model(X)

def patch_all(model, cases):
    """{what: array (layers, heads)} for the cases: the mean scores of the clean and the corrupted
    run ('clean', 'corrupt': one number each), and for each head the share of the gap that denoising
    brings back ('denoise') and that noising loses ('noise')."""
    model.eval()
    q, answer = cases["q"], cases["answer"]
    clean_rec, clean_logits = record(model, cases["clean"], q)
    corrupt_rec, corrupt_logits = record(model, cases["corrupt"], q)
    s_clean, s_corrupt = score(clean_logits, q, answer), score(corrupt_logits, q, answer)
    gap = (s_clean.mean() - s_corrupt.mean()).item()
    layers, heads = model.cfg.layers, model.cfg.heads
    denoise, noise = np.zeros((layers, heads)), np.zeros((layers, heads))
    for l in range(layers):
        for h in range(heads):
            back = score(with_head_from(model, cases["corrupt"], q, l, h, clean_rec[l][:, h]), q, answer).mean()
            lost = score(with_head_from(model, cases["clean"], q, l, h, corrupt_rec[l][:, h]), q, answer).mean()
            denoise[l, h] = (back - s_corrupt.mean()).item() / gap
            noise[l, h] = (s_clean.mean() - lost).item() / gap
    return dict(clean=s_clean.mean().item(), corrupt=s_corrupt.mean().item(), gap=gap,
                clean_right=(s_clean > 0).float().mean().item(), follows=(s_corrupt < 0).float().mean().item(),
                denoise=denoise, noise=noise)

def head_distances(model, world, seed=0, n=64):
    """The distance back at which each head puts most of its attention: (layers, heads) ints, to label the
    heads of a table. Measured on rollouts of held-out rules, as in circuit.analyse."""
    side, frames = world["side"], world["frames"]
    X, _ = data2d.rollout_batch(np.random.default_rng(seed), n, data2d.RuleSet().held, side, side, frames)
    ids = torch.from_numpy(circuit.transition_ids_2d(data2d.to_frames(X, side, side)))
    X = torch.from_numpy(X)
    reach = side * side + (side - 1) * side + 5
    total = None
    for part, id_part in zip(X.split(32), ids.split(32)):
        prof = torch.stack([circuit.offset_profile(w, id_part, reach) for w in circuit.all_patterns(model, part)]) * len(part)
        total = prof if total is None else total + prof
    return total.argmax(-1).numpy()

def head_distances_1d(model, seed=0, n=64):
    """head_distances for a 1D radius-2 model."""
    X, _ = data.rollout_batch(np.random.default_rng(seed), n, data.RuleSet(RADIUS_1D).held, radius=RADIUS_1D)
    ids = torch.from_numpy(circuit.transition_ids(tokens.decode(X, WIDTH_1D), RADIUS_1D))
    X = torch.from_numpy(X)
    total = None
    for part, id_part in zip(X.split(32), ids.split(32)):
        prof = torch.stack([circuit.offset_profile(w, id_part, 40) for w in circuit.all_patterns(model, part)]) * len(part)
        total = prof if total is None else total + prof
    return total.argmax(-1).numpy()

def top(table, distances, k=5):
    order = np.dstack(np.unravel_index(np.argsort(-table.ravel()), table.shape))[0][:k]
    return "  ".join(f"L{l}H{h}(d={distances[l, h]}) {table[l, h]:+.2f}" for l, h in order)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wrap", action="store_true", help="2D: flip the neighbour across the edge of the grid")
    args = p.parse_args()

    model = circuit.load(args.checkpoint)
    rng = lambda: np.random.default_rng(args.seed)
    if model.world:
        side = model.world["side"]
        distances = head_distances(model, model.world)
        plans = [(d, f"{d}{' across the edge' if args.wrap else ''}",
                  wrapped_distance(d, side) if args.wrap and d != "self" else expected_distance(d, side),
                  make_cases(model.world, rng(), args.n, d, wrap=args.wrap)) for d in WEIGHT if not (args.wrap and d == "self")]
    else:
        distances = head_distances_1d(model)
        plans = [(o, f"at offset {o:+d}", WIDTH_1D - o, make_cases_1d(rng(), args.n, o)) for o in range(-RADIUS_1D, RADIUS_1D + 1)]
    for _, name, distance, cases in plans:
        if len(cases["q"]) < 20:
            print(f"== flip the neighbour {name}: only {len(cases['q'])} usable cases, skipped\n")
            continue
        r = patch_all(model, cases)
        print(f"== flip the neighbour {name} (a head at d={distance} should carry it), {len(cases['q'])} cases")
        print(f"   score clean {r['clean']:+.2f} ({r['clean_right']:.0%} give the clean answer), "
              f"corrupted {r['corrupt']:+.2f} ({r['follows']:.0%} follow the flip)")
        print(f"   denoising, best heads (share of the gap brought back):  {top(r['denoise'], distances)}")
        print(f"   noising, best heads (share of the gap lost):            {top(r['noise'], distances)}\n")

if __name__ == "__main__":
    main()
