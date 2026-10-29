"""Looking for the circuit: which heads look up the rule from earlier rows?

    python -m noema.circuit checkpoints/eca_final_rope_s0.pt           # 1D
    python -m noema.circuit checkpoints/world2d_rope_s0.pt             # 2D: the checkpoint says so

To predict cell (t + 1, i) a learner that works out the rule in context needs the transition it
makes: the neighbourhood number n of (t, i). It has seen the same n before if some earlier cell
(s + 1, j) came from n(s, j) = n, and then that cell is the answer. So the guess is a head that
attends from the place that predicts (t + 1, i) to earlier cells whose own neighbourhood number
matches. For each head this measures the share of its attention that lands on such cells, next to
the share that uniform attention would give.
"""
import argparse
import contextlib

import numpy as np
import torch

from noema import ca2d, data, data2d, evaluate, gpt, tokens

def patterns(attn, x):
    """The attention weights (B, heads, T, T) of an Attention module given its input x (B, T, C)."""
    B, T, C = x.shape
    q, k, _ = (t.view(B, T, attn.heads, -1).transpose(1, 2) for t in attn.qkv(x).split(C, dim=-1))
    q, k = attn.rotate(attn.q_norm(q), attn.k_norm(k))
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
    return scores.softmax(-1)

@torch.no_grad()
def all_patterns(model, X):
    """One (B, heads, T, T) tensor of weights for each layer, on the batch X (B, T) of token IDs."""
    model.eval()
    got, hooks = [], []
    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(lambda m, args, out: got.append(patterns(m, args[0]))))
    try:
        model(X)
    finally:
        for h in hooks:
            h.remove()
    return got

@contextlib.contextmanager
def zeroed(model, layer, *heads):
    """Inside the block, the heads of that layer write nothing: their output is zero before the
    output projection. The weights are not touched. (Blocks for one layer don't nest: name every
    head of the layer in one call.)"""
    attn = model.blocks[layer].attn
    original = attn.forward

    def forward(x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, attn.heads, -1).transpose(1, 2) for t in attn.qkv(x).split(C, dim=-1))
        q, k = attn.rotate(attn.q_norm(q), attn.k_norm(k))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out[:, list(heads)] = 0
        return attn.proj(out.transpose(1, 2).reshape(B, T, C))

    attn.forward = forward
    try:
        yield
    finally:
        attn.forward = original

def transition_ids(grid, radius):
    """For each token position of a rollout (B, rows, W) laid out as tokens: the neighbourhood number
    of the transition that made the cell there, or -1 for a separator or a cell of the first row."""
    B, rows, W = grid.shape
    ids = np.full((B, rows, W + 1), -1, dtype=np.int64)
    ids[:, 1:, :W] = evaluate.neighbourhoods(grid[:, :-1], radius)
    return ids.reshape(B, -1)

def transition_ids_2d(frames):
    """transition_ids for a 2D rollout (B, frames, H, W), laid out as tokens: a frame is one row of
    H * W cells (line by line) and a separator."""
    B, F, H, W = frames.shape
    ids = np.full((B, F, H * W + 1), -1, dtype=np.int64)
    ids[:, 1:, :H * W] = ca2d.neighbourhoods(frames[:, :-1]).reshape(B, F - 1, H * W)
    return ids.reshape(B, -1)

def named_offsets_2d(height, width):
    """{distance back: which neighbour} for a gather head of the 2D world. The query that predicts
    cell i of a frame is at the token before it, so a neighbour at o places from cell i, in the frame
    before, is C - o tokens back, with C the cells in a frame. Up is o = -W: C + W. The cells on an
    edge wrap round, so their neighbours are somewhere else, which gives four more distances."""
    C = height * width
    return {C + width: "up", C + 1: "left", C: "self", C - 1: "right", C - width: "down",
            C - (width - 1): "left, wrapped", C + (width - 1): "right, wrapped",
            C - (height - 1) * width: "up, wrapped", C + (height - 1) * width: "down, wrapped"}

def match_masses(weights, ids):
    """Per head, from the weights of one layer (B, heads, T, T) and ids (B, T): the mean share of
    attention that goes to keys with the same transition as the cell the query predicts, and the
    share uniform attention would give. A query is the place that predicts the next token; only
    queries that predict a cell of rows 1 on are counted."""
    T = ids.shape[1]
    target = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], dim=1)       # what each position predicts
    same = (ids[:, None, :] == target[:, :, None]) & (ids[:, None, :] >= 0)        # (B, q, k)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    same &= causal
    live = (target >= 0)                                                             # (B, q)
    eligible = (ids[:, None, :] >= 0) & causal
    got = (weights * same[:, None]).sum(-1)                                         # (B, heads, q)
    base = same.sum(-1) / eligible.sum(-1).clamp(min=1)                             # (B, q)
    live_h = live[:, None, :]
    has = live & same.any(-1)                                                        # a match exists to find
    found = ((got * has[:, None, :]).sum((0, 2)) / has.sum()).numpy()
    return ((got * live_h).sum((0, 2)) / live.sum()).numpy(), ((base * live).sum() / live.sum()).item(), found

def offset_profile(weights, ids, max_offset=40):
    """Per head, the mean attention at each distance back (query position - key position), 0 to
    max_offset, over queries that predict a cell of rows 1 on: (heads, max_offset + 1)."""
    B, H, T, _ = weights.shape
    target = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], dim=1)
    live = (target >= 0)
    out = torch.zeros(H, max_offset + 1)
    for d in range(max_offset + 1):
        diag = weights.diagonal(offset=-d, dim1=-2, dim2=-1)                         # (B, H, T - d): query q, key q - d
        mask = live[:, d:][:, None, :]
        out[:, d] = (diag * mask).sum((0, 2)) / live.sum()
    return out

def load(path, device="cpu"):
    """The model of a checkpoint. A 2D checkpoint also leaves its grid in model.world."""
    ck = torch.load(path, map_location=device)
    model = gpt.GPT(gpt.Config(**ck["config"])).to(device)
    model.load_state_dict(ck["weights"])
    model.world = {k: ck[k] for k in ("side", "frames") if k in ck}
    return model.eval()

def print_table(title, table):
    print(title)
    print("layer " + " ".join(f"head{h}" for h in range(table.shape[1])))
    for l, row in enumerate(table):
        print(f"  {l}   " + " ".join(f"{v:5.2f}" for v in row))
    print()

def analyse(model, X, ids, cells, chunk=32, names=None, max_offset=40):
    """Print the three pictures for a batch X (B, T) of token IDs and its ids: where each head puts its
    attention on matching cells, at which distances back it looks, and what zeroing it costs. cells
    is the cells in a row or frame. names maps a distance back to what a head would be looking at."""
    names = names or {}
    sums = found = profile = None
    base, count = 0.0, 0
    for part, id_part in zip(X.split(chunk), ids.split(chunk)):
        weights = all_patterns(model, part)
        masses = [match_masses(w, id_part) for w in weights]
        n = len(part)
        mass, hit = np.stack([m for m, _, _ in masses]) * n, np.stack([f for _, _, f in masses]) * n
        prof = torch.stack([offset_profile(w, id_part, max_offset) for w in weights]).numpy() * n
        sums, found, profile = (mass, hit, prof) if sums is None else (sums + mass, found + hit, profile + prof)
        base += masses[0][1] * n
        count += n
    print("share of attention on cells with the same transition as the one being predicted")
    print(f"uniform attention would give {base / count:.3f}\n")
    print_table("on every query", sums / count)
    print_table("on queries where a match exists", found / count)

    print("heads that look at one distance back (30% or more of their attention), with the cell that is there")
    for l, layer in enumerate(profile / count):
        for h, head in enumerate(layer):
            top = np.argsort(head)[::-1][:3]
            if head[top[0]] >= 0.3:
                print(f"  L{l}H{h}  " + "  ".join(f"d={d}{f' ({names[d]})' if d in names else ''}: {head[d]:.2f}" for d in top))
    print()

    mask = torch.from_numpy(tokens.target_mask(X.shape[1], cells))
    _, whole = evaluate.score(model, X, mask, cells)
    print(f"accuracy with every head in place: {whole.mean():.3f}\n")
    table = np.zeros((model.cfg.layers, model.cfg.heads))
    for l in range(model.cfg.layers):
        for h in range(model.cfg.heads):
            with zeroed(model, l, h):
                table[l, h] = evaluate.score(model, X, mask, cells)[1].mean()
    print_table("accuracy with one head zeroed", table)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--radius", type=int, default=2, help="1D checkpoints only")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = load(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    if model.world:                                                 # a 2D checkpoint
        side, frames = model.world["side"], model.world["frames"]
        X, _ = data2d.rollout_batch(rng, args.n, data2d.RuleSet().held, side, side, frames)
        ids = torch.from_numpy(transition_ids_2d(data2d.to_frames(X, side, side)))
        cells, names, reach = side * side, named_offsets_2d(side, side), side * side + (side - 1) * side + 5
    else:
        X, _ = data.rollout_batch(rng, args.n, data.RuleSet(args.radius).held, radius=args.radius)
        ids = torch.from_numpy(transition_ids(tokens.decode(X, data.WIDTH), args.radius))
        cells, names, reach = data.WIDTH, None, 40
    analyse(model, torch.from_numpy(X), ids, cells, args.chunk, names, reach)

if __name__ == "__main__":
    main()
