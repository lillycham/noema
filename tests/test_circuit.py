import numpy as np
import torch

from noema import circuit, data, gpt, tokens

W = data.WIDTH

def tiny(rope=True):
    torch.manual_seed(0)
    return gpt.GPT(gpt.Config(vocab=tokens.VOCAB, block=data.ROWS * (W + 1), emb=32, heads=2, layers=2, rope=rope)).eval()

def batch(B=4):
    X, _ = data.rollout_batch(np.random.default_rng(0), B, data.RuleSet(2).train, radius=2)
    return X, torch.from_numpy(X)

def test_patterns_are_causal_and_each_row_sums_to_one():
    _, X = batch()
    for w in circuit.all_patterns(tiny(), X):
        assert w.shape == (4, 2, X.shape[1], X.shape[1])
        assert torch.allclose(w.sum(-1), torch.ones(4, 2, X.shape[1]), atol=1e-5)
        assert (w.triu(1) == 0).all()

def test_patterns_reproduce_what_the_attention_computes():
    """Weights times values, through the output projection, is the module's own output."""
    model, (_, X) = tiny(), batch()
    attn, x = model.blocks[0].attn, model.tok(batch()[1])
    B, T, C = x.shape
    v = attn.qkv(x).split(C, dim=-1)[2].view(B, T, attn.heads, -1).transpose(1, 2)
    mine = attn.proj((circuit.patterns(attn, x) @ v).transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(mine, attn(x), atol=1e-5)

def test_transition_ids_are_the_neighbourhoods_that_made_each_cell():
    Xnp, _ = batch()
    ids = circuit.transition_ids(tokens.decode(Xnp, W), 2).reshape(4, data.ROWS, W + 1)
    assert (ids[:, 0] == -1).all() and (ids[:, :, W] == -1).all()           # first row and separators
    assert ids[:, 1:, :W].min() >= 0 and ids[:, 1:, :W].max() < 32

def test_a_zeroed_head_changes_the_output_and_is_put_back():
    model, (_, X) = tiny(), batch()
    before = model(X)
    with circuit.zeroed(model, 0, 1):
        inside = model(X)
    assert not torch.allclose(before, inside)
    assert torch.equal(before, model(X))

def test_zeroing_every_head_of_a_layer_removes_its_attention():
    model, (_, X) = tiny(), batch()
    with circuit.zeroed(model, 0, 0, 1):
        both = model(X)
    block = model.blocks[0]
    original = block.attn.forward
    block.attn.forward = lambda x: torch.zeros_like(x)
    try:
        assert torch.allclose(both, model(X), atol=1e-5)
    finally:
        block.attn.forward = original

def test_match_mass_is_one_for_a_head_that_looks_at_the_matching_cell():
    ids = torch.tensor([[-1, -1, 7, 3, 7, 3]])                  # the cell at 4 (id 7) is predicted from position 3
    T = ids.shape[1]
    weights = torch.zeros(1, 1, T, T)
    weights[..., :, :] = torch.eye(T)                           # everyone looks at themselves
    weights[0, 0, 3, :] = torch.tensor([0, 0, 1.0, 0, 0, 0])    # the query at 3 looks at the earlier 7
    mass, base, found = circuit.match_masses(weights, ids)
    assert found[0] > 0.2 and mass[0] > 0 and 0 < base < 1

def test_offset_profile_finds_a_head_that_looks_a_fixed_distance_back():
    ids = torch.tensor([[-1, 0, 1, 2, 3, 4, 5, 6]])
    T = ids.shape[1]
    weights = torch.zeros(1, 1, T, T)
    for q in range(T):
        weights[0, 0, q, max(q - 2, 0)] = 1.0
    prof = circuit.offset_profile(weights, ids, max_offset=4)
    assert prof[0].argmax() == 2

# ---------- the 2D world ----------
from noema import ca2d, data2d

def test_2d_transition_ids_are_the_neighbourhoods_of_each_frame():
    X, _ = data2d.rollout_batch(np.random.default_rng(0), 3, data2d.RuleSet().train)
    frames = data2d.to_frames(X)
    ids = circuit.transition_ids_2d(frames).reshape(3, data2d.FRAMES, -1)
    C = data2d.HEIGHT * data2d.WIDTH
    assert (ids[:, 0] == -1).all() and (ids[:, :, C] == -1).all()
    assert np.array_equal(ids[:, 1:, :C].reshape(3, -1, data2d.HEIGHT, data2d.WIDTH), ca2d.neighbourhoods(frames[:, :-1]))

def test_the_named_distances_are_where_the_neighbours_are_in_the_tokens():
    """The query that predicts cell (r, c) of frame t + 1 is the token before it. The cell above, in
    frame t, is C + W tokens back, and so on. Checked in the middle of the grid, where nothing wraps."""
    H, W = 5, 5
    C, stride = H * W, H * W + 1
    names = circuit.named_offsets_2d(H, W)
    where = {"up": (-1, 0), "left": (0, -1), "self": (0, 0), "right": (0, 1), "down": (1, 0)}
    r, c, t = 2, 2, 3
    query = (t + 1) * stride + r * W + c - 1
    for d, name in names.items():
        if name in where:
            dr, dc = where[name]
            assert query - d == t * stride + (r + dr) * W + (c + dc)

def test_the_wrapped_distances_are_where_the_wrapped_neighbours_are():
    H, W = 5, 5
    C, stride = H * W, H * W + 1
    names = {v: k for k, v in circuit.named_offsets_2d(H, W).items()}
    t = 2
    for (r, c), name, (nr, nc) in (((1, 0), "left, wrapped", (1, W - 1)), ((1, W - 1), "right, wrapped", (1, 0)),
                                   ((0, 2), "up, wrapped", (H - 1, 2)), ((H - 1, 2), "down, wrapped", (0, 2))):
        query = (t + 1) * stride + r * W + c - 1
        assert query - names[name] == t * stride + nr * W + nc

def test_a_2d_checkpoint_is_read_back_with_its_grid(tmp_path):
    cfg = gpt.Config(vocab=tokens.VOCAB, block=4 * 10, emb=32, heads=2, layers=2)
    path = tmp_path / "w.pt"
    torch.save({"config": cfg.__dict__, "weights": gpt.GPT(cfg).state_dict(), "side": 3, "frames": 4}, path)
    assert circuit.load(path).world == {"side": 3, "frames": 4}
