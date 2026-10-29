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
