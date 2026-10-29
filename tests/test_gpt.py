"""The model: tiny versions on the CPU, a few seconds in all."""
import math

import pytest
import torch
import torch.nn.functional as F

from noema import gpt

TINY = dict(vocab=3, block=32, emb=32, heads=2, layers=2)

def tiny(**options):
    torch.manual_seed(0)
    return gpt.GPT(gpt.Config(**TINY, **options))

def tokens(B=2):
    torch.manual_seed(1)
    return torch.randint(0, TINY["vocab"], (B, TINY["block"]))

VARIANTS = [dict(), dict(rope=False), dict(qk_norm=False), dict(rope=False, qk_norm=False)]
every_variant = pytest.mark.parametrize("options", VARIANTS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()) or "default")

# ---------- the shape of the model ----------
@every_variant
def test_forward_shape(options):
    X = tokens()
    assert tiny(**options)(X).shape == (*X.shape, TINY["vocab"])

@every_variant
def test_no_biases_anywhere(options):
    assert not [n for n, _ in tiny(**options).named_parameters() if n.endswith("bias")]

def test_the_default_model_is_about_2_point_7_million_weights():
    """emb 192, 6 layers: the same non-embedding size as tiny-lm's 3.5M stories model."""
    model = gpt.GPT(gpt.Config(vocab=3, block=256))
    assert 2.6e6 < gpt.n_parameters(model) < 2.7e6

def test_rope_means_no_position_table_and_nothing_extra_in_the_checkpoint():
    model = tiny()
    assert model.pos is None
    assert not [k for k in model.state_dict() if k.endswith(("cos", "sin"))]
    assert tiny(rope=False).pos is not None

def test_qk_norm_is_there_when_asked_for():
    assert isinstance(tiny().blocks[0].attn.q_norm, gpt.RMSNorm)
    assert not isinstance(tiny(qk_norm=False).blocks[0].attn.q_norm, gpt.RMSNorm)

def test_every_parameter_gets_a_gradient():
    model, X = tiny(), tokens()
    F.cross_entropy(model(X).transpose(1, 2), X).backward()
    assert not [n for n, p in model.named_parameters() if p.grad is None]

def test_too_long_a_sequence_is_refused():
    with pytest.raises(ValueError):
        tiny()(torch.zeros(1, TINY["block"] + 1, dtype=torch.long))

def test_rope_needs_an_even_head_size():
    with pytest.raises(ValueError):
        gpt.GPT(gpt.Config(vocab=3, block=16, emb=30, heads=2))      # heads of 15

# ---------- behaviour ----------
@every_variant
@torch.no_grad()
def test_causal(options):
    """Changing token t must not change any output before t."""
    model = tiny(**options).eval()
    X = tokens(B=1)
    before = model(X)
    X[0, 6] = (X[0, 6] + 1) % TINY["vocab"]
    after = model(X)
    assert torch.allclose(before[0, :6], after[0, :6])
    assert not torch.allclose(before[0, 6], after[0, 6])

def test_scaled_init_shrinks_the_layers_that_write_to_the_stream():
    torch.manual_seed(0)
    model = gpt.GPT(gpt.Config(vocab=3, block=16, emb=64, heads=4, layers=8))
    want = 0.02 / math.sqrt(2 * 8)
    for b in model.blocks:
        for layer in (b.attn.proj, b.ffwd.down):
            assert layer.weight.std().item() == pytest.approx(want, rel=0.15)

# ---------- RMSNorm, SwiGLU, the loss ----------
def test_rmsnorm_gives_unit_root_mean_square():
    x = torch.randn(4, 7, 16) * 5
    y = gpt.RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 7), atol=1e-4)

def test_rmsnorm_does_not_subtract_the_mean():
    """The one thing that makes it different from LayerNorm, which would give a mean of 0."""
    x = torch.randn(3, 16) + 10
    assert (gpt.RMSNorm(16)(x).mean(-1) > 0.9).all()

@pytest.mark.parametrize("C", [192, 384, 768])
def test_swiglu_keeps_the_weights_of_a_4x_mlp(C):
    ff = gpt.FeedForward(C)
    assert ff.gate_value.weight.numel() + ff.down.weight.numel() == 2 * C * 4 * C

def test_swiglu_formula():
    """silu(x Wg) * (x Wv), then Wd. The fused layer holds Wg on top of Wv."""
    torch.manual_seed(0)
    ff, x = gpt.FeedForward(12), torch.randn(2, 5, 12)
    H, W = gpt.swiglu_hidden(12), ff.gate_value.weight
    want = (F.silu(x @ W[:H].T) * (x @ W[H:].T)) @ ff.down.weight.T
    assert torch.allclose(ff(x), want, atol=1e-5)

def test_masked_cross_entropy_counts_only_the_masked_positions():
    torch.manual_seed(0)
    logits, targets = torch.randn(3, 6, 3), torch.randint(0, 3, (3, 6))
    mask = torch.tensor([False, False, True, True, False, True])
    per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    assert gpt.masked_cross_entropy(logits, targets, mask).item() == pytest.approx(per_token[:, mask].mean().item())

def test_param_groups_put_every_parameter_in_exactly_one_group():
    model = tiny()
    matrices, vectors = gpt.param_groups(model, 0.1)
    ids = [id(p) for g in (matrices, vectors) for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert matrices["weight_decay"] == 0.1 and vectors["weight_decay"] == 0.0

# ---------- RoPE ----------
def attn():
    torch.manual_seed(0)
    return gpt.Attention(gpt.Config(vocab=3, block=32, emb=12, heads=3))       # head size 4: two pairs

def test_rope_scores_depend_on_the_distance():
    """The point of RoPE: q . k for a query at m and a key at n is the same when both move by the same step."""
    a = attn()
    torch.manual_seed(1)
    q, k = torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)
    turned = lambda x, p: a.rotate(x, x, start=p)[0]
    score = lambda m, n: (turned(q, m) * turned(k, n)).sum(-1)
    assert torch.allclose(score(5, 2), score(9, 6), atol=1e-5)
    assert torch.allclose(score(5, 2), score(25, 22), atol=1e-4)
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-3)

def test_rope_is_a_turn():
    """A turn keeps the length of every vector, and at position 0 it does nothing."""
    a = attn()
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8, 4)
    y, _ = a.rotate(x, x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.equal(y[:, :, 0], x[:, :, 0])

def test_rope_matches_complex_numbers():
    """The same turn as multiplying a + ib by e^(i angle), for the pairs (x[i], x[i + hs/2])."""
    hs, T = 4, 6
    torch.manual_seed(1)
    x = torch.randn(1, 1, T, hs)
    z = torch.complex(x[..., :hs // 2], x[..., hs // 2:])
    angle = torch.arange(T)[:, None] * 10000.0 ** (-torch.arange(0, hs, 2) / hs)
    turned = z * torch.polar(torch.ones_like(angle), angle)
    assert torch.allclose(attn().rotate(x, x)[0], torch.cat([turned.real, turned.imag], dim=-1), atol=1e-5)

def test_rope_is_used_by_the_model():
    """A model that built the tables but never turned q and k would match one without."""
    model = tiny().eval()
    X = tokens()
    with torch.no_grad():
        with_rope = model(X)
        for b in model.blocks:
            b.attn.rope = False
        assert not torch.allclose(with_rope, model(X))
