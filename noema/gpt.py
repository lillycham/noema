"""A small decoder-only transformer with the choices most current models make.

    - Pre-norm blocks with RMSNorm: no mean subtraction and no bias, just a scale.
    - RoPE: positions turn q and k, so attention sees the distance between tokens and the model
      has no table of position vectors. (Su et al., 2021)
    - QK-norm: q and k are normalised per head before the dot product, which keeps the attention
      logits from growing and lets the learning rate be higher.
    - SwiGLU feed-forward, with the hidden size chosen to keep the weights of a 4x MLP. (Shazeer, 2020)
    - No biases anywhere. GPT-2's scaled init for the layers that write to the residual stream.

Adapted from the GPT in tiny-lm (core/gpt.py, commit f5c1135) with GPT-2's choices swapped for
these. In tiny-lm's 3.5M TinyStories ablation, RoPE was worth about 0.12 nats and SwiGLU about
0.03. This file has no KV cache and no dropout: the sequences are short and the data never repeats.

Two switches are kept for comparison runs: rope=False gives a learned table of positions instead,
and qk_norm=False takes out the q and k norms.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class Config:
    vocab: int
    block: int              # the longest sequence, in tokens
    emb: int = 192
    heads: int = 6          # heads of emb / heads numbers each
    layers: int = 6
    rope: bool = True
    qk_norm: bool = True

class RMSNorm(nn.Module):
    """x / rms(x) * weight. Like LayerNorm without the mean and the bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def rope_tables(block, hs, base=10000.0):
    """cos and sin of the turning angle of every position, each (block, hs).

    RoPE turns every pair of numbers in a query or key by an angle that grows with the position.
    The dot product of a query at position m and a key at position n then depends on m - n, the
    distance, and not on m and n themselves. The pairs are (x[i], x[i + hs/2]), and pair i turns
    by position * base^(-2i/hs): fast for the first pairs and slow for the last.
    """
    speed = base ** (-torch.arange(0, hs, 2).float() / hs)      # (hs/2,): one speed for each pair
    angle = torch.arange(block).float()[:, None] * speed        # (block, hs/2)
    angle = torch.cat([angle, angle], dim=-1)                   # (block, hs): both halves of a pair
    return angle.cos(), angle.sin()

def rotate_half(x):
    """(a, b) -> (-b, a) for the two halves of the last axis: a quarter turn of every pair."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def apply_rope(x, cos, sin):
    """Turn every pair of x (..., T, hs) by its angle: (a, b) -> (a cos - b sin, b cos + a sin)."""
    return x * cos + rotate_half(x) * sin

class Attention(nn.Module):
    """Causal self-attention, all heads together: x (B, T, C) -> (B, T, C)."""

    def __init__(self, cfg):
        super().__init__()
        C, hs = cfg.emb, cfg.emb // cfg.heads
        if C % cfg.heads or (cfg.rope and hs % 2):
            raise ValueError("emb must be a multiple of heads, and with rope each head needs an even size")
        self.heads = cfg.heads
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        self.q_norm = RMSNorm(hs) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(hs) if cfg.qk_norm else nn.Identity()
        self.rope = cfg.rope
        if cfg.rope:
            cos, sin = rope_tables(cfg.block, hs)
            # Buffers, not parameters: nothing to train, and rebuilt on load, so not in the checkpoint.
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def rotate(self, q, k, start=0):
        """RoPE for q and k, both (B, heads, T, hs), for positions start to start + T."""
        if not self.rope:
            return q, k
        T = q.shape[2]
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in self.qkv(x).split(C, dim=-1))
        q, k = self.rotate(self.q_norm(q), self.k_norm(k))      # norm first, then turn
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))

def swiglu_hidden(C):
    """The hidden size that keeps the weights of a 4C MLP: three C x H matrices against two
    C x 4C, so H = 8C/3, rounded to a multiple of 8."""
    return max(1, round(8 * C / 3 / 8)) * 8

class FeedForward(nn.Module):
    """SwiGLU: silu(x Wg) * (x Wv), then Wd. One Linear makes the gate and the value together."""

    def __init__(self, C):
        super().__init__()
        H = swiglu_hidden(C)
        self.gate_value = nn.Linear(C, 2 * H, bias=False)
        self.down = nn.Linear(H, C, bias=False)

    def forward(self, x):
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.emb), Attention(cfg)
        self.norm2, self.ffwd = RMSNorm(cfg.emb), FeedForward(cfg.emb)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffwd(self.norm2(x))

class GPT(nn.Module):
    """Token embeddings -> blocks -> RMSNorm -> logits. The output layer is not tied to the
    embeddings: with a vocabulary of three tokens, tying would save nothing."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.emb)
        self.pos = None if cfg.rope else nn.Embedding(cfg.block, cfg.emb)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.emb)
        self.out = nn.Linear(cfg.emb, cfg.vocab, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:       # 2 writes into the residual stream for each layer
            for layer in (b.attn.proj, b.ffwd.down):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    def forward(self, X):
        """X (B, T) token IDs -> logits (B, T, vocab)."""
        T = X.shape[1]
        if T > self.cfg.block:
            raise ValueError(f"{T} tokens is more than the block of {self.cfg.block}")
        x = self.tok(X)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=X.device))
        return self.out(self.norm(self.blocks(x)))

def n_parameters(model):
    return sum(p.numel() for p in model.parameters())

def param_groups(model, weight_decay):
    """Weight decay on the matrices (Linear and Embedding weights), none on the RMSNorm scales."""
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    return [dict(params=matrices, weight_decay=weight_decay), dict(params=vectors, weight_decay=0.0)]

def masked_cross_entropy(logits, targets, mask):
    """Mean cross-entropy over the positions where mask is True. logits (B, T, V), targets (B, T),
    mask (T,) bool."""
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")      # (B, T)
    return (loss * mask).sum() / (mask.sum() * loss.shape[0])
