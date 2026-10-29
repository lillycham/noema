"""Tokens for a rollout: 0 and 1 are the cells, SEP ends a row.

A rollout of n rows and W cells is n * (W + 1) tokens. With W = 15 a row takes 16 tokens, so the
cell above position p is at p - 16: a fixed distance back, whatever the row. A model with
relative positions (RoPE) can use that directly.

The model is never told the rule, so no token says what it is.
"""
import numpy as np

ZERO, ONE, SEP = 0, 1, 2
VOCAB = 3

def encode(rows):
    """rows (..., n, W) of 0 and 1 -> tokens (..., n * (W + 1)), int64."""
    rows = np.asarray(rows)
    sep = np.full(rows.shape[:-1] + (1,), SEP, dtype=rows.dtype)
    both = np.concatenate([rows, sep], axis=-1)                  # (..., n, W + 1)
    return both.reshape(*rows.shape[:-2], -1).astype(np.int64)

def decode(tokens, width):
    """tokens (..., n * (W + 1)) -> rows (..., n, W). The inverse of encode.
    Raises ValueError if the length is not a whole number of rows, or a separator is missing."""
    tokens = np.asarray(tokens)
    stride = width + 1
    if tokens.shape[-1] % stride:
        raise ValueError(f"{tokens.shape[-1]} tokens is not a whole number of rows of {stride}")
    rows = tokens.reshape(*tokens.shape[:-1], -1, stride)
    if not (rows[..., width] == SEP).all():
        raise ValueError("a row does not end in a separator")
    return rows[..., :width].astype(np.uint8)

def target_mask(length, width):
    """Which targets count in the loss, for a sequence of this length. The targets are tokens[1:],
    so entry i is the token at position i + 1. True for the cells of the second row and later.

    Not the first row, which is random and cannot be predicted. Not the separators, which are
    always in the same place and say nothing about the rule.
    """
    position = np.arange(1, length)
    return (position >= width + 1) & (position % (width + 1) < width)
