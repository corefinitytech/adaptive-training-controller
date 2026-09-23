"""A tiny, from-scratch decoder-only character transformer.

Built directly on `torch.nn` primitives (no `transformers` dependency, matching the
project's own core-package constraint) and deliberately small: sized to run this whole
example's four controller comparisons in minutes on this project's Mac CPU/MPS target,
not to produce good text.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape
        q, k, v = self.qkv(x).split(embed_dim, dim=2)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyCharTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
