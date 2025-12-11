"""
Transformer Attention Layers with data caching support for inference.

Adapted from: https://github.com/lucidrains/x-transformers
"""

import math
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat


# Positional embeddings
class AbsolutePositionalEmbedding(nn.Module):
    """Learnable absolute positional embeddings."""

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device)
        return self.emb(pos) * self.scale


# Attention components
class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional cross-attention support."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None
    ) -> Tensor:
        b, n, d = x.shape

        # Compute queries
        q = self.to_q(x)

        # Compute keys and values (from context if cross-attention)
        kv_input = context if context is not None else x
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if mask is not None or context_mask is not None:
            if context is not None and context_mask is not None:
                mask_value = -torch.finfo(dots.dtype).max
                context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
                dots = dots.masked_fill(~context_mask, mask_value)
            elif mask is not None:
                mask_value = -torch.finfo(dots.dtype).max
                mask = rearrange(mask, 'b j -> b 1 1 j')
                dots = dots.masked_fill(~mask, mask_value)

        # Apply causal mask if needed
        if self.causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            dots = dots.masked_fill(causal_mask, -torch.finfo(dots.dtype).max)

        # Apply attention
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and optional cross-attention."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        causal: bool = False,
        cross_attend: bool = False
    ):
        super().__init__()
        self.cross_attend = cross_attend

        # Self-attention
        self.self_attn = MultiHeadAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=attn_dropout,
            causal=causal
        )
        self.self_attn_norm = nn.LayerNorm(dim)

        # Cross-attention
        if cross_attend:
            self.cross_attn = MultiHeadAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
                causal=False
            )
            self.cross_attn_norm = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self-attention
        x = x + self.self_attn(self.self_attn_norm(x), mask=mask)

        # Cross-attention (if enabled and context provided)
        if self.cross_attend and context is not None:
            x = x + self.cross_attn(
                self.cross_attn_norm(x),
                context=context,
                context_mask=context_mask
            )

        # Feed-forward
        x = x + self.ff(self.ff_norm(x))

        return x


class Transformer(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        causal: bool = False,
        cross_attend: bool = False
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                attn_dropout=attn_dropout,
                causal=causal,
                cross_attend=cross_attend
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, context=context, context_mask=context_mask)

        return self.norm(x)