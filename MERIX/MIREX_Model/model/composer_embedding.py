import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
# import traditional transformer block
from .transformer import MultiHeadAttention, FeedForward

class ComposerEmbedding(nn.Module):
    
    def __init__(
        self,
        num_composers: int,
        embedding_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.composer_embedding = nn.Embedding(num_composers, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.composer_embedding.weight, std=0.02)
    
    def forward(self, composer_ids: Tensor) -> Tensor:
        """
        Args:
            composer_ids: [batch_size] composer ID
        Returns:
            composer_emb: [batch_size, embedding_dim] composer embedding
        """
        emb = self.composer_embedding(composer_ids)
        emb = self.norm(emb)
        emb = self.dropout(emb)
        return emb

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization
    """

    def __init__(
            self,
            normalized_shape: int,
            composer_dim: int,
            modulation_strength: float = 0.2
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.modulation_strength = modulation_strength

        #LayerNorm
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # scale and shift parameters
        self.base_scale = nn.Parameter(torch.ones(normalized_shape))
        self.base_shift = nn.Parameter(torch.zeros(normalized_shape))

        # composer condition
        # Generate scale modulation
        self.scale_modulation = nn.Sequential(
            nn.Linear(composer_dim, normalized_shape),
            nn.Tanh()
        )

        # Generate shift modulation
        self.shift_modulation = nn.Linear(composer_dim, normalized_shape)

        nn.init.zeros_(self.scale_modulation[0].weight)
        nn.init.zeros_(self.scale_modulation[0].bias)
        nn.init.zeros_(self.shift_modulation.weight)
        nn.init.zeros_(self.shift_modulation.bias)

    def forward(self, x: Tensor, composer_emb: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim] input features
            composer_emb: [batch, composer_dim] composer embedding
        Returns:
            output: [batch, seq_len, dim]
        """

        if composer_emb is None:
            normalized = self.layer_norm(x)
            return normalized * self.base_scale.unsqueeze(0).unsqueeze(0) + self.base_shift.unsqueeze(0).unsqueeze(0)
        normalized = self.layer_norm(x)

        # Calculate modulation parameters
        scale_mod = self.scale_modulation(composer_emb)  # [batch, dim]
        shift_mod = self.shift_modulation(composer_emb)  # [batch, dim]

        final_scale = self.base_scale + self.modulation_strength * scale_mod
        final_shift = self.base_shift + shift_mod

        output = normalized * final_scale.unsqueeze(1) + final_shift.unsqueeze(1)

        return output

class ComposerConditionedTransformerBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            ff_mult: int = 4,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            causal: bool = False,
            cross_attend: bool = False,
            composer_dim: int = 128
    ):
        super().__init__()

        self.cross_attend = cross_attend

        # Self-attention
        self.self_attn = MultiHeadAttention(
            dim=dim, heads=heads, dim_head=dim_head,
            dropout=attn_dropout, causal=causal
        )

        # Cross-attention (optional)
        if cross_attend:
            self.cross_attn = MultiHeadAttention(
                dim=dim, heads=heads, dim_head=dim_head,
                dropout=attn_dropout, causal=False
            )

        # Feed-forward
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

        # Use AdaLayerNorm
        self.self_attn_norm = AdaLayerNorm(dim, composer_dim)
        if cross_attend:
            self.cross_attn_norm = AdaLayerNorm(dim, composer_dim)
        self.ff_norm = AdaLayerNorm(dim, composer_dim)

    def forward(
            self,
            x: Tensor,
            composer_emb: Tensor,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim] Input features
            composer_emb: [batch, composer_dim] Composer Embedding
            mask: [batch, seq_len] attention mask
            context: [batch, context_len, dim] Cross-attention context
            context_mask: [batch, context_len] context mask
        Returns:
            output: [batch, seq_len, dim] output
        """
        # Self-attention with AdaLN
        normed_x = self.self_attn_norm(x, composer_emb)
        x = x + self.self_attn(normed_x, mask=mask)

        # Cross-attention with AdaLN (if enabled)
        if self.cross_attend and context is not None:
            normed_x = self.cross_attn_norm(x, composer_emb)
            x = x + self.cross_attn(normed_x, context=context, context_mask=context_mask)

        # Feed-forward with AdaLN
        normed_x = self.ff_norm(x, composer_emb)
        x = x + self.ff(normed_x)

        return x


class ComposerConditionedTransformer(nn.Module):
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
            cross_attend: bool = False,
            composer_dim: int = 128
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ComposerConditionedTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                attn_dropout=attn_dropout,
                causal=causal,
                cross_attend=cross_attend,
                composer_dim=composer_dim
            )
            for _ in range(depth)
        ])

        # The final AdaLayerNorm
        self.final_norm = AdaLayerNorm(dim, composer_dim)

    def forward(
            self,
            x: Tensor,
            composer_emb: Tensor,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim] Input features
            composer_emb: [batch, composer_dim] Composer Embedding
        Returns:
            output: [batch, seq_len, dim] Output characteristics
        """
        for layer in self.layers:
            x = layer(x, composer_emb, mask=mask, context=context, context_mask=context_mask)

        x = self.final_norm(x, composer_emb)

        return x