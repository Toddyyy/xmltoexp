import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
# import traditional transformer block
from .transformer import MultiHeadAttention, FeedForward


class ComposerEmbedding(nn.Module):
    """Composer embedding module."""

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


class TempoComposerFusion(nn.Module):
    """
    Fuses tempo and composer information for joint conditioning.
    """

    def __init__(
            self,
            tempo_dim: int = 2,  # mean, std
            composer_dim: int = 128,
            output_dim: int = 128,
            fusion_type: str = "concat"  # "concat", "add", "cross_attention"
    ):
        super().__init__()

        self.tempo_dim = tempo_dim
        self.composer_dim = composer_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        # Tempo projection to higher dimension
        self.tempo_projection = nn.Sequential(
            nn.Linear(tempo_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )

        if fusion_type == "concat":
            # Concatenate tempo and composer, then project
            self.fusion_layer = nn.Sequential(
                nn.Linear(output_dim + composer_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
        elif fusion_type == "add":
            # Project composer to same dim as tempo, then add
            self.composer_projection = nn.Linear(composer_dim, output_dim)
        elif fusion_type == "cross_attention":
            # Cross-attention between tempo and composer
            self.cross_attention = nn.MultiheadAttention(
                output_dim, num_heads=4, batch_first=True
            )
            self.composer_projection = nn.Linear(composer_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
            self,
            tempo_params: Optional[Tensor] = None,
            composer_emb: Optional[Tensor] = None
    ) -> Tensor:
        """
        Fuse tempo and composer information.

        Args:
            tempo_params: [batch, 2] - (mean_tempo, std_tempo), can be None
            composer_emb: [batch, composer_dim] - composer embedding, can be None

        Returns:
            fused_condition: [batch, output_dim] - fused conditioning vector
        """
        batch_size = 1
        if tempo_params is not None:
            batch_size = tempo_params.size(0)
        elif composer_emb is not None:
            batch_size = composer_emb.size(0)

        # Handle missing inputs
        if tempo_params is None:
            tempo_params = torch.zeros(batch_size, self.tempo_dim,
                                       device=composer_emb.device if composer_emb is not None else torch.device('cpu'))

        if composer_emb is None:
            composer_emb = torch.zeros(batch_size, self.composer_dim,
                                       device=tempo_params.device)

        # Project tempo to higher dimension
        tempo_projected = self.tempo_projection(tempo_params)  # [batch, output_dim]

        if self.fusion_type == "concat":
            # Concatenate and project
            concatenated = torch.cat([tempo_projected, composer_emb], dim=1)
            fused = self.fusion_layer(concatenated)

        elif self.fusion_type == "add":
            # Project composer and add
            composer_projected = self.composer_projection(composer_emb)
            fused = tempo_projected + composer_projected

        elif self.fusion_type == "cross_attention":
            # Cross-attention fusion
            composer_projected = self.composer_projection(composer_emb)

            # Add sequence dimension for attention
            tempo_seq = tempo_projected.unsqueeze(1)  # [batch, 1, output_dim]
            composer_seq = composer_projected.unsqueeze(1)  # [batch, 1, output_dim]

            # Cross-attention: tempo attends to composer
            attended, _ = self.cross_attention(
                query=tempo_seq,
                key=composer_seq,
                value=composer_seq
            )

            fused = attended.squeeze(1)  # [batch, output_dim]

        return fused


class EnhancedAdaLayerNorm(nn.Module):
    """
    Enhanced Adaptive Layer Normalization that handles both tempo and composer conditioning.
    """

    def __init__(
            self,
            normalized_shape: int,
            condition_dim: int,  # Combined tempo+composer dimension
            modulation_strength: float = 0.2
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.modulation_strength = modulation_strength

        # LayerNorm without learnable parameters
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # Base scale and shift parameters
        self.base_scale = nn.Parameter(torch.ones(normalized_shape))
        self.base_shift = nn.Parameter(torch.zeros(normalized_shape))

        # Conditioning networks
        self.scale_modulation = nn.Sequential(
            nn.Linear(condition_dim, normalized_shape),
            nn.Tanh()
        )
        self.shift_modulation = nn.Linear(condition_dim, normalized_shape)

        # Initialize modulation networks to zero (identity at start)
        nn.init.zeros_(self.scale_modulation[0].weight)
        nn.init.zeros_(self.scale_modulation[0].bias)
        nn.init.zeros_(self.shift_modulation.weight)
        nn.init.zeros_(self.shift_modulation.bias)

    def forward(self, x: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim] input features
            condition: [batch, condition_dim] combined tempo+composer condition
        Returns:
            output: [batch, seq_len, dim]
        """
        # Layer normalization
        normalized = self.layer_norm(x)

        if condition is None:
            # No conditioning: use base parameters
            return normalized * self.base_scale.view(1, 1, -1) + self.base_shift.view(1, 1, -1)

        # Compute modulation parameters
        scale_mod = self.scale_modulation(condition)  # [batch, dim]
        shift_mod = self.shift_modulation(condition)  # [batch, dim]

        # Combine base and modulation
        final_scale = self.base_scale + self.modulation_strength * scale_mod
        final_shift = self.base_shift + shift_mod

        # Apply modulation
        return normalized * final_scale.unsqueeze(1) + final_shift.unsqueeze(1)


class EnhancedComposerConditionedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with joint tempo+composer conditioning.
    """

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
            condition_dim: int = 128  # Combined tempo+composer dimension
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

        # Enhanced AdaLayerNorm with joint conditioning
        self.self_attn_norm = EnhancedAdaLayerNorm(dim, condition_dim)
        if cross_attend:
            self.cross_attn_norm = EnhancedAdaLayerNorm(dim, condition_dim)
        self.ff_norm = EnhancedAdaLayerNorm(dim, condition_dim)

    def forward(
            self,
            x: Tensor,
            condition: Optional[Tensor] = None,  # Combined tempo+composer condition
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim] Input features
            condition: [batch, condition_dim] Combined tempo+composer condition
            mask: [batch, seq_len] attention mask
            context: [batch, context_len, dim] Cross-attention context
            context_mask: [batch, context_len] context mask
        Returns:
            output: [batch, seq_len, dim] output
        """
        # Self-attention with enhanced AdaLN
        normed_x = self.self_attn_norm(x, condition)
        x = x + self.self_attn(normed_x, mask=mask)

        # Cross-attention with enhanced AdaLN (if enabled)
        if self.cross_attend and context is not None:
            normed_x = self.cross_attn_norm(x, condition)
            x = x + self.cross_attn(normed_x, context=context, context_mask=context_mask)

        # Feed-forward with enhanced AdaLN
        normed_x = self.ff_norm(x, condition)
        x = x + self.ff(normed_x)

        return x


class EnhancedComposerConditionedTransformer(nn.Module):
    """
    Enhanced transformer with joint tempo+composer conditioning.
    """

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
            condition_dim: int = 128
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EnhancedComposerConditionedTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                attn_dropout=attn_dropout,
                causal=causal,
                cross_attend=cross_attend,
                condition_dim=condition_dim
            )
            for _ in range(depth)
        ])

        # Final enhanced AdaLayerNorm
        self.final_norm = EnhancedAdaLayerNorm(dim, condition_dim)

    def forward(
            self,
            x: Tensor,
            condition: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim] Input features
            condition: [batch, condition_dim] Combined tempo+composer condition
        Returns:
            output: [batch, seq_len, dim] Output features
        """
        for layer in self.layers:
            x = layer(x, condition, mask=mask, context=context, context_mask=context_mask)

        x = self.final_norm(x, condition)

        return x