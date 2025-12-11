"""
Tempo prediction module for ScorePerformer.
Predicts global tempo mean and std from score encoding.
FIXED VERSION: Adjusted for actual data ranges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class TempoPredictor(nn.Module):
    """
    Predicts global tempo parameters (mean and std) from score encoding.
    Uses attention pooling to aggregate sequence-level information.
    FIXED: Adjusted output ranges to match actual data distribution
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1,
            # NEW: Configurable tempo ranges based on actual data
            tempo_mean_range: Tuple[float, float] = (20.0, 420.0),  # Expanded range
            tempo_std_range: Tuple[float, float] = (0.1, 10.0)      # Realistic std range
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tempo_mean_range = tempo_mean_range
        self.tempo_std_range = tempo_std_range

        # Attention pooling for sequence aggregation
        self.attention_pool = nn.MultiheadAttention(
            input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Learnable query for attention pooling
        self.global_query = nn.Parameter(torch.randn(1, 1, input_dim))

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # mean_tempo, std_tempo
        )

        # Initialize global query
        nn.init.normal_(self.global_query, std=0.02)
        
        # Print configuration
        print(f"TempoPredictor configured with:")
        print(f"  Mean tempo range: {tempo_mean_range} BPM")
        print(f"  Std tempo range: {tempo_std_range}")

    def forward(
            self,
            score_encoding: Tensor,
            score_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Predict tempo parameters from score encoding.

        Args:
            score_encoding: [batch, seq_len, dim] - encoded score representation
            score_mask: [batch, seq_len] - mask for valid positions (True for valid)

        Returns:
            tempo_params: [batch, 2] - (mean_tempo, std_tempo)
        """
        batch_size = score_encoding.size(0)

        # Expand global query for batch
        query = self.global_query.expand(batch_size, -1, -1)  # [batch, 1, dim]

        # Attention pooling to aggregate sequence information
        # Convert mask: True for valid -> False for attention mask (padding)
        key_padding_mask = None
        if score_mask is not None:
            key_padding_mask = ~score_mask  # Invert: True for padding positions

        pooled_representation, attention_weights = self.attention_pool(
            query=query,
            key=score_encoding,
            value=score_encoding,
            key_padding_mask=key_padding_mask
        )

        # Remove sequence dimension: [batch, 1, dim] -> [batch, dim]
        pooled_representation = pooled_representation.squeeze(1)

        # Predict tempo parameters
        tempo_params = self.predictor(pooled_representation)  # [batch, 2]

        # Apply activations to ensure valid ranges - FIXED RANGES
        # Mean tempo: sigmoid -> [tempo_mean_range[0], tempo_mean_range[1]] BPM
        min_tempo, max_tempo = self.tempo_mean_range
        mean_tempo = torch.sigmoid(tempo_params[:, 0]) * (max_tempo - min_tempo) + min_tempo
        
        # Std tempo: sigmoid -> [tempo_std_range[0], tempo_std_range[1]]
        min_std, max_std = self.tempo_std_range
        std_tempo = torch.sigmoid(tempo_params[:, 1]) * (max_std - min_std) + min_std

        return torch.stack([mean_tempo, std_tempo], dim=1)  # [batch, 2]

    def compute_loss(
            self,
            predicted_tempo: Tensor,
            target_tempo: Tensor,
            loss_type: str = "huber"  # Changed default to huber for robustness
    ) -> Tensor:
        """
        Compute tempo prediction loss.

        Args:
            predicted_tempo: [batch, 2] - predicted (mean, std)
            target_tempo: [batch, 2] - target (mean, std)
            loss_type: "mse", "mae", or "huber"

        Returns:
            loss: scalar tensor
        """
        if loss_type == "mse":
            return F.mse_loss(predicted_tempo, target_tempo)
        elif loss_type == "mae":
            return F.l1_loss(predicted_tempo, target_tempo)
        elif loss_type == "huber":
            # Huber loss is more robust to outliers
            return F.huber_loss(predicted_tempo, target_tempo, delta=1.0)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")


class MultiScaleTempoPredictor(TempoPredictor):
    """
    Enhanced tempo predictor with multi-scale analysis.
    Uses multiple pooling strategies for robust tempo estimation.
    FIXED VERSION with proper ranges
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1,
            tempo_mean_range: Tuple[float, float] = (5.0, 500.0),
            tempo_std_range: Tuple[float, float] = (0.1, 10.0)
    ):
        super().__init__(input_dim, hidden_dim, num_heads, dropout, tempo_mean_range, tempo_std_range)

        # Additional pooling strategies
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Combine different pooling results
        self.fusion_layer = nn.Linear(input_dim * 3, input_dim)  # attention + max + avg

    def forward(
            self,
            score_encoding: Tensor,
            score_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Multi-scale tempo prediction with fixed ranges."""
        batch_size, seq_len, dim = score_encoding.shape

        # 1. Attention pooling (from parent class)
        query = self.global_query.expand(batch_size, -1, -1)
        key_padding_mask = ~score_mask if score_mask is not None else None

        attention_pooled, _ = self.attention_pool(
            query=query,
            key=score_encoding,
            value=score_encoding,
            key_padding_mask=key_padding_mask
        )
        attention_pooled = attention_pooled.squeeze(1)  # [batch, dim]

        # 2. Max pooling
        # Apply mask before pooling
        masked_encoding = score_encoding
        if score_mask is not None:
            mask_expanded = score_mask.unsqueeze(-1).expand_as(score_encoding)
            masked_encoding = score_encoding.masked_fill(~mask_expanded, float('-inf'))

        max_pooled = self.max_pool(masked_encoding.transpose(1, 2)).squeeze(-1)  # [batch, dim]

        # 3. Average pooling
        if score_mask is not None:
            # Compute average only over valid positions
            mask_expanded = score_mask.unsqueeze(-1).expand_as(score_encoding)
            masked_encoding = score_encoding.masked_fill(~mask_expanded, 0.0)
            avg_pooled = masked_encoding.sum(dim=1) / score_mask.sum(dim=1, keepdim=True).float()
        else:
            avg_pooled = score_encoding.mean(dim=1)  # [batch, dim]

        # Combine all pooling results
        combined = torch.cat([attention_pooled, max_pooled, avg_pooled], dim=1)
        fused = self.fusion_layer(combined)  # [batch, dim]

        # Predict tempo parameters
        tempo_params = self.predictor(fused)

        # Apply activations with FIXED ranges
        min_tempo, max_tempo = self.tempo_mean_range
        mean_tempo = torch.sigmoid(tempo_params[:, 0]) * (max_tempo - min_tempo) + min_tempo
        
        min_std, max_std = self.tempo_std_range
        std_tempo = torch.sigmoid(tempo_params[:, 1]) * (max_std - min_std) + min_std

        return torch.stack([mean_tempo, std_tempo], dim=1)