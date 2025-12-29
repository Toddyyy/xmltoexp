"""
Beat-level boundary prediction model.

Input: score-only features shaped [batch, beats, feature_dim].
Output: per-beat boundary logits/probabilities shaped [batch, beats].
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


@dataclass
class BeatBoundaryConfig:
    input_dim: int = 64
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 4096


class BeatBoundaryModel(nn.Module):
    """
    Transformer encoder that predicts beat-level boundary probabilities.
    """

    def __init__(self, config: BeatBoundaryConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, dropout=config.dropout, max_len=config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.d_model, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        score_feats: Tensor,
        attn_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            score_feats: [batch, beats, input_dim] score features per beat.
            attn_mask: optional boolean mask [batch, beats], True for valid tokens.
            labels: optional target probs [batch, beats] in [0,1].
        Returns:
            logits: [batch, beats]
            loss: scalar (if labels provided) else None
        """
        x = self.input_proj(score_feats)  # [B, T, d_model]
        x = self.pos_enc(x)

        # TransformerEncoder expects src_key_padding_mask with True for padding positions
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask.bool()

        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, d_model]
        logits = self.head(enc).squeeze(-1)  # [B, T]

        loss = None
        if labels is not None:
            loss_mask = attn_mask if attn_mask is not None else torch.ones_like(labels, dtype=torch.bool)
            per_token = self.loss_fn(logits, labels)
            loss = (per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return logits, loss
