"""
Beat-level boundary prediction model.

Input (note-level): score-only note features shaped [batch, notes, feature_dim] with a companion
beat_ids tensor [batch, notes] mapping每个音符属于哪个 beat（0-based，填充音符可设为 -1）。
先用 BiLSTM 编码 note 序列，再按 beat 聚合成 beat-level embedding，
最后用 beat-level Transformer 输出 per-beat 边界 logits/probabilities。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import math
import torch
from torch import nn, Tensor

from beat_embedding import BiLSTMNoteToBeatEmbedder

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = self._build_pe(max_len, device=None)
        self.register_buffer("pe", pe)

    def _build_pe(self, max_len: int, device: Optional[torch.device]) -> Tensor:
        position = torch.arange(max_len, device=device).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Extend positional encodings on demand.
            self.pe = self._build_pe(seq_len, device=x.device)
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
    fixed_beats: Optional[int] = None
    include_empty_beats: bool = False
    dual_head: bool = False
    note_rnn_hidden: Optional[int] = None
    note_rnn_layers: int = 1
    note_rnn_dropout: float = 0.0
    performer_cond: bool = False
    performer_emb_dim: int = 32
    performer_vocab_size: int = 0


class BeatBoundaryModel(nn.Module):
    """
    BiLSTM note encoder + beat-level Transformer encoder，输出 beat-level 边界概率。
    """

    def __init__(
        self,
        config: BeatBoundaryConfig,
        pos_weight: Optional[float] = None,
        loss_type: str = "bce",
        prob_loss_type: str = "bce",
        prob_pos_weight: Optional[float] = None,
        prob_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.pos_weight = pos_weight
        self.loss_type = loss_type.lower()
        self.dual_head = bool(config.dual_head)
        self.prob_loss_type = prob_loss_type.lower()
        self.prob_pos_weight = prob_pos_weight if prob_pos_weight is not None else pos_weight
        self.prob_loss_weight = float(prob_loss_weight)
        self.performer_cond = bool(config.performer_cond)
        rnn_hidden = config.note_rnn_hidden
        if rnn_hidden is None:
            rnn_hidden = max(1, config.d_model // 2)

        self.note_embedder = BiLSTMNoteToBeatEmbedder(
            input_dim=config.input_dim,
            hidden_dim=rnn_hidden,
            num_layers=config.note_rnn_layers,
            dropout=config.note_rnn_dropout,
            out_dim=config.d_model,
        )
        self.beat_pos_enc = PositionalEncoding(config.d_model, dropout=config.dropout, max_len=config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.beat_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.d_model, 1)
        self.performer_emb = None
        self.performer_proj = None
        if self.performer_cond:
            if config.performer_vocab_size <= 0:
                raise ValueError("performer_vocab_size must be > 0 when performer_cond is enabled.")
            self.performer_emb = nn.Embedding(
                int(config.performer_vocab_size),
                int(config.performer_emb_dim),
                padding_idx=0,
            )
            self.performer_proj = nn.Linear(int(config.performer_emb_dim), config.d_model, bias=False)
            nn.init.zeros_(self.performer_proj.weight)
        self.loss_fn, self.loss_on_logits = self._build_loss_fn(self.loss_type, pos_weight)
        self.head_prob = None
        self.prob_loss_fn = None
        self.prob_loss_on_logits = None
        if self.dual_head:
            self.head_prob = nn.Linear(config.d_model, 1)
            self.prob_loss_fn, self.prob_loss_on_logits = self._build_loss_fn(
                self.prob_loss_type, self.prob_pos_weight
            )

    @staticmethod
    def _build_loss_fn(loss_type: str, pos_weight: Optional[float]):
        loss_type = loss_type.lower()
        if loss_type == "bce":
            if pos_weight is None:
                return nn.BCEWithLogitsLoss(reduction="none"), True
            pos_w = torch.tensor([float(pos_weight)])
            return nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_w), True
        if loss_type == "mse":
            return nn.MSELoss(reduction="none"), False
        if loss_type == "l1":
            return nn.L1Loss(reduction="none"), False
        if loss_type in {"huber", "smoothl1"}:
            return nn.SmoothL1Loss(reduction="none"), False
        raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'bce', 'mse', 'l1', or 'huber'.")

    def forward(
        self,
        note_feats: Tensor,
        beat_ids: Tensor,
        num_beats: Optional[int] = None,
        num_beats_per_sample: Optional[Tensor] = None,
        performer_ids: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        labels_prob: Optional[Tensor] = None,
        output_head: str = "dist",
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], Optional[Tensor]]:
        """
        Args:
            note_feats: [batch, notes, input_dim] note-level features.
            beat_ids: [batch, notes] int，>=0 的 beat index，填充位置可用 -1。
            num_beats: 可选，输出的 beat 序列长度；若未提供，使用 beat_ids 中的 max+1。
            num_beats_per_sample: 可选 [batch]，每个样本的有效 beat 数。
            attn_mask: 可选 [batch, notes] bool，True 表示该 note 有效。
            labels: 可选 [batch, beats] 目标标签（默认 dist 或 ratio）。
            labels_prob: 可选 [batch, beats] 概率标签（dual head 时使用，非二值）。
            output_head: dist|prob|both，决定返回哪个 logits。
        Returns:
            logits: [batch, beats] 或 {"dist": ..., "prob": ...}
            loss: scalar (若提供 labels)
        """
        device = note_feats.device
        bsz = note_feats.size(0)

        if num_beats is None and self.config.fixed_beats is not None:
            num_beats = int(self.config.fixed_beats)

        beat_emb, beat_mask = self.note_embedder(
            note_feats,
            beat_ids=beat_ids,
            num_beats=num_beats,
            attn_mask=attn_mask,
        )
        max_beats = beat_emb.size(1)
        if max_beats == 0:
            logits = torch.zeros(bsz, 0, device=device)
            return logits, None

        beat_valid = None
        if num_beats_per_sample is not None:
            num_beats_per_sample = num_beats_per_sample.to(device)
            beat_valid = torch.arange(max_beats, device=device).unsqueeze(0) < num_beats_per_sample.unsqueeze(1)

        if beat_valid is None:
            if self.config.include_empty_beats:
                attn_mask_beats = torch.ones_like(beat_mask, dtype=torch.bool, device=device)
            else:
                attn_mask_beats = beat_mask
        else:
            if self.config.include_empty_beats:
                attn_mask_beats = beat_valid
            else:
                attn_mask_beats = beat_valid & beat_mask

        x = self.beat_pos_enc(beat_emb)
        beat_key_padding = ~attn_mask_beats.bool() if attn_mask_beats is not None else None
        enc = self.beat_encoder(x, src_key_padding_mask=beat_key_padding)  # [B, beats, d_model]
        logits_dist = self.head(enc).squeeze(-1)  # [B, beats]
        logits_prob = None
        if self.dual_head:
            logits_prob = self.head_prob(enc).squeeze(-1)
        if self.performer_cond and performer_ids is not None:
            performer_ids = performer_ids.to(device)
            if performer_ids.dim() == 0:
                performer_ids = performer_ids.unsqueeze(0)
            performer_ids = performer_ids.view(-1).long()
            cond = self.performer_proj(self.performer_emb(performer_ids))  # [B, d_model]
            delta = torch.einsum("btd,bd->bt", enc, cond) / math.sqrt(cond.size(-1))
            logits_dist = logits_dist + delta
            if logits_prob is not None:
                logits_prob = logits_prob + delta

        loss = None
        if labels is not None:
            labels, attn_mask_beats = self._align_labels_and_mask(
                labels, attn_mask_beats, max_beats, device
            )
            if self.loss_on_logits:
                per_token = self.loss_fn(logits_dist, labels)
            else:
                per_token = self.loss_fn(torch.sigmoid(logits_dist), labels)
            loss = (per_token * attn_mask_beats).sum() / attn_mask_beats.sum().clamp(min=1)

            if self.dual_head and labels_prob is not None and self.prob_loss_weight != 0.0:
                labels_prob = self._align_labels(labels_prob, max_beats, device)
                if self.prob_loss_on_logits:
                    per_prob = self.prob_loss_fn(logits_prob, labels_prob)
                else:
                    per_prob = self.prob_loss_fn(torch.sigmoid(logits_prob), labels_prob)
                prob_loss = (per_prob * attn_mask_beats).sum() / attn_mask_beats.sum().clamp(min=1)
                loss = loss + self.prob_loss_weight * prob_loss

        if output_head == "both":
            outputs = {"dist": logits_dist}
            if self.dual_head:
                outputs["prob"] = logits_prob
            return outputs, loss
        if output_head == "prob":
            if not self.dual_head:
                raise ValueError("output_head='prob' requested but dual_head is disabled.")
            return logits_prob, loss
        return logits_dist, loss

    @staticmethod
    def _align_labels(labels: Tensor, max_beats: int, device: torch.device) -> Tensor:
        if labels.shape[1] == max_beats:
            return labels
        if labels.shape[1] > max_beats:
            return labels[:, :max_beats]
        pad = max_beats - labels.shape[1]
        return torch.cat([labels, torch.zeros(labels.size(0), pad, device=device)], dim=1)

    @staticmethod
    def _align_labels_and_mask(
        labels: Tensor, beat_mask: Tensor, max_beats: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        if labels.shape[1] == max_beats:
            return labels, beat_mask
        if labels.shape[1] > max_beats:
            labels = labels[:, :max_beats]
            beat_mask = beat_mask[:, :max_beats]
            return labels, beat_mask
        pad = max_beats - labels.shape[1]
        labels = torch.cat([labels, torch.zeros(labels.size(0), pad, device=device)], dim=1)
        pad_mask = torch.zeros(beat_mask.size(0), pad, device=device, dtype=torch.bool)
        beat_mask = torch.cat([beat_mask, pad_mask], dim=1)
        return labels, beat_mask
