from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMNoteToBeatEmbedder(nn.Module):
    """
    Encode note-level features with a BiLSTM and aggregate them into beat-level embeddings.

    Inputs:
      - note_feats: [B, N, F]
      - beat_ids: [B, N] (>=0 for valid beats, -1 for padding)
      - num_beats: optional int to fix output length
      - attn_mask: optional [B, N] bool mask of valid notes
    Outputs:
      - beat_emb: [B, max_beats, D]
      - beat_mask: [B, max_beats] bool (True if beat has notes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_dim = int(out_dim) if out_dim is not None else self.hidden_dim * 2

        rnn_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )

        rnn_out_dim = self.hidden_dim * 2
        self.proj = None
        if self.out_dim != rnn_out_dim:
            self.proj = nn.Linear(rnn_out_dim, self.out_dim)

    def forward(
        self,
        note_feats: Tensor,
        beat_ids: Tensor,
        num_beats: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        device = note_feats.device
        bsz, _, _ = note_feats.shape
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=1).to(torch.long)
            if lengths.max().item() == 0:
                empty = torch.zeros(bsz, 0, self.out_dim, device=device)
                mask = torch.zeros(bsz, 0, dtype=torch.bool, device=device)
                return empty, mask

            safe_lengths = lengths.clone()
            zero_mask = safe_lengths == 0
            if zero_mask.any():
                safe_lengths[zero_mask] = 1
                note_feats = note_feats.clone()
                note_feats[zero_mask] = 0.0

            packed = pack_padded_sequence(
                note_feats,
                safe_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            enc_packed, _ = self.rnn(packed)
            enc, _ = pad_packed_sequence(
                enc_packed,
                batch_first=True,
                total_length=note_feats.size(1),
            )
        else:
            enc, _ = self.rnn(note_feats)
        if self.proj is not None:
            enc = self.proj(enc)

        valid_mask = beat_ids >= 0
        if attn_mask is not None:
            valid_mask = valid_mask & attn_mask.bool()

        if num_beats is None:
            max_beats = beat_ids[valid_mask].max().item() + 1 if valid_mask.any() else 0
        else:
            max_beats = int(num_beats)

        if max_beats <= 0:
            empty = torch.zeros(bsz, 0, self.out_dim, device=device)
            mask = torch.zeros(bsz, 0, dtype=torch.bool, device=device)
            return empty, mask

        if valid_mask.any():
            valid_mask = valid_mask & (beat_ids < max_beats)

        beat_sum = torch.zeros(bsz, max_beats, self.out_dim, device=device)
        beat_cnt = torch.zeros(bsz, max_beats, 1, device=device)

        for b in range(bsz):
            v = valid_mask[b]
            if not v.any():
                continue
            ids = beat_ids[b, v]
            feats = enc[b, v]
            beat_sum[b].index_add_(0, ids, feats)
            beat_cnt[b].index_add_(
                0,
                ids,
                torch.ones((ids.shape[0], 1), device=device),
            )

        beat_emb = beat_sum / beat_cnt.clamp(min=1.0)
        beat_mask = beat_cnt.squeeze(-1) > 0
        return beat_emb, beat_mask
