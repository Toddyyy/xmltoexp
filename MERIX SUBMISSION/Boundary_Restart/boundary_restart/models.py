from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if lengths is not None:
            packed = pack_padded_sequence(
                x,
                lengths.detach().to(device="cpu", dtype=torch.int64),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.encoder(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            x, _ = self.encoder(x)
        x = self.dropout(x)
        logits = self.head(x)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TCNTagger(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for block_idx, out_channels in enumerate(channels):
            dilation = 2 ** block_idx
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Conv1d(in_channels, output_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.head(x)
        return x.squeeze(1) if x.size(1) == 1 else x.transpose(1, 2)


def build_sequence_model(model_type: str, input_dim: int, cfg: dict, output_dim: int = 1) -> nn.Module:
    seq_cfg = cfg.get("sequence", {})
    model_type = model_type.lower()
    if model_type == "bilstm":
        return BiLSTMTagger(
            input_dim=input_dim,
            hidden_dim=int(seq_cfg.get("hidden_dim", 64)),
            num_layers=int(seq_cfg.get("num_layers", 2)),
            dropout=float(seq_cfg.get("dropout", 0.2)),
            output_dim=output_dim,
        )
    if model_type == "tcn":
        channels = [int(v) for v in seq_cfg.get("tcn_channels", [64, 64, 64])]
        return TCNTagger(
            input_dim=input_dim,
            channels=channels,
            kernel_size=int(seq_cfg.get("kernel_size", 3)),
            dropout=float(seq_cfg.get("dropout", 0.2)),
            output_dim=output_dim,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
