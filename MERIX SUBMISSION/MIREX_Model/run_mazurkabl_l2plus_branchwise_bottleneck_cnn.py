from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
RICH_RUN_SCRIPT = MIREX / "run_mazurkabl_l2plus_rich_midibert_mlp_cnn.py"
OUT_DIR = MIREX / "mazurkabl_l2plus_branchwise_bottleneck_cnn"

HIDDEN_DIM = 768
BRANCH_DIM = 24
SCALAR_DIM = 7
BEAT_EMB_DIM = 128


def load_rich_runner():
    spec = importlib.util.spec_from_file_location("mazurka_rich_runner_for_branchwise", RICH_RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RICH_RUN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_rich_runner_for_branchwise"] = module
    spec.loader.exec_module(module)
    return module


runner = load_rich_runner()


class BranchwiseBottleneckCNN(nn.Module):
    def __init__(self, *, base_dim: int, cfg: dict):
        super().__init__()
        dropout = float(cfg.get("sequence", {}).get("dropout", 0.2))
        self.base_dim = int(base_dim)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(HIDDEN_DIM, BRANCH_DIM),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(6)
            ]
        )
        self.beat_mlp = nn.Sequential(
            nn.Linear(6 * BRANCH_DIM + SCALAR_DIM, BEAT_EMB_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cnn = runner.build_sequence_model(
            "cnn",
            input_dim=self.base_dim + BEAT_EMB_DIM,
            cfg=cfg,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.base_dim > 0:
            base_x = x[..., : self.base_dim]
            rich_x = x[..., self.base_dim :]
        else:
            base_x = None
            rich_x = x
        branch_parts = []
        for idx, layer in enumerate(self.branches):
            start = idx * HIDDEN_DIM
            end = start + HIDDEN_DIM
            branch_parts.append(layer(rich_x[..., start:end]))
        scalars = rich_x[..., HIDDEN_DIM * 6 : HIDDEN_DIM * 6 + SCALAR_DIM]
        beat_emb = self.beat_mlp(torch.cat([*branch_parts, scalars], dim=-1))
        if base_x is not None:
            beat_emb = torch.cat([base_x, beat_emb], dim=-1)
        return self.cnn(beat_emb, lengths=lengths)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_norm(features: dict[str, np.ndarray], pieces: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([features[p] for p in pieces], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def build_setting_features(setting: str, base_features, rich_features, pieces, seed: int):
    rng = np.random.default_rng(seed)
    out = {}
    for piece in pieces:
        if setting == "random_branchwise":
            rich = rng.standard_normal(rich_features[piece].shape).astype(np.float32)
        else:
            rich = rich_features[piece].astype(np.float32)
        if setting in {"branchwise_rich_only"}:
            out[piece] = rich
        else:
            out[piece] = np.concatenate([base_features[piece].astype(np.float32), rich], axis=1)
    base_dim = 0 if setting == "branchwise_rich_only" else base_features[pieces[0]].shape[1]
    return out, base_dim


def train_one(cfg, features, labels, train_pieces, *, base_dim: int, seed: int, device: torch.device):
    set_seed(seed)
    mean, std = fit_norm(features, train_pieces)
    model = BranchwiseBottleneckCNN(base_dim=base_dim, cfg=cfg).to(device)
    y_all = np.concatenate([labels[p] for p in train_pieces], axis=0)
    pos = float(y_all.sum())
    neg = float(len(y_all) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    seq_cfg = cfg.get("sequence", {})
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(seq_cfg.get("epochs", 18))
    patience = int(seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _epoch in range(1, epochs + 1):
        order = list(train_pieces)
        random.shuffle(order)
        losses = []
        model.train()
        for piece in order:
            x = ((features[piece] - mean) / std)[None].astype(np.float32)
            y = labels[piece][None].astype(np.float32)
            lengths = torch.tensor([x.shape[1]], dtype=torch.int64, device=device)
            opt.zero_grad()
            logits = model(torch.from_numpy(x).to(device), lengths=lengths)
            loss = loss_fn(logits, torch.from_numpy(y).to(device))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        value = float(np.mean(losses))
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


@torch.no_grad()
def predict(model, features, pieces, mean, std, device):
    model.eval()
    out = {}
    for piece in pieces:
        x = ((features[piece] - mean) / std)[None].astype(np.float32)
        lengths = torch.tensor([x.shape[1]], dtype=torch.int64, device=device)
        logits = model(torch.from_numpy(x).to(device), lengths=lengths)
        out[piece] = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out


def run_setting(setting, cfg, pieces, labels, base_features, rich_features, folds, device):
    rows = []
    th_total = runner.Result()
    den_total = runner.Result()
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        features, base_dim = build_setting_features(
            setting,
            base_features,
            rich_features,
            pieces,
            seed=200000 + fold_idx,
        )
        model, mean, std = train_one(
            cfg,
            features,
            labels,
            train_pieces,
            base_dim=base_dim,
            seed=9900 + fold_idx,
            device=device,
        )
        train_scores = predict(model, features, train_pieces, mean, std, device)
        val_scores = predict(model, features, val_pieces, mean, std, device)
        threshold, train_metric = runner.base.choose_threshold(
            train_scores,
            {p: labels[p] for p in train_pieces},
            tolerance=1,
        )
        th, den = runner.evaluate_fold(labels, val_scores, val_pieces, threshold, train_pieces)
        for attr in ("pred", "true", "match", "wr_num", "wr_den"):
            setattr(th_total, attr, getattr(th_total, attr) + getattr(th, attr))
            setattr(den_total, attr, getattr(den_total, attr) + getattr(den, attr))
        row = {
            "setting": setting,
            "fold": fold_idx,
            "threshold": float(threshold),
            "train_f1_tol1": float(train_metric.f1),
            **runner.pack("threshold", th),
            **runner.pack("density", den),
        }
        rows.append(row)
        print(
            f"{setting} fold {fold_idx}: "
            f"threshold UP/WR/F1={row['threshold_UP']:.3f}/{row['threshold_WR']:.3f}/{row['threshold_f1']:.3f}; "
            f"density UP/WR/F1={row['density_UP']:.3f}/{row['density_WR']:.3f}/{row['density_f1']:.3f}"
        )
    return pd.DataFrame(rows), {"setting": setting, **runner.pack("threshold", th_total), **runner.pack("density", den_total)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()
    pieces, labels, _ = runner.base.load_l2plus_weighted_labels()
    base_features, feature_cols = runner.load_piece_features(pieces, cfg)
    rich_features = runner.load_rich_features(pieces)
    missing = sorted(set(pieces) - set(rich_features))
    if missing:
        raise RuntimeError(f"Missing rich features: {missing}")
    bad = [
        (p, len(base_features[p]), len(labels[p]), len(rich_features[p]))
        for p in pieces
        if len(base_features[p]) != len(labels[p]) or len(labels[p]) != len(rich_features[p])
    ]
    if bad:
        raise RuntimeError(f"Length mismatch: {bad}")
    pieces = sorted(pieces)
    folds = runner.base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()
    settings = ["random_branchwise", "branchwise_rich_only", "handcrafted_plus_branchwise"]
    print(
        f"device={device}; branch_dim={BRANCH_DIM}; scalar_dim={SCALAR_DIM}; "
        f"beat_emb_dim={BEAT_EMB_DIM}; settings={settings}"
    )
    fold_frames = []
    aggregates = []
    for setting in settings:
        fold_df, aggregate = run_setting(setting, cfg, pieces, labels, base_features, rich_features, folds, device)
        fold_frames.append(fold_df)
        aggregates.append(aggregate)
    pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "rich_dir": str(runner.RICH_DIR),
                "branch_dim": BRANCH_DIM,
                "scalar_dim": SCALAR_DIM,
                "branchwise_concat_dim": 6 * BRANCH_DIM + SCALAR_DIM,
                "beat_emb_dim": BEAT_EMB_DIM,
                "settings": settings,
                "pieces": pieces,
                "folds": folds,
                "feature_columns": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
