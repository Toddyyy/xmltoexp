from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = SRC / "experiments"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "mazurkabl_l2plus_weighted_auto_meter.yaml"
BASE_SCRIPT = EXPERIMENTS / "run_mazurkabl_l2plus_weighted_target_experiment.py"
RICH_DIR = DATA / "features" / "mazurkabl_midibert_rich_beat_features"
OUT_DIR = RESULTS / "mazurkabl_l2plus_rich_midibert_mlp_cnn"
BEAT_MLP_DIM = 128


def load_base():
    spec = importlib.util.spec_from_file_location("mazurka_l2plus_rich_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_l2plus_rich_base"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
sys.path.insert(0, str(SRC.resolve()))
from boundary_restart.models import build_sequence_model  # noqa: E402
from boundary_restart.table_io import feature_columns, load_table  # noqa: E402


@dataclass
class Result:
    pred: int = 0
    true: int = 0
    match: int = 0
    wr_num: float = 0.0
    wr_den: float = 0.0


class BeatMLPCNN(nn.Module):
    def __init__(self, *, base_dim: int, rich_dim: int, cfg: dict):
        super().__init__()
        self.base_dim = int(base_dim)
        self.rich_dim = int(rich_dim)
        self.beat_mlp = nn.Sequential(
            nn.Linear(self.rich_dim, 512),
            nn.GELU(),
            nn.Dropout(float(cfg.get("sequence", {}).get("dropout", 0.2))),
            nn.Linear(512, BEAT_MLP_DIM),
            nn.GELU(),
        )
        cnn_input_dim = self.base_dim + BEAT_MLP_DIM
        self.cnn = build_sequence_model("cnn", input_dim=cnn_input_dim, cfg=cfg)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.base_dim > 0:
            base_x = x[..., : self.base_dim]
            rich_x = x[..., self.base_dim :]
            z = torch.cat([base_x, self.beat_mlp(rich_x)], dim=-1)
        else:
            z = self.beat_mlp(x)
        return self.cnn(z, lengths=lengths)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    beat_table = Path(cfg.get("data", {}).get("beat_table_path", ""))
    if beat_table and not beat_table.is_absolute():
        cfg["data"]["beat_table_path"] = str(ROOT / beat_table)
    return cfg


def select_feature_columns(cfg: dict, columns: list[str]) -> list[str]:
    feature_cfg = cfg.get("features", {})
    include = feature_cfg.get("include")
    exclude = set(feature_cfg.get("exclude", []))
    selected = list(columns)
    if include:
        include_set = set(include)
        selected = [col for col in selected if col in include_set]
    if exclude:
        selected = [col for col in selected if col not in exclude]
    return [col for col in selected if col != "protocol_split"]


def load_piece_features(pieces: list[str], cfg: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    df = load_table(Path(cfg["data"]["beat_table_path"]))
    df = df[df["piece_id"].isin(pieces)].copy()
    cols = select_feature_columns(cfg, feature_columns(df))
    out = {}
    for piece in pieces:
        piece_df = df[df["piece_id"] == piece].sort_values(["beat_idx", "sample_id"])
        one = piece_df.groupby("beat_idx", as_index=False).first().sort_values("beat_idx")
        got = one["beat_idx"].to_numpy(dtype=int)
        expected = np.arange(len(one), dtype=int)
        if not np.array_equal(got, expected):
            raise RuntimeError(f"{piece}: non-contiguous beat_idx")
        out[piece] = one[cols].to_numpy(dtype=np.float32)
    return out, cols


def load_rich_features(pieces: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        path = RICH_DIR / f"{piece}_midibert_rich_beat_features.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            out[piece] = np.asarray(data["rich_beat_features"], dtype=np.float32)
    return out


def build_setting_features(
    setting: str,
    base_features: dict[str, np.ndarray],
    rich_features: dict[str, np.ndarray],
    pieces: list[str],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], str, int, int]:
    if setting == "baseline_cnn":
        return base_features, "cnn", base_features[pieces[0]].shape[1], 0
    rich_dim = rich_features[pieces[0]].shape[1]
    rng = np.random.default_rng(seed)
    out = {}
    for piece in pieces:
        base_x = base_features[piece].astype(np.float32)
        if setting == "random_rich_mlp128":
            rich_x = rng.standard_normal(rich_features[piece].shape).astype(np.float32)
            out[piece] = np.concatenate([base_x, rich_x], axis=1)
        elif setting == "rich_midibert_mlp128_only":
            out[piece] = rich_features[piece].astype(np.float32)
        elif setting == "handcrafted_plus_rich_midibert_mlp128":
            out[piece] = np.concatenate([base_x, rich_features[piece].astype(np.float32)], axis=1)
        else:
            raise ValueError(setting)
    base_dim = 0 if setting == "rich_midibert_mlp128_only" else base_features[pieces[0]].shape[1]
    return out, "mlp_cnn", base_dim, rich_dim


def fit_norm(features: dict[str, np.ndarray], pieces: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([features[p] for p in pieces], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def make_model(model_kind: str, cfg: dict, input_dim: int, base_dim: int, rich_dim: int) -> nn.Module:
    if model_kind == "cnn":
        return build_sequence_model("cnn", input_dim=input_dim, cfg=cfg)
    if model_kind == "mlp_cnn":
        return BeatMLPCNN(base_dim=base_dim, rich_dim=rich_dim, cfg=cfg)
    raise ValueError(model_kind)


def train_one(
    cfg: dict,
    model_kind: str,
    input_dim: int,
    base_dim: int,
    rich_dim: int,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    train_pieces: list[str],
    seed: int,
    device: torch.device,
):
    set_seed(seed)
    mean, std = fit_norm(features, train_pieces)
    model = make_model(model_kind, cfg, input_dim, base_dim, rich_dim).to(device)
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
        model.train()
        order = list(train_pieces)
        random.shuffle(order)
        losses = []
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
def predict(model, features: dict[str, np.ndarray], pieces: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device):
    model.eval()
    out = {}
    for piece in pieces:
        x = ((features[piece] - mean) / std)[None].astype(np.float32)
        lengths = torch.tensor([x.shape[1]], dtype=torch.int64, device=device)
        logits = model(torch.from_numpy(x).to(device), lengths=lengths)
        out[piece] = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out


def match_true_indices(pred: np.ndarray, true: np.ndarray, tolerance: int) -> list[int]:
    used = set()
    matched_true = []
    for p in pred.tolist():
        best = None
        best_dist = tolerance + 1
        for j, t in enumerate(true.tolist()):
            if j in used:
                continue
            dist = abs(int(p) - int(t))
            if dist <= tolerance and dist < best_dist:
                best = j
                best_dist = dist
        if best is not None:
            used.add(best)
            matched_true.append(int(true[best]))
    return matched_true


def evaluate_fold(labels, scores, val_pieces, threshold, train_pieces) -> tuple[Result, Result]:
    th = Result()
    den = Result()
    for piece in val_pieces:
        true = np.flatnonzero(labels[piece] >= base.EVENT_MIN).astype(np.int32)
        pred_th = base.extract_events(scores[piece], threshold=threshold)
        m_th = base.metrics_from_events(pred_th, true, tolerance=1)
        mt = match_true_indices(pred_th, true, tolerance=1)
        th.pred += m_th.pred_events
        th.true += m_th.true_events
        th.match += m_th.matches
        th.wr_num += float(labels[piece][mt].sum()) if mt else 0.0
        th.wr_den += float(labels[piece][true].sum()) if len(true) else 0.0

        expected = base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
        pred_den = base.extract_top_density(scores[piece], expected)
        m_den = base.metrics_from_events(pred_den, true, tolerance=1)
        md = match_true_indices(pred_den, true, tolerance=1)
        den.pred += m_den.pred_events
        den.true += m_den.true_events
        den.match += m_den.matches
        den.wr_num += float(labels[piece][md].sum()) if md else 0.0
        den.wr_den += float(labels[piece][true].sum()) if len(true) else 0.0
    return th, den


def pack(prefix: str, totals: Result) -> dict[str, float | int]:
    up = totals.match / totals.pred if totals.pred else 0.0
    recall = totals.match / totals.true if totals.true else 0.0
    wr = totals.wr_num / totals.wr_den if totals.wr_den else 0.0
    f1 = 2 * up * recall / (up + recall) if up + recall else 0.0
    return {
        f"{prefix}_pred_events": int(totals.pred),
        f"{prefix}_true_events": int(totals.true),
        f"{prefix}_matches_tol1": int(totals.match),
        f"{prefix}_UP": up,
        f"{prefix}_recall": recall,
        f"{prefix}_WR": wr,
        f"{prefix}_f1": f1,
    }


def run_setting(setting, cfg, pieces, labels, base_features, rich_features, folds, device):
    rows = []
    th_total = Result()
    den_total = Result()
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        features, model_kind, base_dim, rich_dim = build_setting_features(
            setting,
            base_features,
            rich_features,
            pieces,
            seed=100000 + fold_idx,
        )
        model, mean, std = train_one(
            cfg,
            model_kind,
            features[pieces[0]].shape[1],
            base_dim,
            rich_dim,
            features,
            labels,
            train_pieces,
            seed=9400 + fold_idx,
            device=device,
        )
        train_scores = predict(model, features, train_pieces, mean, std, device)
        val_scores = predict(model, features, val_pieces, mean, std, device)
        threshold, train_metric = base.choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)
        th, den = evaluate_fold(labels, val_scores, val_pieces, threshold, train_pieces)
        for attr in ("pred", "true", "match", "wr_num", "wr_den"):
            setattr(th_total, attr, getattr(th_total, attr) + getattr(th, attr))
            setattr(den_total, attr, getattr(den_total, attr) + getattr(den, attr))
        row = {
            "setting": setting,
            "fold": fold_idx,
            "threshold": float(threshold),
            "train_f1_tol1": float(train_metric.f1),
            **pack("threshold", th),
            **pack("density", den),
        }
        rows.append(row)
        print(
            f"{setting} fold {fold_idx}: "
            f"threshold UP/WR/F1={row['threshold_UP']:.3f}/{row['threshold_WR']:.3f}/{row['threshold_f1']:.3f}; "
            f"density UP/WR/F1={row['density_UP']:.3f}/{row['density_WR']:.3f}/{row['density_f1']:.3f}"
        )
    return pd.DataFrame(rows), {"setting": setting, **pack("threshold", th_total), **pack("density", den_total)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    pieces, labels, _ = base.load_l2plus_weighted_labels()
    base_features, feature_cols = load_piece_features(pieces, cfg)
    rich_features = load_rich_features(pieces)
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
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    device = resolve_device()
    print(
        f"device={device}; base_dim={base_features[pieces[0]].shape[1]}; "
        f"rich_dim={rich_features[pieces[0]].shape[1]}; beat_mlp_dim={BEAT_MLP_DIM}"
    )
    settings = [
        "baseline_cnn",
        "random_rich_mlp128",
        "rich_midibert_mlp128_only",
        "handcrafted_plus_rich_midibert_mlp128",
    ]
    fold_frames = []
    aggregates = []
    for setting in settings:
        folds_df, aggregate = run_setting(setting, cfg, pieces, labels, base_features, rich_features, folds, device)
        fold_frames.append(folds_df)
        aggregates.append(aggregate)
    pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "config_path": str(CONFIG_PATH),
                "rich_dir": str(RICH_DIR),
                "base_feature_dim": int(base_features[pieces[0]].shape[1]),
                "rich_feature_dim": int(rich_features[pieces[0]].shape[1]),
                "beat_mlp_dim": BEAT_MLP_DIM,
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
