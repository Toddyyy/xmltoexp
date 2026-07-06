from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_ROOT = ROOT / "MERIX SUBMISSION" / "Boundary_Restart"
MIREX_ROOT = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
COMPARE_DIR = MIREX_ROOT / "vienna_atepp_k331_tempo_compare"
MODEL_ROOT = BOUNDARY_ROOT / "reports" / "mazurkabl_l2plus_weighted_protocol_seed42"
OUT_DIR = MIREX_ROOT / "k331_mazurkabl_l2plus_direct_prediction"
VIENNA_SCORE = ROOT / "datasets" / "Vienna4x4" / "vienna4x22_rematched-master" / "musicxml" / "Mozart_K331_1st-mov.musicxml"
ATEPP_SCORE = (
    ROOT
    / "ATEPP-1.2"
    / "ATEPP-1.2"
    / "Wolfgang_Amadeus_Mozart"
    / "Piano_Sonata_No._11_in_A_Major,_K._331"
    / "1._Tema_(Andante_grazioso)_con_variazioni"
    / "musicxml_cleaned.musicxml"
)
CONFIG_PATH = BOUNDARY_ROOT / "configs" / "mazurkabl_l2plus_weighted_auto_meter.yaml"

STR_VEC = [3, 2, 2, 2, 2, 2]
LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.0}
EVENT_MIN = 0.05
MIN_DISTANCE = 6
TOLERANCE = 1

sys.path.insert(0, str(BOUNDARY_ROOT))
sys.path.insert(0, str(MIREX_ROOT))

from boundary_restart.config import load_config  # noqa: E402
from boundary_restart.features import PeakConfig  # noqa: E402
from boundary_restart.metrics import decode_events, greedy_match_pairs  # noqa: E402
from boundary_restart.models import build_sequence_model  # noqa: E402
from compare_xls_mazurka_boundary_labels import group_analysis_hierarchy  # noqa: E402
from predict_new_scores_merge56_seed44 import build_feature_frame_for_score  # noqa: E402


def weighted_l2plus_target_from_tempo(tempo: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    _, level_sets = group_analysis_hierarchy(tempo, STR_VEC, enforce_nested=True)
    n = int(len(tempo))
    weighted = np.zeros(n, dtype=np.float32)
    components: dict[int, np.ndarray] = {}
    for level, weight in LEVEL_WEIGHTS.items():
        mask = np.zeros(n, dtype=np.float32)
        idx = np.asarray(level_sets.get(level, []), dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < n)]
        mask[idx] = 1.0
        components[level] = mask
        weighted = np.maximum(weighted, mask * float(weight))
    return weighted, components


def match_metrics(pred_events: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    true_events = np.flatnonzero(np.asarray(target) >= EVENT_MIN).astype(np.int32)
    pairs = greedy_match_pairs(pred_events, true_events, tolerance=TOLERANCE)
    matched_true = [true_idx for _, true_idx, _ in pairs]
    matched_weight = float(np.sum(target[true_events[matched_true]])) if matched_true else 0.0
    total_weight = float(np.sum(target[true_events])) if true_events.size else 0.0
    matches = len(pairs)
    precision = matches / len(pred_events) if len(pred_events) else 0.0
    recall = matches / len(true_events) if len(true_events) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_recall = matched_weight / total_weight if total_weight > 0 else 0.0
    return {
        "pred_events": int(len(pred_events)),
        "true_events": int(len(true_events)),
        "matches": int(matches),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_weight": matched_weight,
        "total_weight": total_weight,
        "weighted_recall": float(weighted_recall),
    }


def load_score_feature_frame() -> pd.DataFrame:
    cfg = load_config(CONFIG_PATH)
    data_cfg = cfg.get("data", {})
    frame = build_feature_frame_for_score(
        score_path=ATEPP_SCORE,
        piece_id="Mozart_K331_score_aligned_theme",
        peak_cfg=PeakConfig(
            distance=int(data_cfg.get("peak_distance", 6)),
            height=float(data_cfg.get("peak_height", 0.15)),
            prominence=float(data_cfg.get("peak_prominence", 0.05)),
        ),
        beat_unit=0.5,
        measure_cycle=int(data_cfg.get("measure_cycle", 3)),
        symmetry_window=int(data_cfg.get("symmetry_window", 4)),
        deviation_window=int(data_cfg.get("deviation_window", 8)),
        expand_repeats=False,
    ).sort_values("beat_idx")
    frame = frame[frame["beat_idx"] < 216].copy()
    frame["num_beats"] = 216
    return frame


@torch.no_grad()
def predict_one_checkpoint(checkpoint_path: Path, frame: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_cols = list(checkpoint["feature_columns"])
    local = frame.copy()
    for col in feature_cols:
        if col not in local.columns:
            local[col] = 0.0
    x = local[feature_cols].to_numpy(dtype=np.float32)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((x - mean) / std).astype(np.float32)
    model = build_sequence_model(
        str(checkpoint["model_type"]),
        input_dim=len(feature_cols),
        cfg=cfg,
        output_dim=int(checkpoint.get("output_dim", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logits = model(torch.from_numpy(x[None, :, :]))
    score = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
    return score, float(checkpoint.get("best_threshold", 0.05))


def load_tempo_means() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(COMPARE_DIR / "k331_score_aligned_mean_tempo.csv")
    return (
        df["vienna_mean_bpm"].to_numpy(dtype=np.float32),
        df["atepp_score_aligned_mean_bpm"].to_numpy(dtype=np.float32),
    )


def local_minima_for_plot(tempo: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(-np.asarray(tempo, dtype=np.float32), distance=MIN_DISTANCE)
    return peaks.astype(np.int32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIG_PATH)
    frame = load_score_feature_frame()
    if int(frame["beat_idx"].max()) + 1 != 216:
        raise RuntimeError(f"Expected 216 K331 beats, got max beat_idx={int(frame['beat_idx'].max())}")

    fold_scores = []
    thresholds = []
    fold_rows = []
    for checkpoint_path in sorted(MODEL_ROOT.glob("fold*/detector_best.pt")):
        score, threshold = predict_one_checkpoint(checkpoint_path, frame, cfg)
        fold_scores.append(score)
        thresholds.append(threshold)
        fold_rows.append(
            {
                "fold": checkpoint_path.parent.name,
                "checkpoint_path": str(checkpoint_path),
                "threshold": threshold,
                "score_mean": float(np.mean(score)),
                "score_max": float(np.max(score)),
            }
        )
    if not fold_scores:
        raise FileNotFoundError(f"No checkpoints under {MODEL_ROOT}")
    score_matrix = np.vstack(fold_scores)
    mean_score = np.mean(score_matrix, axis=0).astype(np.float32)
    threshold = float(np.mean(thresholds))
    pred_events = decode_events(
        mean_score,
        threshold=threshold,
        min_distance=MIN_DISTANCE,
        prominence=0.0,
        event_decoder="peak",
    )

    vienna_tempo, atepp_tempo = load_tempo_means()
    vienna_target, vienna_components = weighted_l2plus_target_from_tempo(vienna_tempo)
    atepp_target, atepp_components = weighted_l2plus_target_from_tempo(atepp_tempo)

    rows = []
    for name, tempo, target in [
        ("vienna", vienna_tempo, vienna_target),
        ("atepp", atepp_tempo, atepp_target),
    ]:
        metric = match_metrics(pred_events, target)
        rows.append({"target_source": name, "threshold": threshold, **metric})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "k331_direct_prediction_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "k331_fold_prediction_scores.csv", index=False)

    curve_df = pd.DataFrame(
        {
            "beat_idx": np.arange(len(mean_score), dtype=np.int32),
            "model_score_mean5fold": mean_score,
            "model_pred_event": np.isin(np.arange(len(mean_score)), pred_events).astype(np.int8),
            "vienna_mean_bpm": vienna_tempo,
            "atepp_mean_bpm": atepp_tempo,
            "vienna_weighted_target": vienna_target,
            "atepp_weighted_target": atepp_target,
            "vienna_local_minimum": np.isin(np.arange(len(mean_score)), local_minima_for_plot(vienna_tempo)).astype(np.int8),
            "atepp_local_minimum": np.isin(np.arange(len(mean_score)), local_minima_for_plot(atepp_tempo)).astype(np.int8),
        }
    )
    for level, arr in vienna_components.items():
        curve_df[f"vienna_L{level}"] = arr
    for level, arr in atepp_components.items():
        curve_df[f"atepp_L{level}"] = arr
    curve_df.to_csv(OUT_DIR / "k331_direct_prediction_curves.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.15, 0.9, 0.95]})
    x = np.arange(len(mean_score))
    axes[0].plot(x, vienna_tempo, color="#1f4e8c", linewidth=2.0, label="Vienna mean tempo")
    axes[0].plot(x, atepp_tempo, color="#c45a00", linewidth=2.0, label="ATEPP score-aligned mean tempo")
    axes[0].scatter(np.flatnonzero(vienna_target >= EVENT_MIN), vienna_tempo[vienna_target >= EVENT_MIN], color="#1f4e8c", s=16, marker="v", label="Vienna L2+ target")
    axes[0].scatter(np.flatnonzero(atepp_target >= EVENT_MIN), atepp_tempo[atepp_target >= EVENT_MIN], color="#c45a00", s=16, marker="x", label="ATEPP L2+ target")
    axes[0].set_ylabel("Tempo BPM")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(x, mean_score, color="#222222", linewidth=2.0, label="MazurkaBL model score mean over 5 folds")
    axes[1].axhline(threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"decode threshold={threshold:.3f}")
    axes[1].scatter(pred_events, mean_score[pred_events], color="#d62728", s=26, label=f"predicted events n={len(pred_events)}")
    axes[1].set_ylabel("Model score")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)

    y_v = np.ones_like(np.flatnonzero(vienna_target >= EVENT_MIN), dtype=float)
    y_a = np.ones_like(np.flatnonzero(atepp_target >= EVENT_MIN), dtype=float) * 0.55
    y_p = np.ones_like(pred_events, dtype=float) * 0.1
    axes[2].scatter(np.flatnonzero(vienna_target >= EVENT_MIN), y_v, color="#1f4e8c", marker="|", s=160, label="Vienna target")
    axes[2].scatter(np.flatnonzero(atepp_target >= EVENT_MIN), y_a, color="#c45a00", marker="|", s=160, label="ATEPP target")
    axes[2].scatter(pred_events, y_p, color="#d62728", marker="|", s=180, label="Prediction")
    axes[2].set_yticks([0.1, 0.55, 1.0])
    axes[2].set_yticklabels(["Pred", "ATEPP", "Vienna"])
    axes[2].set_xlabel("Aligned K331 beat index")
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].legend(frameon=False, ncol=3, loc="upper right")

    fig.suptitle("MazurkaBL L2+ weighted model directly predicting Mozart K331 score fragment", y=0.995)
    fig.tight_layout()
    fig_path = OUT_DIR / "k331_direct_prediction_vs_vienna_atepp_targets.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(fig_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
