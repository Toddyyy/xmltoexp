#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.features import PeakConfig
from boundary_restart.lbdm import compute_lbdm_beat_salience_from_npz
from boundary_restart.metrics import evaluate_union_frequency_event_sets, greedy_match_pairs
from boundary_restart.table_io import feature_columns, load_table
from train_piece_union_protocol import (
    apply_piece_protocol_split,
    apply_rest_span_training_labels,
    build_piece_union_frame,
    build_predicted_event_frame,
    detector_sequence_maps,
    search_union_frequency_threshold,
    union_metrics_to_dict,
)


DEFAULT_CONFIG = "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
DEFAULT_OUTER = ("M06-1", "M06-2", "M06-3")
TARGET_SPECS = {
    "L1+": {"target": "level1plus_boundary", "min_precision": 0.85},
    "L2+": {"target": "level2plus_boundary", "min_precision": 0.85},
    "L3+": {"target": "level3plus_boundary", "min_precision": 0.85},
    "L4+": {"target": "level4plus_boundary", "min_precision": 0.85},
    "L5+6": {"target": "level56_boundary", "min_precision": 0.80},
}
WEIGHTED_COMPONENT_WEIGHTS = {
    "level56": 1.0,
    "level4": 0.64,
    "level3": 0.46,
    "level2": 0.28,
    "level1": 0.16,
}
PERIODIC_TARGET_K = {
    "L1+": 3,
    "L2+": 6,
    "L3+": 12,
    "L4+": 24,
    "L5+6": 48,
}
DIRECT_RULE_MODELS = {"all_boundary", "periodic", "downbeat"}
SKLEARN_MODELS = {"logreg", "logreg_window7", "mlp", "lbdm"}
MODEL_CHOICES = sorted(DIRECT_RULE_MODELS | SKLEARN_MODELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run score-only clean-outer baselines on the fixed Mazurka outer split."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["logreg", "mlp"])
    parser.add_argument("--targets", nargs="+", choices=list(TARGET_SPECS.keys()), default=list(TARGET_SPECS.keys()))
    parser.add_argument("--outer_heldout_piece", nargs="+", default=list(DEFAULT_OUTER))
    parser.add_argument("--target_design", choices=["weighted_topdown", "simple_union"], default="weighted_topdown")
    parser.add_argument("--feature_family", choices=["all", "xml_only", "note_only"], default="all")
    parser.add_argument("--min_train_frequency_target", type=float, default=0.05)
    parser.add_argument("--selection_metric", default="weighted_recall")
    parser.add_argument("--precision_metric", default="union_precision")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max_inner_folds", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def target_design_params(target_design: str) -> tuple[int, dict[str, float] | None]:
    if target_design == "weighted_topdown":
        return 2, dict(WEIGHTED_COMPONENT_WEIGHTS)
    if target_design == "simple_union":
        return 0, None
    raise ValueError(f"Unsupported target_design: {target_design}")


def select_feature_family(cols: list[str], family: str) -> list[str]:
    cols = [col for col in cols if col not in {"protocol_split", "lbdm_score"}]
    if family == "all":
        return list(cols)
    if family == "xml_only":
        return [col for col in cols if col.startswith("xml_")]
    if family == "note_only":
        return [col for col in cols if not col.startswith("xml_")]
    raise ValueError(f"Unsupported feature family: {family}")


def oversample_binary(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0 or pos_idx.size >= neg_idx.size:
        return x, y
    rng = np.random.default_rng(seed)
    extra = rng.choice(pos_idx, size=int(neg_idx.size - pos_idx.size), replace=True)
    idx = np.concatenate([np.arange(y.shape[0]), extra], axis=0)
    rng.shuffle(idx)
    return x[idx], y[idx]


def context_window_radius(model_name: str) -> int:
    return 3 if model_name == "logreg_window7" else 0


def base_model_name(model_name: str) -> str:
    if model_name == "logreg_window7":
        return "logreg"
    return model_name


def build_model(model_name: str, seed: int):
    model_name = base_model_name(model_name)
    if model_name == "logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def attach_lbdm_scores(raw_df: pd.DataFrame, beat_unit_fallback: float) -> pd.DataFrame:
    frame = raw_df.copy()
    piece_rows: list[pd.DataFrame] = []
    for piece_id, group in frame.groupby("piece_id", sort=False):
        source_path = Path(str(group["source_path"].dropna().iloc[0]))
        beat_scores = compute_lbdm_beat_salience_from_npz(
            npz_path=source_path,
            beat_unit_fallback=beat_unit_fallback,
        )
        piece_frame = pd.DataFrame(
            {
                "piece_id": piece_id,
                "beat_idx": np.arange(beat_scores.shape[0], dtype=np.int32),
                "lbdm_score": beat_scores.astype(np.float32),
            }
        )
        piece_rows.append(piece_frame)
    if not piece_rows:
        frame["lbdm_score"] = 0.0
        return frame
    lbdm_df = pd.concat(piece_rows, ignore_index=True)
    frame = frame.merge(lbdm_df, on=["piece_id", "beat_idx"], how="left", validate="many_to_one")
    frame["lbdm_score"] = frame["lbdm_score"].fillna(0.0).astype(np.float32)
    return frame


def augment_piece_features_with_context(
    piece_df: pd.DataFrame,
    feature_cols: list[str],
    radius: int,
) -> tuple[pd.DataFrame, list[str]]:
    if radius <= 0:
        return piece_df, list(feature_cols)
    frame = piece_df.sort_values(["piece_id", "beat_idx"]).copy()
    context_cols: list[str] = []
    grouped = frame.groupby("piece_id", sort=False)
    context_frames: list[pd.DataFrame] = []
    for offset in range(-radius, radius + 1):
        if offset < 0:
            suffix = f"m{abs(offset)}"
        elif offset > 0:
            suffix = f"p{offset}"
        else:
            suffix = "c"
        shifted = grouped[feature_cols].shift(-offset).fillna(0.0).astype(np.float32)
        renamed_cols = [f"{col}__{suffix}" for col in feature_cols]
        shifted.columns = renamed_cols
        context_frames.append(shifted)
        context_cols.extend(renamed_cols)
    frame = pd.concat([frame, *context_frames], axis=1)
    return frame.reset_index(drop=True), context_cols


def fit_and_score(
    model_name: str,
    seed: int,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
):
    model_name = base_model_name(model_name)
    if model_name == "lbdm":
        if "lbdm_score" not in eval_df.columns:
            raise KeyError("eval_df is missing lbdm_score")
        scores = eval_df["lbdm_score"].to_numpy(dtype=np.float32)
        return None, None, scores
    scaler = StandardScaler()
    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = (train_df["train_frequency_target"].to_numpy(dtype=np.float32) > 0.0).astype(np.int32)
    x_eval = eval_df[feature_cols].to_numpy(dtype=np.float32)
    x_train = scaler.fit_transform(x_train)
    x_eval = scaler.transform(x_eval)
    if model_name == "mlp":
        x_train, y_train = oversample_binary(x_train, y_train, seed=seed)
    model = build_model(model_name, seed=seed)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)[:, 1].astype(np.float32)
    return model, scaler, scores


def predict_frame(eval_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    pred_df = eval_df[
        [
            "piece_id",
            "piece_sample_id",
            "beat_idx",
            "performer_count",
            "union_target",
            "frequency_target",
        ]
    ].copy()
    pred_df = pred_df.rename(columns={"piece_sample_id": "sample_id"})
    pred_df["detector_score"] = scores.astype(np.float32)
    return pred_df


def row_from_saved_summary(summary: dict) -> dict[str, object]:
    return {
        "model": str(summary["model"]),
        "target_design": str(summary["target_design"]),
        "feature_family": str(summary["feature_family"]),
        "level_label": str(summary["level_label"]),
        "target_mode": str(summary["target_mode"]),
        "fixed_threshold": float(summary["fixed_threshold"]),
        "context_window_radius": int(summary.get("context_window_radius", 0)),
        "period_k": summary.get("period_k"),
        **dict(summary["union_metrics"]),
    }


def inner_threshold_search(
    piece_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    seed: int,
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    prominence: float,
    selection_metric: str,
    precision_metric: str,
    min_precision: float,
) -> tuple[float, list[dict[str, object]]]:
    dev_pieces = sorted(piece_df["piece_id"].unique().tolist())
    if args_global.max_inner_folds is not None:
        dev_pieces = dev_pieces[: int(args_global.max_inner_folds)]
    rows: list[dict[str, object]] = []
    chosen_thresholds: list[float] = []
    for fold_idx, val_piece in enumerate(dev_pieces, start=1):
        train_pieces = [piece for piece in dev_pieces if piece != val_piece]
        fold_df = apply_piece_protocol_split(piece_df, heldout_pieces=[val_piece], train_pieces=train_pieces)
        fold_df = apply_rest_span_training_labels(
            fold_df,
            min_train_frequency_target=float(args_global.min_train_frequency_target),
            mode="none",
            min_len=2,
            source_col="xml_rest_duration_norm",
            source_threshold=1e-8,
            tolerance_negative_weight=1.0,
        )
        train_df = fold_df[fold_df["protocol_split"] == "train"].copy()
        val_df = fold_df[fold_df["protocol_split"] == "val"].copy()
        model, scaler, scores = fit_and_score(
            model_name=model_name,
            seed=seed + fold_idx,
            feature_cols=feature_cols,
            train_df=train_df,
            eval_df=val_df,
        )
        pred_df = predict_frame(val_df, scores)
        sequence_scores, sequence_union, sequence_frequency = detector_sequence_maps(pred_df)
        metrics = search_union_frequency_threshold(
            sequence_scores=sequence_scores,
            sequence_union_labels=sequence_union,
            sequence_frequency_targets=sequence_frequency,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=min_precision,
            consensus_threshold=0.5,
            prominence=prominence,
            primary_metric=selection_metric,
            precision_metric=precision_metric,
            min_union_precision=min_precision if precision_metric == "union_precision" else 0.0,
            min_frequency_weighted_precision=min_precision if precision_metric == "frequency_weighted_precision" else 0.0,
            min_consensus_precision=min_precision if precision_metric == "consensus_precision" else 0.0,
        )
        chosen_thresholds.append(float(metrics.threshold))
        rows.append(
            {
                "fold_idx": fold_idx,
                "val_piece": val_piece,
                "threshold": float(metrics.threshold),
                "union_precision": float(metrics.union_precision),
                "union_recall": float(metrics.union_recall),
                "weighted_recall": float(metrics.weighted_recall),
                "consensus_recall": float(metrics.consensus_recall),
                "union_f1": float(metrics.union_f1),
                "pred_events": int(metrics.pred_events),
                "true_union_events": int(metrics.true_union_events),
            }
        )
        del model, scaler
    fixed_threshold = float(np.mean(chosen_thresholds)) if chosen_thresholds else float(thresholds[0])
    return fixed_threshold, rows


def periodic_k_for_level(level_label: str) -> int:
    try:
        return int(PERIODIC_TARGET_K[level_label])
    except KeyError as exc:
        raise KeyError(f"Missing periodic-k mapping for {level_label}") from exc


def direct_rule_event_sequences(
    eval_df: pd.DataFrame,
    model_name: str,
    level_label: str,
) -> tuple[dict[str, np.ndarray], dict[str, int | None]]:
    sequences: dict[str, np.ndarray] = {}
    meta: dict[str, int | None] = {"period_k": None}
    ordered = eval_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        group = group.reset_index(drop=True)
        length = len(group)
        if model_name == "all_boundary":
            pred_events = np.arange(length, dtype=np.int32)
        elif model_name == "periodic":
            period_k = periodic_k_for_level(level_label)
            meta["period_k"] = int(period_k)
            pred_events = np.arange(0, length, int(period_k), dtype=np.int32)
        elif model_name == "downbeat":
            if "xml_downbeat_actual" in group.columns:
                pred_events = np.flatnonzero(group["xml_downbeat_actual"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
            elif "beat_pos_in_measure" in group.columns:
                pred_events = np.flatnonzero(group["beat_pos_in_measure"].to_numpy(dtype=np.float32) <= 1e-6).astype(
                    np.int32
                )
            else:
                pred_events = np.arange(0, length, 3, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported direct rule model: {model_name}")
        sequences[str(sample_id)] = pred_events
    return sequences, meta


def direct_rule_prediction_frame(
    eval_df: pd.DataFrame,
    sequence_pred_events: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    ordered = eval_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        group = group.reset_index(drop=True)
        scores = np.zeros(len(group), dtype=np.float32)
        pred_events = np.asarray(sequence_pred_events[str(sample_id)], dtype=np.int32)
        pred_events = pred_events[(pred_events >= 0) & (pred_events < len(group))]
        scores[pred_events] = 1.0
        pred_df = group[
            [
                "piece_id",
                "piece_sample_id",
                "beat_idx",
                "performer_count",
                "union_target",
                "frequency_target",
            ]
        ].copy()
        pred_df = pred_df.rename(columns={"piece_sample_id": "sample_id"})
        pred_df["detector_score"] = scores
        rows.append(pred_df)
    if not rows:
        return pd.DataFrame(
            columns=[
                "piece_id",
                "sample_id",
                "beat_idx",
                "performer_count",
                "union_target",
                "frequency_target",
                "detector_score",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_direct_predicted_event_frame(
    pred_df: pd.DataFrame,
    sequence_pred_events: dict[str, np.ndarray],
    tolerance: int,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    ordered = pred_df.sort_values(["sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        group = group.reset_index(drop=True)
        pred_events = np.asarray(sequence_pred_events[str(sample_id)], dtype=np.int32)
        pred_events = pred_events[(pred_events >= 0) & (pred_events < len(group))]
        pred_events = np.asarray(sorted(set(pred_events.tolist())), dtype=np.int32)
        true_union_events = np.flatnonzero(group["union_target"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        match_pairs = greedy_match_pairs(pred_events, true_union_events, tolerance=int(tolerance))
        match_map = {pred_idx: (true_idx, offset) for pred_idx, true_idx, offset in match_pairs}

        for event_rank, pred_pos in enumerate(pred_events.tolist(), start=1):
            row = group.iloc[int(pred_pos)]
            true_match = match_map.get(event_rank - 1)
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "piece_id": str(row["piece_id"]),
                    "event_rank": int(event_rank),
                    "beat_idx": int(row["beat_idx"]),
                    "detector_score": float(row["detector_score"]),
                    "threshold": float(threshold),
                    "union_target_at_beat": float(row["union_target"]),
                    "frequency_target_at_beat": float(row["frequency_target"]),
                    "performer_count": int(row["performer_count"]),
                    "matched_union": bool(true_match is not None),
                    "match_offset": int(true_match[1]) if true_match is not None else None,
                    "matched_true_beat_idx": int(true_union_events[true_match[0]]) if true_match is not None else None,
                }
            )
    return pd.DataFrame(rows)


def run_outer_target(
    *,
    cfg: dict,
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    outer_holdout: list[str],
    model_name: str,
    level_label: str,
    target_mode: str,
    min_precision: float,
    target_design: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    target_out = output_dir / target_mode
    summary_path = target_out / "summary.json"
    if args_global.reuse_existing and summary_path.exists():
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return row_from_saved_summary(saved_summary)

    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    merge_tolerance, component_weights = target_design_params(target_design)
    effective_feature_cols = ["lbdm_score"] if model_name == "lbdm" else list(feature_cols)
    piece_df = build_piece_union_frame(
        raw_df,
        feature_cols=effective_feature_cols,
        target_mode=target_mode,
        peak_cfg=peak_cfg,
        beat_unit_fallback=float(data_cfg.get("beat_unit_fallback", 1.0)),
        cumulative_merge_tolerance=merge_tolerance,
        cumulative_component_weights=component_weights,
    )
    piece_df = piece_df.sort_values(["piece_id", "beat_idx"]).reset_index(drop=True)

    ctx_radius = context_window_radius(model_name)
    if ctx_radius > 0:
        piece_df, effective_feature_cols = augment_piece_features_with_context(piece_df, effective_feature_cols, ctx_radius)

    dev_piece_df = piece_df[~piece_df["piece_id"].isin(outer_holdout)].copy()
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))

    direct_rule = model_name in DIRECT_RULE_MODELS
    direct_rule_meta: dict[str, int | None] = {"period_k": None}
    if direct_rule:
        fixed_threshold = 0.5
        inner_rows: list[dict[str, object]] = []
    else:
        fixed_threshold, inner_rows = inner_threshold_search(
            piece_df=dev_piece_df,
            feature_cols=effective_feature_cols,
            model_name=model_name,
            seed=seed,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
            selection_metric=str(args_global.selection_metric),
            precision_metric=str(args_global.precision_metric),
            min_precision=float(min_precision),
        )

    outer_df = apply_piece_protocol_split(piece_df, heldout_pieces=outer_holdout)
    outer_df = apply_rest_span_training_labels(
        outer_df,
        min_train_frequency_target=float(args_global.min_train_frequency_target),
        mode="none",
        min_len=2,
        source_col="xml_rest_duration_norm",
        source_threshold=1e-8,
        tolerance_negative_weight=1.0,
    )
    train_df = outer_df[outer_df["protocol_split"] == "train"].copy()
    val_df = outer_df[outer_df["protocol_split"] == "val"].copy()

    if direct_rule:
        sequence_pred_events, direct_rule_meta = direct_rule_event_sequences(val_df, model_name=model_name, level_label=level_label)
        pred_df = direct_rule_prediction_frame(val_df, sequence_pred_events)
        _, sequence_union, sequence_frequency = detector_sequence_maps(pred_df)
        outer_metrics = evaluate_union_frequency_event_sets(
            sequence_pred_events=sequence_pred_events,
            sequence_union_labels=sequence_union,
            sequence_frequency_targets=sequence_frequency,
            tolerance=tolerance,
            threshold=float(fixed_threshold),
            consensus_threshold=0.5,
        )
        predicted_events = build_direct_predicted_event_frame(
            pred_df=pred_df,
            sequence_pred_events=sequence_pred_events,
            tolerance=tolerance,
            threshold=float(fixed_threshold),
        )
        model = None
        scaler = None
    else:
        model, scaler, outer_scores = fit_and_score(
            model_name=model_name,
            seed=seed,
            feature_cols=effective_feature_cols,
            train_df=train_df,
            eval_df=val_df,
        )
        pred_df = predict_frame(val_df, outer_scores)
        sequence_scores, sequence_union, sequence_frequency = detector_sequence_maps(pred_df)
        outer_metrics = search_union_frequency_threshold(
            sequence_scores=sequence_scores,
            sequence_union_labels=sequence_union,
            sequence_frequency_targets=sequence_frequency,
            thresholds=np.asarray([fixed_threshold], dtype=np.float32),
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=float(min_precision),
            consensus_threshold=0.5,
            prominence=prominence,
            primary_metric=str(args_global.selection_metric),
            precision_metric=str(args_global.precision_metric),
            min_union_precision=min_precision if args_global.precision_metric == "union_precision" else 0.0,
            min_frequency_weighted_precision=min_precision
            if args_global.precision_metric == "frequency_weighted_precision"
            else 0.0,
            min_consensus_precision=min_precision if args_global.precision_metric == "consensus_precision" else 0.0,
        )
        predicted_events = build_predicted_event_frame(
            pred_df=pred_df,
            threshold=fixed_threshold,
            min_distance=min_distance,
            prominence=prominence,
            tolerance=tolerance,
        )

    target_out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(target_out / "outer_predictions.csv.gz", index=False, compression="gzip")
    predicted_events.to_csv(target_out / "predicted_events.csv.gz", index=False, compression="gzip")
    pd.DataFrame(inner_rows).to_csv(target_out / "inner_fold_results.csv", index=False)
    with (target_out / "model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model_name": model_name,
                "model": model,
                "scaler": scaler,
                "feature_columns": effective_feature_cols,
                "target_mode": target_mode,
                "target_design": target_design,
                "fixed_threshold": float(fixed_threshold),
                "outer_holdout": list(outer_holdout),
                "direct_rule": bool(direct_rule),
                "context_window_radius": int(ctx_radius),
                **direct_rule_meta,
            },
            handle,
        )

    summary = {
        "model": model_name,
        "target_mode": target_mode,
        "level_label": level_label,
        "target_design": target_design,
        "feature_family": args_global.feature_family,
        "outer_heldout_pieces": list(outer_holdout),
        "seed": int(seed),
        "min_train_frequency_target": float(args_global.min_train_frequency_target),
        "selection_metric": str(args_global.selection_metric),
        "precision_metric": str(args_global.precision_metric),
        "min_precision": float(min_precision),
        "fixed_threshold": float(fixed_threshold),
        "feature_columns": effective_feature_cols,
        "train_piece_count": int(train_df["piece_id"].nunique()),
        "outer_piece_count": int(val_df["piece_id"].nunique()),
        "inner_fold_count": int(len(inner_rows)),
        "direct_rule": bool(direct_rule),
        "context_window_radius": int(ctx_radius),
        **direct_rule_meta,
        "union_metrics": union_metrics_to_dict(outer_metrics),
    }
    (target_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "model": model_name,
        "target_design": target_design,
        "feature_family": args_global.feature_family,
        "level_label": level_label,
        "target_mode": target_mode,
        "fixed_threshold": float(fixed_threshold),
        "context_window_radius": int(ctx_radius),
        **direct_rule_meta,
        **union_metrics_to_dict(outer_metrics),
    }


def main() -> None:
    global args_global
    args_global = parse_args()

    cfg_path = Path(args_global.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path(__file__).resolve().parent / cfg_path).resolve()
    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})
    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    raw_df = load_table(table_path)
    raw_df = apply_piece_protocol_split(raw_df, heldout_pieces=[])
    if "lbdm" in set(args_global.models):
        raw_df = attach_lbdm_scores(
            raw_df,
            beat_unit_fallback=float(data_cfg.get("beat_unit_fallback", 1.0)),
        )
    all_feature_cols = feature_columns(raw_df)
    selected_cols = select_feature_family(all_feature_cols, args_global.feature_family)

    output_dir = Path(
        args_global.output_dir
        or (
            Path(__file__).resolve().parent
            / "reports"
            / f"paper_outer_baselines_{args_global.target_design}_{args_global.feature_family}_seed{args_global.seed}"
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outer_holdout = sorted(set(args_global.outer_heldout_piece))
    summary_rows: list[dict[str, object]] = []
    for model_name in args_global.models:
        model_out = output_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        for level_label in args_global.targets:
            spec = TARGET_SPECS[level_label]
            row = run_outer_target(
                cfg=cfg,
                raw_df=raw_df,
                feature_cols=selected_cols,
                outer_holdout=outer_holdout,
                model_name=model_name,
                level_label=level_label,
                target_mode=str(spec["target"]),
                min_precision=float(spec["min_precision"]),
                target_design=str(args_global.target_design),
                seed=int(args_global.seed),
                output_dir=model_out,
            )
            summary_rows.append(row)
            period_suffix = ""
            if row.get("period_k") is not None:
                period_suffix = f" | k={int(row['period_k'])}"
            print(
                f"{model_name} {level_label} | p={row['union_precision']:.4f} | "
                f"wr={row['weighted_recall']:.4f} | cr={row['consensus_recall']:.4f} | "
                f"thr={row['fixed_threshold']:.3f}{period_suffix}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "outer_summary_by_level.csv", index=False)
    mean_df = (
        summary_df.groupby(["model", "target_design", "feature_family"], sort=False)[
            ["union_precision", "weighted_recall", "union_recall", "consensus_recall", "union_f1"]
        ]
        .mean()
        .reset_index()
    )
    mean_df.to_csv(output_dir / "outer_summary_mean.csv", index=False)
    metadata = {
        "config_path": str(cfg_path),
        "table_path": str(table_path),
        "outer_heldout_pieces": outer_holdout,
        "models": list(args_global.models),
        "target_design": str(args_global.target_design),
        "feature_family": str(args_global.feature_family),
        "seed": int(args_global.seed),
        "feature_columns": selected_cols,
        "periodic_target_k": dict(PERIODIC_TARGET_K),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    args_global = None
    main()
