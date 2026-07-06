#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from boundary_restart.config import load_config, resolve_path
from boundary_restart.table_io import load_table


def build_piece_list(config_path: str) -> list[str]:
    cfg = load_config(config_path)
    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = load_table(table_path)
    return sorted(df["piece_id"].dropna().unique().tolist())


def chunked_folds(items: list[str], n_splits: int) -> list[list[str]]:
    if n_splits <= 1:
        raise ValueError("n_splits must be >= 2")
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for idx, item in enumerate(items):
        folds[idx % n_splits].append(item)
    return [fold for fold in folds if fold]


def make_inner_folds(pieces: list[str], inner_mode: str, n_splits: int, max_inner_folds: int | None) -> list[list[str]]:
    if inner_mode == "leave_one":
        folds = [[piece] for piece in sorted(pieces)]
    elif inner_mode == "groupkfold":
        folds = chunked_folds(sorted(pieces), int(n_splits))
    else:
        raise ValueError(f"Unsupported inner_mode: {inner_mode}")
    if max_inner_folds is not None:
        folds = folds[: int(max_inner_folds)]
    return folds


def slugify_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text.replace("/", "_").replace(" ", "_").replace(".", "p")


def candidate_slug(candidate: dict) -> str:
    keys = [
        "model",
        "detector_target",
        "selection_metric",
        "min_precision",
        "loss_type",
        "label_engineering",
        "crf_state_count",
        "cumulative_merge_tolerance",
        "event_decoder",
        "event_tolerance",
        "rest_span_label_mode",
        "rest_span_tolerance_negative_weight",
    ]
    parts = []
    for key in keys:
        if key in candidate:
            parts.append(f"{key}-{slugify_value(candidate[key])}")
    return "__".join(parts) if parts else "candidate"


def candidate_primary_metric(candidate: dict, summary: dict) -> float:
    metric = str(candidate.get("selection_metric", "weighted_recall"))
    union_metrics = summary["union_metrics"]
    if metric == "union_recall":
        return float(union_metrics["union_recall"])
    if metric == "consensus_recall":
        return float(union_metrics["consensus_recall"])
    return float(union_metrics["weighted_recall"])


def build_base_candidate(args: argparse.Namespace) -> dict:
    candidate = {
        "model": str(args.model),
        "detector_target": str(args.detector_target),
        "selection_metric": str(args.selection_metric),
        "precision_metric": str(args.precision_metric),
        "min_precision": float(args.min_precision),
        "loss_type": str(args.loss_type),
        "rest_span_label_mode": str(args.rest_span_label_mode),
        "rest_span_tolerance_negative_weight": float(args.rest_span_tolerance_negative_weight),
        "min_train_frequency_target": float(args.min_train_frequency_target),
        "label_engineering": str(args.label_engineering),
        "label_decay_radius": int(args.label_decay_radius),
        "label_decay_rate": float(args.label_decay_rate),
        "center_margin": float(args.center_margin),
        "center_margin_weight": float(args.center_margin_weight),
        "phase_loss_weight": float(args.phase_loss_weight),
        "linear_max_span": int(args.linear_max_span),
        "crf_state_count": int(args.crf_state_count),
        "cumulative_merge_tolerance": int(args.cumulative_merge_tolerance),
        "cumulative_component_weights_json": str(args.cumulative_component_weights_json)
        if args.cumulative_component_weights_json is not None
        else None,
        "event_decoder": str(args.event_decoder),
        "event_tolerance": None if args.event_tolerance is None else int(args.event_tolerance),
        "device": str(args.device),
        "batch_size": int(args.batch_size) if args.batch_size is not None else None,
        "epochs": int(args.epochs) if args.epochs is not None else None,
        "early_stop_patience": int(args.early_stop_patience) if args.early_stop_patience is not None else None,
        "seed": int(args.seed),
    }
    if bool(args.skip_stage_grading):
        candidate["skip_stage_grading"] = True
    return candidate


def load_candidates(args: argparse.Namespace) -> list[dict]:
    base_candidate = build_base_candidate(args)
    if not args.candidate_json:
        return [base_candidate]

    payload = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate_json must contain a list of candidate dicts")
    candidates = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"candidate_json entry {idx} is not a dict")
        merged = dict(base_candidate)
        merged.update(item)
        candidates.append(merged)
    return candidates


def append_arg(command: list[str], flag: str, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        command.append(flag)
        command.extend(str(v) for v in value)
        return
    command.extend([flag, str(value)])


def run_protocol(
    python_exec: str,
    protocol_script: Path,
    config_path: str,
    candidate: dict,
    heldout_pieces: list[str],
    train_pieces: list[str],
    output_dir: Path,
    reuse_existing: bool,
) -> dict:
    summary_path = output_dir / "summary.json"
    if reuse_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    command = [python_exec, str(protocol_script), "--config", config_path]
    append_arg(command, "--heldout_piece", heldout_pieces)
    append_arg(command, "--train_pieces", train_pieces)
    append_arg(command, "--output_dir", str(output_dir))

    supported_keys = [
        "model",
        "detector_target",
        "selection_metric",
        "precision_metric",
        "min_precision",
        "loss_type",
        "rest_span_label_mode",
        "rest_span_tolerance_negative_weight",
        "min_train_frequency_target",
        "label_engineering",
        "label_decay_radius",
        "label_decay_rate",
        "center_margin",
        "center_margin_weight",
        "phase_loss_weight",
        "linear_max_span",
        "crf_state_count",
        "cumulative_merge_tolerance",
        "cumulative_component_weights_json",
        "event_decoder",
        "event_tolerance",
        "device",
        "batch_size",
        "epochs",
        "early_stop_patience",
        "seed",
        "skip_stage_grading",
    ]
    for key in supported_keys:
        if key not in candidate:
            continue
        append_arg(command, f"--{key}", candidate[key])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command), encoding="utf-8")
    subprocess.run(command, check=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def summarize_fold_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No rows to summarize")
    metrics = [
        "union_precision",
        "union_recall",
        "weighted_recall",
        "consensus_recall",
        "union_f1",
        "primary_metric",
    ]
    summary = {"fold_count": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        summary[f"mean_{metric}"] = mean
        summary[f"std_{metric}"] = math.sqrt(var)
    return summary


def rank_key(candidate_summary: dict) -> tuple:
    return (
        float(candidate_summary["mean_union_precision"] >= candidate_summary["min_precision"]),
        candidate_summary["mean_primary_metric"],
        candidate_summary["mean_union_precision"],
        candidate_summary["mean_consensus_recall"],
        candidate_summary["mean_union_f1"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested piece-level CV runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outer_heldout_piece", nargs="+", required=True)
    parser.add_argument("--inner_mode", choices=["leave_one", "groupkfold"], default="leave_one")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_inner_folds", type=int, default=None)
    parser.add_argument("--candidate_json")
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--protocol_script", default="train_piece_union_protocol.py")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--run_outer_fit", action="store_true")

    parser.add_argument("--model", default="tcn")
    parser.add_argument("--detector_target", default="level56_boundary")
    parser.add_argument("--selection_metric", default="union_recall")
    parser.add_argument("--precision_metric", default="union_precision")
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--loss_type", default="bce")
    parser.add_argument("--rest_span_label_mode", default="none")
    parser.add_argument("--rest_span_tolerance_negative_weight", type=float, default=1.0)
    parser.add_argument("--min_train_frequency_target", type=float, default=0.0)
    parser.add_argument("--label_engineering", choices=["none", "exponential_decay", "linear_ascend"], default="none")
    parser.add_argument("--label_decay_radius", type=int, default=2)
    parser.add_argument("--label_decay_rate", type=float, default=0.5)
    parser.add_argument("--center_margin", type=float, default=0.05)
    parser.add_argument("--center_margin_weight", type=float, default=0.0)
    parser.add_argument("--phase_loss_weight", type=float, default=0.0)
    parser.add_argument("--linear_max_span", type=int, default=64)
    parser.add_argument("--crf_state_count", type=int, default=64)
    parser.add_argument("--cumulative_merge_tolerance", type=int, default=0)
    parser.add_argument("--cumulative_component_weights_json", default=None)
    parser.add_argument("--event_decoder", choices=["peak", "crf"], default="peak")
    parser.add_argument("--event_tolerance", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_stage_grading", action="store_true")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    protocol_script = Path(args.protocol_script)
    if not protocol_script.is_absolute():
        protocol_script = Path(__file__).resolve().parent / protocol_script
    protocol_script = protocol_script.resolve()

    all_pieces = build_piece_list(config_path)
    outer_holdout = sorted(set(args.outer_heldout_piece))
    missing_outer = sorted(set(outer_holdout) - set(all_pieces))
    if missing_outer:
        raise ValueError(f"Unknown outer heldout pieces: {missing_outer}")

    inner_piece_pool = [piece for piece in all_pieces if piece not in outer_holdout]
    inner_folds = make_inner_folds(
        pieces=inner_piece_pool,
        inner_mode=str(args.inner_mode),
        n_splits=int(args.n_splits),
        max_inner_folds=args.max_inner_folds,
    )
    candidates = load_candidates(args)

    outer_slug = "__".join(outer_holdout)
    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = Path(__file__).resolve().parent / "reports" / "nested_piece_cv" / outer_slug
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "inner_folds.json").write_text(json.dumps(inner_folds, indent=2), encoding="utf-8")
    (out_root / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    fold_rows: list[dict] = []
    candidate_summaries: list[dict] = []

    for cand_idx, candidate in enumerate(candidates, start=1):
        cand_slug = candidate_slug(candidate)
        inner_rows = []
        for fold_idx, val_pieces in enumerate(inner_folds, start=1):
            train_pieces = [piece for piece in inner_piece_pool if piece not in set(val_pieces)]
            fold_dir = out_root / "inner_cv" / cand_slug / f"fold_{fold_idx:02d}"
            summary = run_protocol(
                python_exec=str(args.python_exec),
                protocol_script=protocol_script,
                config_path=config_path,
                candidate=candidate,
                heldout_pieces=val_pieces,
                train_pieces=train_pieces,
                output_dir=fold_dir,
                reuse_existing=bool(args.reuse_existing),
            )
            union_metrics = summary["union_metrics"]
            primary_metric = candidate_primary_metric(candidate, summary)
            row = {
                "candidate_index": cand_idx,
                "candidate_slug": cand_slug,
                "fold_index": fold_idx,
                "val_pieces": ",".join(val_pieces),
                "train_piece_count": len(train_pieces),
                "union_precision": float(union_metrics["union_precision"]),
                "union_recall": float(union_metrics["union_recall"]),
                "weighted_recall": float(union_metrics["weighted_recall"]),
                "consensus_recall": float(union_metrics["consensus_recall"]),
                "union_f1": float(union_metrics["union_f1"]),
                "threshold": float(union_metrics["threshold"]),
                "primary_metric": float(primary_metric),
            }
            inner_rows.append(row)
            fold_rows.append(row)

        candidate_summary = {
            "candidate_index": cand_idx,
            "candidate_slug": cand_slug,
            "candidate": candidate,
            "min_precision": float(candidate.get("min_precision", args.min_precision)),
            **summarize_fold_metrics(inner_rows),
        }
        candidate_summaries.append(candidate_summary)

    best_candidate = max(candidate_summaries, key=rank_key)

    fold_csv = out_root / "inner_fold_results.csv"
    with fold_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    candidate_csv = out_root / "candidate_summary.csv"
    candidate_csv_fields = [
        "candidate_index",
        "candidate_slug",
        "min_precision",
        "fold_count",
        "mean_union_precision",
        "std_union_precision",
        "mean_union_recall",
        "std_union_recall",
        "mean_weighted_recall",
        "std_weighted_recall",
        "mean_consensus_recall",
        "std_consensus_recall",
        "mean_union_f1",
        "std_union_f1",
        "mean_primary_metric",
        "std_primary_metric",
    ]
    with candidate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_csv_fields)
        writer.writeheader()
        for item in candidate_summaries:
            writer.writerow({key: item[key] for key in candidate_csv_fields})

    result = {
        "outer_heldout_pieces": outer_holdout,
        "inner_mode": str(args.inner_mode),
        "inner_fold_count": len(inner_folds),
        "inner_piece_pool_size": len(inner_piece_pool),
        "best_candidate": best_candidate,
    }

    if args.run_outer_fit:
        outer_dir = out_root / "outer_test" / best_candidate["candidate_slug"]
        outer_summary = run_protocol(
            python_exec=str(args.python_exec),
            protocol_script=protocol_script,
            config_path=config_path,
            candidate=best_candidate["candidate"],
            heldout_pieces=outer_holdout,
            train_pieces=inner_piece_pool,
            output_dir=outer_dir,
            reuse_existing=bool(args.reuse_existing),
        )
        result["outer_test_summary"] = outer_summary

    (out_root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote nested CV results to {out_root}")
    print(
        f"Best candidate: {best_candidate['candidate_slug']} | "
        f"mean_primary_metric={best_candidate['mean_primary_metric']:.4f} | "
        f"mean_union_precision={best_candidate['mean_union_precision']:.4f}"
    )


if __name__ == "__main__":
    main()
