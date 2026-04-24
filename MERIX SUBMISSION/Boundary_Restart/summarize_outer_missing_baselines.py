#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize newly added outer-baseline runs across seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--report_root", default=None)
    parser.add_argument("--nonseq_dir_template", default="paper_outer_missing_baselines_seed{seed}")
    parser.add_argument("--bilstm_dir_template", default="paper_outer_baselines_weighted_topdown_all_seed{seed}_bilstm")
    parser.add_argument("--grouper_levelwise_csv", default=None)
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def normalize_target(level_label: str) -> str:
    return "L5" if str(level_label) == "L5+6" else str(level_label)


def map_setting(model: str, target_design: str, feature_family: str) -> str:
    model = str(model)
    design_slug = "weighted" if str(target_design) == "weighted_topdown" else str(target_design)
    if model == "bilstm":
        return "bilstm_weighted_all"
    if model == "all_boundary":
        return "all_boundary_direct"
    if model == "periodic":
        return "periodic_k_direct"
    if model == "downbeat":
        return "downbeat_only_direct"
    if model == "logreg_window7":
        return f"logreg_window7_{design_slug}_{feature_family}"
    return f"{model}_{design_slug}_{feature_family}"


def load_seed_rows(report_dir: Path, seed: int) -> list[dict[str, object]]:
    summary_path = report_dir / "outer_summary_by_level.csv"
    if not summary_path.exists():
        return []
    df = pd.read_csv(summary_path)
    rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "seed": int(seed),
                "setting": map_setting(row["model"], row["target_design"], row["feature_family"]),
                "target": normalize_target(row["level_label"]),
                "union_precision": float(row["union_precision"]),
                "weighted_recall": float(row["weighted_recall"]),
                "union_recall": float(row["union_recall"]),
                "consensus_recall": float(row["consensus_recall"]),
                "period_k": None if pd.isna(row.get("period_k")) else int(row["period_k"]),
                "report_dir": str(report_dir),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    report_root = Path(args.report_root).resolve() if args.report_root else (script_dir / "reports").resolve()
    seeds = [int(seed) for seed in args.seeds]
    seed_slug = "".join(str(seed) for seed in seeds)
    output_prefix = args.output_prefix or f"paper_outer_missing_baseline_summary_seed{seed_slug}"

    raw_rows: list[dict[str, object]] = []
    source_dirs: dict[str, list[str]] = {"nonseq": [], "bilstm": []}

    for seed in seeds:
        nonseq_dir = report_root / args.nonseq_dir_template.format(seed=seed)
        bilstm_dir = report_root / args.bilstm_dir_template.format(seed=seed)
        if nonseq_dir.exists():
            raw_rows.extend(load_seed_rows(nonseq_dir, seed=seed))
            source_dirs["nonseq"].append(str(nonseq_dir))
        if bilstm_dir.exists():
            raw_rows.extend(load_seed_rows(bilstm_dir, seed=seed))
            source_dirs["bilstm"].append(str(bilstm_dir))

    grouper_status = "not_provided"
    if args.grouper_levelwise_csv:
        grouper_path = Path(args.grouper_levelwise_csv).resolve()
        if grouper_path.exists():
            grouper_df = pd.read_csv(grouper_path)
            expected = {"seed", "target", "union_precision", "weighted_recall", "union_recall", "consensus_recall"}
            missing = sorted(expected - set(grouper_df.columns))
            if missing:
                raise ValueError(f"Grouper CSV is missing columns: {missing}")
            for row in grouper_df.to_dict(orient="records"):
                raw_rows.append(
                    {
                        "seed": int(row["seed"]),
                        "setting": "grouper_weighted_all",
                        "target": normalize_target(row["target"]),
                        "union_precision": float(row["union_precision"]),
                        "weighted_recall": float(row["weighted_recall"]),
                        "union_recall": float(row["union_recall"]),
                        "consensus_recall": float(row["consensus_recall"]),
                        "period_k": None,
                        "report_dir": str(grouper_path),
                    }
                )
            grouper_status = "merged"
        else:
            grouper_status = f"missing:{grouper_path}"

    if not raw_rows:
        raise RuntimeError("No baseline outputs found to summarize")

    raw_df = pd.DataFrame(raw_rows).sort_values(["setting", "seed", "target"]).reset_index(drop=True)
    levelwise_df = (
        raw_df.groupby(["setting", "target"], sort=False)[
            ["union_precision", "weighted_recall", "union_recall", "consensus_recall"]
        ]
        .mean()
        .reset_index()
    )
    mean_df = (
        levelwise_df.groupby(["setting"], sort=False)[
            ["union_precision", "weighted_recall", "union_recall", "consensus_recall"]
        ]
        .mean()
        .reset_index()
    )

    raw_path = report_root / f"{output_prefix}_raw.csv"
    levelwise_path = report_root / f"{output_prefix}_levelwise.csv"
    mean_path = report_root / f"{output_prefix}_mean.csv"
    manifest_path = report_root / f"{output_prefix}_manifest.json"

    raw_df.to_csv(raw_path, index=False)
    levelwise_df.to_csv(levelwise_path, index=False)
    mean_df.to_csv(mean_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "report_root": str(report_root),
                "source_dirs": source_dirs,
                "grouper_status": grouper_status,
                "raw_csv": str(raw_path),
                "levelwise_csv": str(levelwise_path),
                "mean_csv": str(mean_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(levelwise_path))
    print(str(mean_path))


if __name__ == "__main__":
    main()
