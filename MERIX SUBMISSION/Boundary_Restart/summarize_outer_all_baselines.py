#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TCN_TARGET_DIRS = {
    "L1+": "level1plus_boundary",
    "L2+": "level2plus_boundary",
    "L3+": "level3plus_boundary",
    "L4+": "level4plus_boundary",
    "L5": "level56_boundary",
}

SETTING_ORDER = {
    "tcn_weighted_all": 0,
    "logreg_weighted_all": 1,
    "logreg_simple_union_all": 2,
    "logreg_weighted_note_only": 3,
    "logreg_weighted_xml_only": 4,
    "lbdm_weighted_all": 5,
    "all_boundary_direct": 6,
    "periodic_k_direct": 7,
    "downbeat_only_direct": 8,
    "logreg_window7_weighted_all": 9,
    "bilstm_weighted_all": 10,
    "grouper_weighted_all": 11,
}

TARGET_ORDER = {"L1+": 0, "L2+": 1, "L3+": 2, "L4+": 3, "L5": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all outer Mazurka baselines across seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--report_root", default=None)
    parser.add_argument("--tcn_dir_template", default="nested_piece_cv/weighted_topdown_merge56_{target_slug}_seed{seed}")
    parser.add_argument(
        "--logreg_weighted_all_template",
        default="paper_outer_baselines_weighted_topdown_all_seed{seed}_logreg",
    )
    parser.add_argument(
        "--logreg_simple_union_all_template",
        default="paper_outer_baselines_simple_union_all_seed{seed}_logreg",
    )
    parser.add_argument(
        "--logreg_weighted_note_only_template",
        default="paper_outer_baselines_weighted_topdown_note_only_seed{seed}_logreg",
    )
    parser.add_argument(
        "--logreg_weighted_xml_only_template",
        default="paper_outer_baselines_weighted_topdown_xml_only_seed{seed}_logreg",
    )
    parser.add_argument(
        "--lbdm_weighted_all_template",
        default="paper_outer_baselines_weighted_topdown_all_seed{seed}_lbdm_only",
    )
    parser.add_argument("--nonseq_dir_template", default="paper_outer_missing_baselines_seed{seed}")
    parser.add_argument("--bilstm_dir_template", default="paper_outer_baselines_weighted_topdown_all_seed{seed}_bilstm")
    parser.add_argument("--grouper_levelwise_csv", default=None)
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def normalize_target(level_label: str) -> str:
    return "L5" if str(level_label) == "L5+6" else str(level_label)


def first_non_null(series: pd.Series):
    for value in series:
        if pd.notna(value):
            return value
    return None


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


def load_outer_summary_csv(report_dir: Path, seed: int) -> list[dict[str, object]]:
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


def load_tcn_rows(report_root: Path, seed: int, dir_template: str) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing_paths: list[str] = []
    for target_label, target_slug in TCN_TARGET_DIRS.items():
        run_dir = report_root / dir_template.format(seed=seed, target_slug=target_slug)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            missing_paths.append(str(summary_path))
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        union_metrics = payload["outer_test_summary"]["union_metrics"]
        rows.append(
            {
                "seed": int(seed),
                "setting": "tcn_weighted_all",
                "target": str(target_label),
                "union_precision": float(union_metrics["union_precision"]),
                "weighted_recall": float(union_metrics["weighted_recall"]),
                "union_recall": float(union_metrics["union_recall"]),
                "consensus_recall": float(union_metrics["consensus_recall"]),
                "period_k": None,
                "report_dir": str(run_dir),
            }
        )
    return rows, missing_paths


def load_grouper_rows(grouper_levelwise_csv: str | None) -> tuple[list[dict[str, object]], str]:
    if not grouper_levelwise_csv:
        return [], "not_provided"
    grouper_path = Path(grouper_levelwise_csv).resolve()
    if not grouper_path.exists():
        return [], f"missing:{grouper_path}"
    grouper_df = pd.read_csv(grouper_path)
    expected = {"seed", "target", "union_precision", "weighted_recall", "union_recall", "consensus_recall"}
    missing = sorted(expected - set(grouper_df.columns))
    if missing:
        raise ValueError(f"Grouper CSV is missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for row in grouper_df.to_dict(orient="records"):
        rows.append(
            {
                "seed": int(row["seed"]),
                "setting": "grouper_weighted_all",
                "target": normalize_target(row["target"]),
                "union_precision": float(row["union_precision"]),
                "weighted_recall": float(row["weighted_recall"]),
                "union_recall": float(row["union_recall"]),
                "consensus_recall": float(row["consensus_recall"]),
                "period_k": None if pd.isna(row.get("period_k")) else int(row["period_k"]),
                "report_dir": str(grouper_path),
            }
        )
    return rows, "merged"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    report_root = Path(args.report_root).resolve() if args.report_root else (script_dir / "reports").resolve()
    seeds = [int(seed) for seed in args.seeds]
    seed_slug = "".join(str(seed) for seed in seeds)
    output_prefix = args.output_prefix or f"paper_outer_all_baseline_summary_seed{seed_slug}"

    raw_rows: list[dict[str, object]] = []
    source_dirs: dict[str, list[str]] = {
        "tcn": [],
        "logreg_weighted_all": [],
        "logreg_simple_union_all": [],
        "logreg_weighted_note_only": [],
        "logreg_weighted_xml_only": [],
        "lbdm_weighted_all": [],
        "nonseq": [],
        "bilstm": [],
    }
    missing_paths: list[str] = []

    root_templates = [
        ("logreg_weighted_all", args.logreg_weighted_all_template),
        ("logreg_simple_union_all", args.logreg_simple_union_all_template),
        ("logreg_weighted_note_only", args.logreg_weighted_note_only_template),
        ("logreg_weighted_xml_only", args.logreg_weighted_xml_only_template),
        ("lbdm_weighted_all", args.lbdm_weighted_all_template),
    ]

    for seed in seeds:
        tcn_rows, tcn_missing = load_tcn_rows(report_root, seed=seed, dir_template=args.tcn_dir_template)
        raw_rows.extend(tcn_rows)
        source_dirs["tcn"].extend(sorted({row["report_dir"] for row in tcn_rows}))
        missing_paths.extend(tcn_missing)

        for source_key, template in root_templates:
            report_dir = report_root / template.format(seed=seed)
            rows = load_outer_summary_csv(report_dir, seed=seed)
            if rows:
                raw_rows.extend(rows)
                source_dirs[source_key].append(str(report_dir))
            else:
                missing_paths.append(str(report_dir / "outer_summary_by_level.csv"))

        nonseq_dir = report_root / args.nonseq_dir_template.format(seed=seed)
        nonseq_rows = load_outer_summary_csv(nonseq_dir, seed=seed)
        if nonseq_rows:
            raw_rows.extend(nonseq_rows)
            source_dirs["nonseq"].append(str(nonseq_dir))
        else:
            missing_paths.append(str(nonseq_dir / "outer_summary_by_level.csv"))

        bilstm_dir = report_root / args.bilstm_dir_template.format(seed=seed)
        bilstm_rows = load_outer_summary_csv(bilstm_dir, seed=seed)
        if bilstm_rows:
            raw_rows.extend(bilstm_rows)
            source_dirs["bilstm"].append(str(bilstm_dir))
        else:
            missing_paths.append(str(bilstm_dir / "outer_summary_by_level.csv"))

    grouper_rows, grouper_status = load_grouper_rows(args.grouper_levelwise_csv)
    raw_rows.extend(grouper_rows)

    if not raw_rows:
        raise RuntimeError("No baseline outputs found to summarize")

    raw_df = pd.DataFrame(raw_rows)
    raw_df["setting_order"] = raw_df["setting"].map(SETTING_ORDER).fillna(999).astype(int)
    raw_df["target_order"] = raw_df["target"].map(TARGET_ORDER).fillna(999).astype(int)
    raw_df = raw_df.sort_values(["setting_order", "seed", "target_order"]).reset_index(drop=True)

    levelwise_df = (
        raw_df.groupby(["setting", "target"], sort=False)
        .agg(
            union_precision=("union_precision", "mean"),
            weighted_recall=("weighted_recall", "mean"),
            union_recall=("union_recall", "mean"),
            consensus_recall=("consensus_recall", "mean"),
            period_k=("period_k", first_non_null),
        )
        .reset_index()
    )
    levelwise_df["setting_order"] = levelwise_df["setting"].map(SETTING_ORDER).fillna(999).astype(int)
    levelwise_df["target_order"] = levelwise_df["target"].map(TARGET_ORDER).fillna(999).astype(int)
    levelwise_df = levelwise_df.sort_values(["setting_order", "target_order"]).reset_index(drop=True)

    mean_df = (
        levelwise_df.groupby(["setting"], sort=False)[
            ["union_precision", "weighted_recall", "union_recall", "consensus_recall"]
        ]
        .mean()
        .reset_index()
    )
    mean_df["setting_order"] = mean_df["setting"].map(SETTING_ORDER).fillna(999).astype(int)
    mean_df = mean_df.sort_values(["setting_order"]).reset_index(drop=True)

    raw_path = report_root / f"{output_prefix}_raw.csv"
    levelwise_path = report_root / f"{output_prefix}_levelwise.csv"
    mean_path = report_root / f"{output_prefix}_mean.csv"
    manifest_path = report_root / f"{output_prefix}_manifest.json"

    raw_df.drop(columns=["setting_order", "target_order"]).to_csv(raw_path, index=False)
    levelwise_df.drop(columns=["setting_order", "target_order"]).to_csv(levelwise_path, index=False)
    mean_df.drop(columns=["setting_order"]).to_csv(mean_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "report_root": str(report_root),
                "source_dirs": source_dirs,
                "missing_paths": missing_paths,
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
