from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASAP_ROOT = ROOT.parent / "datasets" / "ASAP"
BUILDERS = ROOT / "src" / "data_builders"
DATA = ROOT / "data"
MAZURKA_BUILD = BUILDERS / "build_mazurka_beat_npz_performer_levels.py"
TOP_N = int(os.environ.get("ASAP_TOP_N", "30"))
DATASET_NAME = f"asap{TOP_N}"
OUT_DIR = DATA / "labels" / f"{DATASET_NAME}_tempo_boundary_labels"
NPZ_DIR = OUT_DIR / f"beat_data_asap_top{TOP_N}_performer_levels"

HIERARCHY_DEPTH = 6
SQRT_MASS_WEIGHTS = {
    2: 0.205,
    3: 0.284,
    4: 0.408,
    5: 0.613,
    6: 1.000,
}


def load_mazurka_builder():
    spec = importlib.util.spec_from_file_location("mazurka_level_builder_for_asap", MAZURKA_BUILD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MAZURKA_BUILD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_level_builder_for_asap"] = module
    spec.loader.exec_module(module)
    return module


builder = load_mazurka_builder()


def safe_id(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def performance_id(path: str) -> str:
    return safe_id(Path(path).with_suffix("").name)


def tempo_curve_from_beats(beats: list[float], smooth_window: int = 3, clip_max: float = 600.0) -> np.ndarray:
    times = pd.to_numeric(pd.Series(beats), errors="coerce").to_numpy(dtype=float)
    dt = np.diff(times, prepend=np.nan)
    tempo = 60.0 / dt
    tempo[(dt <= 0) | (~np.isfinite(tempo))] = np.nan
    s = pd.Series(tempo)
    s = s.where((s > 0) & (s < 5000))
    s = s.interpolate("linear", limit_direction="both")
    s = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
    s = s.clip(upper=clip_max)
    return s.to_numpy(dtype=np.float32)


def infer_beats_per_measure(ann: dict) -> tuple[int, str]:
    signatures = ann.get("midi_score_time_signatures") or ann.get("perf_time_signatures") or {}
    counts: dict[int, int] = {}
    labels: dict[int, str] = {}
    for value in signatures.values():
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        label = str(value[0])
        try:
            beats = int(value[1])
        except (TypeError, ValueError):
            continue
        if beats <= 0:
            continue
        counts[beats] = counts.get(beats, 0) + 1
        labels.setdefault(beats, label)
    if not counts:
        return 4, "unknown_default_4"
    beats = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return int(beats), labels.get(beats, str(beats))


def str_vec_for_performance(ann: dict) -> tuple[np.ndarray, int, str]:
    beats_per_measure, meter_label = infer_beats_per_measure(ann)
    str_vec = np.asarray([beats_per_measure] + [2] * (HIERARCHY_DEPTH - 1), dtype=int)
    return str_vec, beats_per_measure, meter_label


def load_inventory(metadata: pd.DataFrame, annotations: dict) -> pd.DataFrame:
    rows = []
    for folder, group in metadata.groupby("folder", sort=False):
        usable_paths = []
        beat_lengths = []
        aligned_false = 0
        missing_json = 0
        length_mismatch = 0
        beats_per_measure_values = []
        meter_labels = []
        for row in group.itertuples(index=False):
            perf_path = str(row.midi_performance)
            ann = annotations.get(perf_path)
            if ann is None:
                missing_json += 1
                continue
            if not bool(ann.get("score_and_performance_aligned", False)):
                aligned_false += 1
                continue
            perf_beats = ann.get("performance_beats", [])
            score_beats = ann.get("midi_score_beats", [])
            if len(perf_beats) != len(score_beats) or len(perf_beats) < 8:
                length_mismatch += 1
                continue
            beats_per_measure, meter_label = infer_beats_per_measure(ann)
            beats_per_measure_values.append(beats_per_measure)
            meter_labels.append(meter_label)
            usable_paths.append(perf_path)
            beat_lengths.append(len(perf_beats))
        main_beats_per_measure = (
            int(pd.Series(beats_per_measure_values).mode().iloc[0]) if beats_per_measure_values else 0
        )
        main_meter_label = (
            str(pd.Series(meter_labels).mode().iloc[0]) if meter_labels else ""
        )
        rows.append(
            {
                "piece_id": safe_id(folder),
                "folder": folder,
                "composer": group["composer"].iloc[0],
                "title": group["title"].iloc[0],
                "xml_score": group["xml_score"].iloc[0],
                "midi_score": group["midi_score"].iloc[0],
                "metadata_performances": int(len(group)),
                "usable_aligned_performances": int(len(usable_paths)),
                "min_beats": int(min(beat_lengths)) if beat_lengths else 0,
                "max_beats": int(max(beat_lengths)) if beat_lengths else 0,
                "unique_beat_lengths": int(len(set(beat_lengths))),
                "main_beats_per_measure": main_beats_per_measure,
                "main_meter_label": main_meter_label,
                "main_str_vec": (
                    " ".join(str(x) for x in [main_beats_per_measure] + [2] * (HIERARCHY_DEPTH - 1))
                    if main_beats_per_measure
                    else ""
                ),
                "aligned_false": int(aligned_false),
                "missing_json": int(missing_json),
                "length_mismatch": int(length_mismatch),
                "usable_performance_paths": "|".join(usable_paths),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["usable_aligned_performances", "metadata_performances", "max_beats"],
        ascending=[False, False, False],
    )


def build_piece(
    piece_row: pd.Series,
    annotations: dict,
    selected_rank: int,
) -> tuple[list[dict], list[dict], dict[int, np.ndarray], dict]:
    piece_id = str(piece_row["piece_id"])
    perf_paths = [p for p in str(piece_row["usable_performance_paths"]).split("|") if p]
    level_arrays: dict[int, list[np.ndarray]] = {level: [] for level in range(1, 7)}
    tempo_rows = []
    perf_rows = []
    n_beats_expected: int | None = None
    str_vec_expected: tuple[int, ...] | None = None
    beats_per_measure_expected: int | None = None
    meter_label_expected: str | None = None

    for perf_idx, perf_path in enumerate(perf_paths):
        ann = annotations[perf_path]
        tempo = tempo_curve_from_beats(ann["performance_beats"])
        str_vec, beats_per_measure, meter_label = str_vec_for_performance(ann)
        if n_beats_expected is None:
            n_beats_expected = int(len(tempo))
            str_vec_expected = tuple(int(x) for x in str_vec)
            beats_per_measure_expected = int(beats_per_measure)
            meter_label_expected = str(meter_label)
        if len(tempo) != n_beats_expected:
            perf_rows.append(
                {
                    "piece_id": piece_id,
                    "performance_path": perf_path,
                    "performance_id": performance_id(perf_path),
                    "status": "skipped_inconsistent_tempo_length",
                    "num_beats": int(len(tempo)),
                }
            )
            continue
        if tuple(int(x) for x in str_vec) != str_vec_expected:
            perf_rows.append(
                {
                    "piece_id": piece_id,
                    "performance_path": perf_path,
                    "performance_id": performance_id(perf_path),
                    "status": "skipped_inconsistent_meter",
                    "num_beats": int(len(tempo)),
                    "beats_per_measure": int(beats_per_measure),
                    "meter_label": str(meter_label),
                    "str_vec": " ".join(str(int(x)) for x in str_vec),
                }
            )
            continue

        results_raw, level_sets = builder.group_analysis_hierarchy(tempo, str_vec, enforce_nested=True)
        for level in range(1, 7):
            boundary = np.zeros(n_beats_expected, dtype=np.float32)
            boundary[np.asarray(level_sets[level], dtype=int)] = 1.0
            level_arrays[level].append(boundary)
            np.savez_compressed(
                NPZ_DIR / f"{piece_id}_{perf_idx:04d}_L{level}.npz",
                piece_id=np.asarray(piece_id),
                performance_id=np.asarray(performance_id(perf_path)),
                performance_path=np.asarray(perf_path),
                level=np.asarray(level, dtype=np.int16),
                boundary_probs=boundary.astype(np.float32),
                tempo_curve=tempo.astype(np.float32),
                str_vec=str_vec.astype(np.int16),
                beats_per_measure=np.asarray(beats_per_measure, dtype=np.int16),
                meter_label=np.asarray(str(meter_label)),
                enforce_nested=np.asarray(True),
            )

        for beat_idx, bpm in enumerate(tempo):
            tempo_rows.append(
                {
                    "piece_id": piece_id,
                    "performance_id": performance_id(perf_path),
                    "performance_path": perf_path,
                    "beat_idx": int(beat_idx),
                    "tempo_bpm": float(bpm),
                }
            )
        perf_rows.append(
            {
                "piece_id": piece_id,
                "performance_path": perf_path,
                "performance_id": performance_id(perf_path),
                "status": "ok",
                "num_beats": int(len(tempo)),
                "tempo_mean": float(np.nanmean(tempo)),
                "tempo_std": float(np.nanstd(tempo)),
                "beats_per_measure": int(beats_per_measure),
                "meter_label": str(meter_label),
                "str_vec": " ".join(str(int(x)) for x in str_vec),
                **{f"L{level}_count": int(level_arrays[level][-1].sum()) for level in range(1, 7)},
            }
        )

    consensus = {}
    for level, arrays in level_arrays.items():
        if arrays:
            consensus[level] = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
        else:
            consensus[level] = np.zeros(0, dtype=np.float32)

    n_ok = int(sum(1 for row in perf_rows if row["status"] == "ok"))
    target = np.zeros_like(consensus[2])
    if target.size:
        target = np.max(
            np.stack([SQRT_MASS_WEIGHTS[level] * consensus[level] for level in range(2, 7)], axis=0),
            axis=0,
        ).astype(np.float32)

    summary = {
        "selected_rank": int(selected_rank),
        "piece_id": piece_id,
        "folder": piece_row["folder"],
        "composer": piece_row["composer"],
        "title": piece_row["title"],
        "metadata_performances": int(piece_row["metadata_performances"]),
        "usable_aligned_performances": int(piece_row["usable_aligned_performances"]),
        "processed_performances": n_ok,
        "num_beats": int(len(target)),
        "beats_per_measure": int(beats_per_measure_expected or 0),
        "meter_label": str(meter_label_expected or ""),
        "str_vec": " ".join(str(int(x)) for x in (str_vec_expected or ())),
        "target_sum_sqrtmass_l2plus": float(target.sum()) if target.size else 0.0,
        "target_ge_0p01": int(np.count_nonzero(target >= 0.01)) if target.size else 0,
        "target_ge_0p03": int(np.count_nonzero(target >= 0.03)) if target.size else 0,
        "target_ge_0p05": int(np.count_nonzero(target >= 0.05)) if target.size else 0,
    }
    for level in range(1, 7):
        c = consensus[level]
        summary[f"L{level}_consensus_support_gt0"] = int(np.count_nonzero(c > 0))
        summary[f"L{level}_consensus_mass"] = float(c.sum()) if c.size else 0.0
        summary[f"L{level}_mean_boundary_count_per_perf"] = float(c.sum()) if c.size else 0.0
        summary[f"L{level}_consensus_ge_0p10"] = int(np.count_nonzero(c >= 0.10))
        summary[f"L{level}_consensus_ge_0p25"] = int(np.count_nonzero(c >= 0.25))
        summary[f"L{level}_consensus_ge_0p50"] = int(np.count_nonzero(c >= 0.50))
    return tempo_rows, perf_rows, consensus, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ASAP_ROOT / "metadata.csv")
    annotations = json.loads((ASAP_ROOT / "asap_annotations.json").read_text(encoding="utf-8"))

    inventory = load_inventory(metadata, annotations)
    inventory.to_csv(OUT_DIR / "asap_piece_inventory.csv", index=False)

    strict = inventory[inventory["usable_aligned_performances"] > 25].copy()
    strict.to_csv(OUT_DIR / "asap_gt25_manifest.csv", index=False)

    selected = inventory[inventory["usable_aligned_performances"] > 0].head(TOP_N).copy()
    selected.insert(0, "selected_rank", np.arange(1, len(selected) + 1))
    selected.to_csv(OUT_DIR / f"asap_top{TOP_N}_manifest.csv", index=False)

    all_tempo_rows = []
    all_perf_rows = []
    summaries = []
    consensus_rows = []
    for row in selected.itertuples(index=False):
        piece_row = pd.Series(row._asdict())
        tempo_rows, perf_rows, consensus, summary = build_piece(piece_row, annotations, int(piece_row["selected_rank"]))
        all_tempo_rows.extend(tempo_rows)
        all_perf_rows.extend(perf_rows)
        summaries.append(summary)
        for level, values in consensus.items():
            for beat_idx, value in enumerate(values):
                consensus_rows.append(
                    {
                "piece_id": summary["piece_id"],
                "str_vec": summary["str_vec"],
                "level": int(level),
                        "beat_idx": int(beat_idx),
                        "consensus": float(value),
                    }
                )

    pd.DataFrame(all_perf_rows).to_csv(OUT_DIR / f"asap_top{TOP_N}_performance_summary.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT_DIR / f"asap_top{TOP_N}_boundary_summary.csv", index=False)
    pd.DataFrame(all_tempo_rows).to_csv(OUT_DIR / f"asap_top{TOP_N}_tempo_curves_long.csv.gz", index=False, compression="gzip")
    pd.DataFrame(consensus_rows).to_csv(OUT_DIR / f"asap_top{TOP_N}_level_consensus_long.csv.gz", index=False, compression="gzip")

    summary_df = pd.DataFrame(summaries)
    aggregate = {
        "inventory_pieces": int(len(inventory)),
        "strict_gt25_pieces": int(len(strict)),
        f"selected_top{TOP_N}_pieces": int(len(selected)),
        "selected_processed_performances": int(summary_df["processed_performances"].sum()),
        "selected_total_beats": int(summary_df["num_beats"].sum()),
        "selected_min_usable_performances": int(summary_df["usable_aligned_performances"].min()),
        "selected_max_usable_performances": int(summary_df["usable_aligned_performances"].max()),
        "selected_target_ge_0p01": int(summary_df["target_ge_0p01"].sum()),
        "selected_target_ge_0p03": int(summary_df["target_ge_0p03"].sum()),
        "selected_target_ge_0p05": int(summary_df["target_ge_0p05"].sum()),
        "str_vec_rule": "[beats_per_measure, 2, 2, 2, 2, 2]",
        "enforce_nested": True,
        "tempo_curve_rule": "60/diff(performance_beats, prepend=nan), interpolate, rolling window=3, clip<=600",
        "target_rule": "max_{L2..L6}(sqrt-mass weight_L * performer consensus_L)",
        "sqrt_mass_weights": json.dumps(SQRT_MASS_WEIGHTS),
    }
    pd.DataFrame([aggregate]).to_csv(OUT_DIR / f"asap_top{TOP_N}_aggregate_summary.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("Strict >25 usable aligned pieces:")
    if strict.empty:
        print("(none)")
    else:
        print(strict[["piece_id", "composer", "title", "usable_aligned_performances", "min_beats", "max_beats"]].to_string(index=False))
    print(f"\nSelected top{TOP_N} by usable aligned performance count:")
    print(selected[["selected_rank", "piece_id", "composer", "title", "usable_aligned_performances", "min_beats", "max_beats"]].to_string(index=False))
    print("\nAggregate:")
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
