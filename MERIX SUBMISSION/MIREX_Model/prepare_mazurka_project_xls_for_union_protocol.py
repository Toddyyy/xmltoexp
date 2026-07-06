from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_SCRIPT = SCRIPT_DIR / "compare_xls_mazurka_boundary_labels.py"
BEAT_TABLE = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto" / "outputs" / "beat_table_salience_auto_meter_hi8_xml.csv.gz"
OUT_ROOT = ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "outputs" / "mazurka_project_xls_l2plus_weighted"
NPZ_DIR = OUT_ROOT / "beat_data_xls_project_performer_levels"
TABLE_PATH = OUT_ROOT / "beat_table_mazurka_project_xls_hi8_xml.csv.gz"
PIECE_XLS = {
    "M17-4": ROOT / "datasets" / "mazurka17-4.xls",
    "M24-2": ROOT / "datasets" / "mazurka24-2.xls",
    "M30-2": ROOT / "datasets" / "mazurka30-2.xls",
    "M63-3": ROOT / "datasets" / "mazurka63-3.xls",
    "M68-3": ROOT / "datasets" / "mazurka68-3.xls",
}


def load_compare_module():
    spec = importlib.util.spec_from_file_location("xls_compare", COMPARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {COMPARE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return value or "unknown"


def save_npz(path: Path, boundary: np.ndarray) -> None:
    n = int(len(boundary))
    note_feats = np.zeros((1, 6), dtype=np.float32)
    beat_ids = np.zeros(1, dtype=np.int32)
    np.savez(
        path,
        note_feats=note_feats,
        beat_ids=beat_ids,
        boundary_probs=np.asarray(boundary, dtype=np.float32),
        num_beats=np.asarray(n, dtype=np.int32),
        beat_unit=np.asarray(1.0, dtype=np.float32),
    )


def main() -> None:
    cmp = load_compare_module()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BEAT_TABLE)
    base = base[base["piece_id"].isin(PIECE_XLS)].copy()
    rows = []
    manifest_rows = []
    skipped_rows = []

    for piece_id, xls_path in PIECE_XLS.items():
        beat_time_path = ROOT / "MazurkaBL-master" / "beat_time" / f"{piece_id}beat_time.csv"
        beat_grid, _ = cmp.load_beat_time(beat_time_path)
        num_beats = len(beat_grid)
        curves, skipped = cmp.load_xls_time_curves(xls_path, num_beats)
        for item in skipped:
            skipped_rows.append({"piece_id": piece_id, "xls_file": xls_path.name, **item})
        if not curves:
            raise RuntimeError(f"No usable xls curves for {piece_id}")

        one = (
            base[base["piece_id"] == piece_id]
            .sort_values(["beat_idx", "sample_id"])
            .groupby("beat_idx", as_index=False)
            .first()
            .sort_values("beat_idx")
        )
        if len(one) != num_beats:
            raise RuntimeError(f"{piece_id}: base beat table len {len(one)} != xls beat grid len {num_beats}")

        tempo_df = pd.DataFrame({"measure_number": beat_grid["measure_number"], "beat_number": beat_grid["beat_number"]})
        for performer, times in curves.items():
            tempo_df[performer] = times
        tempo_arrays = cmp.compute_tempo_curves(tempo_df, list(curves), smooth_window=3, bpm_range=(0, 5000), clip_max=600)

        for performer, curve in tempo_arrays.items():
            performer_id = safe_id(performer)
            _, level_sets = cmp.group_analysis_hierarchy(curve, cmp.STR_VEC, enforce_nested=True)
            for level in range(1, 7):
                boundary = cmp.boundaries_to_mask(num_beats, level_sets.get(level, np.array([], dtype=int)))
                save_npz(NPZ_DIR / f"{piece_id}_{performer_id}_L{level}.npz", boundary)
            source_path = str((NPZ_DIR / f"{piece_id}_{performer_id}_L1.npz").resolve())
            sample_id = f"{piece_id}_{performer_id}_XLS"
            perf_rows = one.copy()
            perf_rows["source_path"] = source_path
            perf_rows["sample_id"] = sample_id
            perf_rows["performer_id"] = performer_id
            perf_rows["level"] = 1
            perf_rows["split"] = "all"
            perf_rows["boundary_prob"] = 0.0
            perf_rows["boundary_peak"] = 0.0
            rows.append(perf_rows)
            manifest_rows.append(
                {
                    "piece_id": piece_id,
                    "performer": performer,
                    "performer_id": performer_id,
                    "source_path": source_path,
                    "num_beats": num_beats,
                }
            )

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(TABLE_PATH, index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT_ROOT / "xls_performer_manifest.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(OUT_ROOT / "skipped_xls_sheets.csv", index=False)
    print(f"Wrote table: {TABLE_PATH}")
    print(f"Wrote npz dir: {NPZ_DIR}")
    print(f"performer curves: {len(manifest_rows)}")
    print(f"rows: {len(table)}")
    print(f"skipped: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
