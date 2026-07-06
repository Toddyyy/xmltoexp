from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
VELOCITY = ROOT / "MERIX SUBMISSION" / "Velocity"
BEAT_DYN_DIR = ROOT / "MazurkaBL-master" / "beat_dyn"
MARKINGS_DYN_DIR = ROOT / "MazurkaBL-master" / "markings_dyn"
TEMPO_LABEL_DIR = MIREX / "beat_data_mazurka_performer_levels"
OUT_DIR = MIREX / "mazurkabl_dynamics_curve_signal"
STR_VEC = [3, 2, 2, 2, 2, 2]
CONSENSUS_THRESHOLD = 0.05


def load_velocity_module():
    path = VELOCITY / "build_mazurka_velocity_npz_performer_levels.py"
    spec = importlib.util.spec_from_file_location("velocity_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_raw_id(raw_id: str) -> str:
    opus, num = raw_id[1:].split("-")
    return f"M{int(opus):02d}-{int(num)}"


def consensus_from_npz(piece: str, level: int) -> np.ndarray | None:
    arrays = []
    for path in sorted(TEMPO_LABEL_DIR.glob(f"{piece}_*_L{level}.npz")):
        with np.load(path) as data:
            arrays.append(np.asarray(data["boundary_probs"], dtype=float))
    if not arrays:
        return None
    n = min(len(a) for a in arrays)
    return np.vstack([a[:n] for a in arrays]).mean(axis=0)


def event_metrics(a: np.ndarray, b: np.ndarray, tol: int = 1) -> dict[str, float]:
    a_set = set(np.where(a >= CONSENSUS_THRESHOLD)[0].tolist())
    b_set = set(np.where(b >= CONSENSUS_THRESHOLD)[0].tolist())
    if not a_set and not b_set:
        return {
            "a_events": 0,
            "b_events": 0,
            "exact_jaccard": 1.0,
            "tol_precision": 1.0,
            "tol_recall": 1.0,
            "tol_f1": 1.0,
        }
    inter = len(a_set & b_set)
    union = len(a_set | b_set)

    matched_a = set()
    matched_b = set()
    for x in sorted(a_set):
        candidates = [y for y in b_set if abs(y - x) <= tol and y not in matched_b]
        if candidates:
            y = min(candidates, key=lambda z: (abs(z - x), z))
            matched_a.add(x)
            matched_b.add(y)
    precision = len(matched_a) / len(a_set) if a_set else 0.0
    recall = len(matched_b) / len(b_set) if b_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "a_events": len(a_set),
        "b_events": len(b_set),
        "exact_jaccard": inter / union if union else 0.0,
        "tol_precision": precision,
        "tol_recall": recall,
        "tol_f1": f1,
    }


def parse_dynamic_markings(raw_id: str) -> list[int]:
    path = MARKINGS_DYN_DIR / f"{raw_id}markings.csv"
    if not path.exists():
        return []
    rows = list(csv.reader(path.open()))
    if len(rows) < 2:
        return []
    positions = []
    for cell in rows[1]:
        try:
            positions.append(int(float(cell)) - 1)
        except ValueError:
            continue
    return positions


def marking_recall(consensus: np.ndarray, markings: list[int], tol: int = 1) -> float:
    events = set(np.where(consensus >= CONSENSUS_THRESHOLD)[0].tolist())
    valid = [p for p in markings if 0 <= p < len(consensus)]
    if not valid:
        return np.nan
    hit = 0
    for p in valid:
        if any(abs(e - p) <= tol for e in events):
            hit += 1
    return hit / len(valid)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vel = load_velocity_module()
    overlap_rows = []
    common_rows = []
    marking_rows = []

    for file_path in sorted(BEAT_DYN_DIR.glob("*beat_dynNORM.csv")):
        raw_id = file_path.name.replace("beat_dynNORM.csv", "")
        piece = normalize_raw_id(raw_id)
        df, performer_cols = vel.load_beat_dyn(file_path)
        curves = vel.compute_dyn_curves(df, performer_cols, smooth_window=3)
        n = len(df)
        counts = {level: np.zeros(n, dtype=float) for level in range(1, 7)}

        for curve in curves.values():
            _, level_sets = vel.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=True)
            for level in range(1, 7):
                idx = np.asarray(level_sets[level], dtype=int)
                idx = idx[(idx >= 0) & (idx < n)]
                counts[level][idx] += 1.0

        markings = parse_dynamic_markings(raw_id)
        for level in range(1, 7):
            dyn_consensus = counts[level] / max(len(curves), 1)
            tempo_consensus = consensus_from_npz(piece, level)
            common_rows.append(
                {
                    "piece": piece,
                    "level": level,
                    "performers": len(curves),
                    "beats": n,
                    "dyn_event_count_ge_0p05": int((dyn_consensus >= CONSENSUS_THRESHOLD).sum()),
                    "dyn_consensus_sum": float(dyn_consensus.sum()),
                    "dyn_consensus_max": float(dyn_consensus.max()) if len(dyn_consensus) else 0.0,
                }
            )
            marking_rows.append(
                {
                    "piece": piece,
                    "level": level,
                    "dynamic_markings": len(markings),
                    "marking_recall_tol1": marking_recall(dyn_consensus, markings, tol=1),
                    "marking_recall_tol2": marking_recall(dyn_consensus, markings, tol=2),
                }
            )
            if tempo_consensus is None:
                continue
            m = min(len(dyn_consensus), len(tempo_consensus))
            metrics = event_metrics(dyn_consensus[:m], tempo_consensus[:m], tol=1)
            metrics.update({"piece": piece, "level": level})
            overlap_rows.append(metrics)

    pd.DataFrame(common_rows).to_csv(OUT_DIR / "dynamics_level_commonality.csv", index=False)
    pd.DataFrame(marking_rows).to_csv(OUT_DIR / "dynamics_marking_alignment.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT_DIR / "dynamics_vs_tempo_overlap.csv", index=False)

    overlap = pd.DataFrame(overlap_rows)
    common = pd.DataFrame(common_rows)
    marks = pd.DataFrame(marking_rows)
    summary = []
    for level in range(1, 7):
        o = overlap[overlap["level"] == level]
        c = common[common["level"] == level]
        ma = marks[marks["level"] == level]
        summary.append(
            {
                "level": level,
                "mean_dyn_events_ge_0p05": c["dyn_event_count_ge_0p05"].mean(),
                "mean_dyn_consensus_max": c["dyn_consensus_max"].mean(),
                "mean_tempo_dyn_exact_jaccard": o["exact_jaccard"].mean(),
                "mean_tempo_dyn_tol1_f1": o["tol_f1"].mean(),
                "mean_dyn_marking_recall_tol1": ma["marking_recall_tol1"].mean(),
                "mean_dyn_marking_recall_tol2": ma["marking_recall_tol2"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "summary_by_level.csv", index=False)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
