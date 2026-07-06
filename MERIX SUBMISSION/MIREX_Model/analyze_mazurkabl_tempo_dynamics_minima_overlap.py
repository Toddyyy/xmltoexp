from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
VELOCITY = ROOT / "MERIX SUBMISSION" / "Velocity"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
BEAT_DYN_DIR = ROOT / "MazurkaBL-master" / "beat_dyn"
OUT_DIR = MIREX / "mazurkabl_tempo_dynamics_minima_overlap"
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


def performer_cols(df: pd.DataFrame) -> list[str]:
    meta = {"Unnamed: 0", "measure_number", "beat_number"}
    return [c for c in df.columns if c not in meta]


def normalize_raw_id(raw_id: str) -> str:
    opus, num = raw_id[1:].split("-")
    return f"M{int(opus):02d}-{int(num)}"


def timestamps_to_bpm_curve(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    bpm = np.full(len(times), np.nan, dtype=float)
    valid = np.isfinite(dt) & (dt > 0)
    bpm[1:][valid] = 60.0 / dt[valid]
    if len(bpm) > 1:
        bpm[0] = bpm[1]
    s = pd.Series(bpm)
    s = s.interpolate("linear", limit_direction="both")
    s = s.rolling(window=3, center=True, min_periods=1).mean()
    # Keep extreme rubato spikes from dominating higher-level energy.
    lo, hi = np.nanpercentile(s.to_numpy(), [1, 99])
    return s.clip(lower=lo, upper=hi).to_numpy(dtype=float)


def dyn_curve(vals: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(vals, dtype=float))
    s = s.where((s >= 0.0) & (s <= 1.0))
    s = s.interpolate("linear", limit_direction="both")
    s = s.rolling(window=3, center=True, min_periods=1).mean()
    return s.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)


def set_metrics(a: set[int], b: set[int], tol: int = 1) -> dict[str, float]:
    exact = len(a & b)
    union = len(a | b)
    matched_a = set()
    matched_b = set()
    for x in sorted(a):
        candidates = [y for y in b if abs(y - x) <= tol and y not in matched_b]
        if candidates:
            y = min(candidates, key=lambda z: (abs(z - x), z))
            matched_a.add(x)
            matched_b.add(y)
    precision = len(matched_a) / len(a) if a else np.nan
    recall = len(matched_b) / len(b) if b else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    return {
        "tempo_events": len(a),
        "dyn_events": len(b),
        "exact_jaccard": exact / union if union else np.nan,
        "tempo_to_dyn_precision_tol1": precision,
        "dyn_to_tempo_recall_tol1": recall,
        "f1_tol1": f1,
    }


def nearest_offsets(source: set[int], target: set[int], max_abs: int = 6) -> list[int]:
    if not source or not target:
        return []
    target_sorted = np.array(sorted(target), dtype=int)
    offsets = []
    for x in sorted(source):
        nearest = target_sorted[np.argmin(np.abs(target_sorted - x))]
        off = int(nearest - x)
        if abs(off) <= max_abs:
            offsets.append(off)
    return offsets


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vel = load_velocity_module()
    performer_rows = []
    consensus_rows = []
    offset_rows = []

    for time_path in sorted(BEAT_TIME_DIR.glob("*beat_time.csv")):
        raw_id = time_path.name.replace("beat_time.csv", "")
        piece = normalize_raw_id(raw_id)
        dyn_path = BEAT_DYN_DIR / f"{raw_id}beat_dynNORM.csv"
        if not dyn_path.exists():
            continue
        time_df = pd.read_csv(time_path)
        dyn_df = pd.read_csv(dyn_path)
        common = sorted(set(performer_cols(time_df)) & set(performer_cols(dyn_df)))
        n = min(len(time_df), len(dyn_df))
        tempo_counts = {level: np.zeros(n, dtype=float) for level in range(1, 7)}
        dyn_counts = {level: np.zeros(n, dtype=float) for level in range(1, 7)}

        for perf in common:
            tempo = timestamps_to_bpm_curve(pd.to_numeric(time_df[perf], errors="coerce").to_numpy())[:n]
            dyn = dyn_curve(pd.to_numeric(dyn_df[perf], errors="coerce").to_numpy())[:n]
            _, tempo_sets = vel.group_analysis_hierarchy(tempo, STR_VEC, enforce_nested=False)
            _, dyn_sets = vel.group_analysis_hierarchy(dyn, STR_VEC, enforce_nested=False)
            for level in range(1, 7):
                t_set = set(np.asarray(tempo_sets[level], dtype=int).tolist())
                d_set = set(np.asarray(dyn_sets[level], dtype=int).tolist())
                t_set = {x for x in t_set if 0 <= x < n}
                d_set = {x for x in d_set if 0 <= x < n}
                for x in t_set:
                    tempo_counts[level][x] += 1.0
                for x in d_set:
                    dyn_counts[level][x] += 1.0
                row = set_metrics(t_set, d_set, tol=1)
                row.update({"piece": piece, "performer": perf, "level": level})
                performer_rows.append(row)
                for off in nearest_offsets(t_set, d_set, max_abs=6):
                    offset_rows.append({"piece": piece, "performer": perf, "level": level, "dyn_minus_tempo": off})

        denom = max(len(common), 1)
        for level in range(1, 7):
            tc = tempo_counts[level] / denom
            dc = dyn_counts[level] / denom
            t_events = set(np.where(tc >= CONSENSUS_THRESHOLD)[0].tolist())
            d_events = set(np.where(dc >= CONSENSUS_THRESHOLD)[0].tolist())
            row = set_metrics(t_events, d_events, tol=1)
            row.update(
                {
                    "piece": piece,
                    "level": level,
                    "performers": len(common),
                    "consensus_corr": corr_safe(tc, dc),
                    "tempo_consensus_sum": float(tc.sum()),
                    "dyn_consensus_sum": float(dc.sum()),
                    "tempo_consensus_max": float(tc.max()) if len(tc) else np.nan,
                    "dyn_consensus_max": float(dc.max()) if len(dc) else np.nan,
                }
            )
            consensus_rows.append(row)

    performer_df = pd.DataFrame(performer_rows)
    consensus_df = pd.DataFrame(consensus_rows)
    offsets_df = pd.DataFrame(offset_rows)
    performer_df.to_csv(OUT_DIR / "performer_level_tempo_dynamics_minima_overlap.csv", index=False)
    consensus_df.to_csv(OUT_DIR / "piece_level_consensus_tempo_dynamics_minima_overlap.csv", index=False)
    offsets_df.to_csv(OUT_DIR / "nearest_dyn_minima_offsets_from_tempo_minima.csv", index=False)

    summary = []
    for level in range(1, 7):
        p = performer_df[performer_df["level"] == level]
        c = consensus_df[consensus_df["level"] == level]
        off = offsets_df[offsets_df["level"] == level]["dyn_minus_tempo"]
        summary.append(
            {
                "level": level,
                "performer_mean_tempo_events": p["tempo_events"].mean(),
                "performer_mean_dyn_events": p["dyn_events"].mean(),
                "performer_mean_exact_jaccard": p["exact_jaccard"].mean(),
                "performer_mean_f1_tol1": p["f1_tol1"].mean(),
                "consensus_mean_exact_jaccard": c["exact_jaccard"].mean(),
                "consensus_mean_f1_tol1": c["f1_tol1"].mean(),
                "consensus_mean_corr": c["consensus_corr"].mean(),
                "nearest_offset_mode_dyn_minus_tempo": int(off.mode().iloc[0]) if len(off) else np.nan,
                "nearest_offset_mean_dyn_minus_tempo": off.mean() if len(off) else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "summary_by_level.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(1, 7)
    ax.plot(x, summary_df["performer_mean_f1_tol1"], marker="o", label="same performer minima F1 +/-1")
    ax.plot(x, summary_df["consensus_mean_f1_tol1"], marker="o", label="consensus minima F1 +/-1")
    ax.plot(x, summary_df["consensus_mean_corr"], marker="o", label="consensus correlation")
    ax.set_xticks(x)
    ax.set_xlabel("level")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Tempo minima vs dynamics minima overlap by level")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tempo_dynamics_minima_overlap_by_level.png", dpi=180)
    plt.close(fig)

    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
