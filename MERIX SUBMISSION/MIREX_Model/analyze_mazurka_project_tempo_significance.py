from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets"
PLOT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurka_project_l1_l6_tempo_human_phrase_plots"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurka_project_tempo_significance"
ALPHA = 0.05


def normalize_piece_id(path: Path) -> str:
    m = re.search(r"mazurka(\d+)-(\d+)", path.stem, flags=re.I)
    if not m:
        return path.stem
    return f"M{int(m.group(1)):02d}-{int(m.group(2))}"


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    q[valid] = out
    return q


def load_xls_summary(path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name="Summary", header=None)
    header = raw.iloc[1].astype(str).str.strip().str.lower()
    measure_col = int(np.flatnonzero(header == "measure")[0])
    beat_col = int(np.flatnonzero(header == "beat")[0])
    non_event_matches = np.flatnonzero(header == "non-event")
    tempo_start = int(non_event_matches[0] + 1) if len(non_event_matches) else beat_col + 1

    data = raw.iloc[2:].reset_index(drop=True)
    measure = pd.to_numeric(data.iloc[:, measure_col], errors="coerce")
    beat = pd.to_numeric(data.iloc[:, beat_col], errors="coerce")
    valid = measure.notna() & beat.notna()
    data = data.loc[valid].reset_index(drop=True)
    tempo = data.iloc[:, tempo_start:].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    tempo = tempo.interpolate(axis=0, limit_direction="both").clip(lower=1, upper=600)
    performer_names = [
        str(raw.iloc[0, col]) if not pd.isna(raw.iloc[0, col]) else f"perf_{i+1:03d}"
        for i, col in enumerate(tempo.columns)
    ]
    tempo.columns = performer_names
    meta = pd.DataFrame(
        {
            "beat": np.arange(1, len(data) + 1),
            "measure": pd.to_numeric(data.iloc[:, measure_col], errors="coerce").to_numpy(),
            "measure_beat": pd.to_numeric(data.iloc[:, beat_col], errors="coerce").to_numpy(),
        }
    )
    phrase_path = PLOT_DIR / f"{normalize_piece_id(path)}_beat_target_values.csv"
    if phrase_path.exists():
        phrase = pd.read_csv(phrase_path)
        meta["human_phrase_label"] = phrase["human_phrase_label"].fillna("").astype(str).reindex(meta.index).fillna("")
        meta["target_weighted_l2plus"] = phrase["target_weighted_l2plus"].reindex(meta.index).to_numpy()
    else:
        meta["human_phrase_label"] = ""
        meta["target_weighted_l2plus"] = np.nan
    return normalize_piece_id(path), meta, tempo


def ttest_by_beat(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(values, axis=1)
    t_stat, p_val = stats.ttest_1samp(values, popmean=0.0, axis=1, nan_policy="omit")
    q_val = bh_fdr(p_val)
    return means, p_val, q_val


def analyze_piece(path: Path) -> tuple[pd.DataFrame, dict]:
    piece_id, meta, tempo_df = load_xls_summary(path)
    tempo = tempo_df.to_numpy(dtype=float)
    log_tempo = np.log(np.clip(tempo, 1e-6, None))

    performer_center = np.nanmean(log_tempo, axis=0, keepdims=True)
    performer_std = np.nanstd(log_tempo, axis=0, keepdims=True)
    performer_std[performer_std < 1e-8] = 1.0
    log_tempo_z = (log_tempo - performer_center) / performer_std
    delta_log_tempo = np.diff(log_tempo, axis=0, prepend=np.nan)

    mean_z, p_z, q_z = ttest_by_beat(log_tempo_z)
    mean_delta, p_delta, q_delta = ttest_by_beat(delta_log_tempo)

    result = meta.copy()
    result["piece_id"] = piece_id
    result["num_performers"] = tempo.shape[1]
    result["mean_tempo_bpm"] = np.nanmean(tempo, axis=1)
    result["mean_log_tempo_z"] = mean_z
    result["tempo_level_p"] = p_z
    result["tempo_level_q_fdr"] = q_z
    result["tempo_level_effect"] = np.where(mean_z > 0, "fast", "slow")
    result["tempo_level_significant"] = q_z < ALPHA
    result["mean_delta_log_tempo"] = mean_delta
    result["tempo_change_p"] = p_delta
    result["tempo_change_q_fdr"] = q_delta
    result["tempo_change_effect"] = np.where(mean_delta > 0, "accelerate", "decelerate")
    result["tempo_change_significant"] = q_delta < ALPHA

    summary = {
        "piece_id": piece_id,
        "num_beats": int(tempo.shape[0]),
        "num_performers": int(tempo.shape[1]),
        "fast_level_sig_count": int(((q_z < ALPHA) & (mean_z > 0)).sum()),
        "slow_level_sig_count": int(((q_z < ALPHA) & (mean_z < 0)).sum()),
        "accelerate_sig_count": int(((q_delta < ALPHA) & (mean_delta > 0)).sum()),
        "decelerate_sig_count": int(((q_delta < ALPHA) & (mean_delta < 0)).sum()),
        "human_phrase_count": int((result["human_phrase_label"].fillna("").astype(str).str.len() > 0).sum()),
    }
    return result, summary


def plot_piece(result: pd.DataFrame, summary: dict) -> None:
    piece_id = summary["piece_id"]
    x = result["beat"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.8, 1.8]})

    axes[0].plot(x, result["mean_tempo_bpm"], color="#222222", linewidth=1.4)
    axes[0].set_ylabel("BPM")
    axes[0].set_title(f"{piece_id}: cross-performer tempo significance")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(x, result["mean_log_tempo_z"], color="#444444", linewidth=1.0)
    fast = result["tempo_level_significant"] & (result["mean_log_tempo_z"] > 0)
    slow = result["tempo_level_significant"] & (result["mean_log_tempo_z"] < 0)
    axes[1].scatter(result.loc[fast, "beat"], result.loc[fast, "mean_log_tempo_z"], color="#d62728", s=16, label="significantly faster")
    axes[1].scatter(result.loc[slow, "beat"], result.loc[slow, "mean_log_tempo_z"], color="#1f77b4", s=16, label="significantly slower")
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    axes[1].set_ylabel("tempo z")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].plot(x, result["mean_delta_log_tempo"], color="#444444", linewidth=1.0)
    acc = result["tempo_change_significant"] & (result["mean_delta_log_tempo"] > 0)
    dec = result["tempo_change_significant"] & (result["mean_delta_log_tempo"] < 0)
    axes[2].scatter(result.loc[acc, "beat"], result.loc[acc, "mean_delta_log_tempo"], color="#d62728", s=16, label="significant acceleration")
    axes[2].scatter(result.loc[dec, "beat"], result.loc[dec, "mean_delta_log_tempo"], color="#1f77b4", s=16, label="significant deceleration")
    axes[2].axhline(0, color="#888888", linewidth=0.8)
    axes[2].set_ylabel("delta log tempo")
    axes[2].set_xlabel("score beat")
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)

    phrase = result["human_phrase_label"].fillna("").astype(str).str.len() > 0
    for ax in axes:
        for beat in result.loc[phrase, "beat"]:
            ax.axvline(beat, color="#6a3d9a", alpha=0.28, linewidth=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{piece_id}_tempo_significance.png", dpi=180)
    fig.savefig(OUT_DIR / f"{piece_id}_tempo_significance.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    summaries = []
    for path in sorted(DATA_DIR.glob("mazurka*.xls")):
        result, summary = analyze_piece(path)
        result.to_csv(OUT_DIR / f"{summary['piece_id']}_tempo_significance_by_beat.csv", index=False)
        plot_piece(result, summary)
        all_results.append(result)
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "tempo_significance_summary.csv", index=False)
    pd.concat(all_results, ignore_index=True).to_csv(OUT_DIR / "tempo_significance_all_beats.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
