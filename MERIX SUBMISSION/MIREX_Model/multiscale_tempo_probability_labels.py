from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "multiscale_tempo_probability_labels"
DEFAULT_LEVELS = (1, 3, 6, 12, 24)


def safe_prob(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nanmedian(values[np.isfinite(values)]) if np.isfinite(values).any() else 1.0)
    values = np.maximum(values, eps)
    inv_sq = 1.0 / (values**2)
    total = float(np.sum(inv_sq))
    if total <= 0 or not np.isfinite(total):
        return np.full_like(inv_sq, 1.0 / max(len(inv_sq), 1), dtype=np.float64)
    return inv_sq / total


def window_e(segment: np.ndarray, mode: str) -> float:
    segment = np.asarray(segment, dtype=np.float64)
    segment = segment[np.isfinite(segment)]
    if segment.size == 0:
        return 1.0
    if mode == "mean":
        return float(np.mean(segment))
    if mode == "min":
        return float(np.min(segment))
    if mode == "rms_minus_std":
        return float(np.sqrt(np.mean(segment**2)) - np.std(segment))
    raise ValueError(f"Unsupported e_mode: {mode}")


def multiscale_boundary_probability(
    tempo: np.ndarray,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    e_mode: str = "mean",
    normalize_output: str = "max",
) -> tuple[np.ndarray, pd.DataFrame]:
    tempo = np.asarray(tempo, dtype=np.float64).reshape(-1)
    n = int(tempo.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32), pd.DataFrame()
    series = pd.Series(tempo).interpolate("linear", limit_direction="both")
    filled = series.to_numpy(dtype=np.float64)
    avg = float(np.nanmean(filled))
    if not np.isfinite(avg) or avg <= 0:
        avg = 1.0
    normalized = filled / avg

    beat_prob = np.ones(n, dtype=np.float64)
    rows = []
    for level in levels:
        level = int(level)
        if level <= 0:
            raise ValueError(f"Invalid level/window length: {level}")
        starts = list(range(0, n, level))
        e_values = []
        windows = []
        for window_idx, start in enumerate(starts):
            end = min(start + level, n)
            if end <= start:
                continue
            e = max(window_e(normalized[start:end], mode=e_mode), 1e-6)
            e_values.append(e)
            windows.append((window_idx, start, end, end - start))
        level_probs = safe_prob(np.asarray(e_values, dtype=np.float64))
        for (window_idx, start, end, length), e, prob in zip(windows, e_values, level_probs):
            beat_prob[start:end] *= float(prob)
            rows.append(
                {
                    "level_window_beats": level,
                    "window_idx": int(window_idx),
                    "start_beat": int(start),
                    "end_beat_exclusive": int(end),
                    "actual_window_beats": int(length),
                    "e_value": float(e),
                    "window_probability": float(prob),
                }
            )

    if normalize_output == "sum":
        total = float(np.sum(beat_prob))
        if total > 0:
            beat_prob = beat_prob / total
    elif normalize_output == "max":
        maxv = float(np.max(beat_prob))
        if maxv > 0:
            beat_prob = beat_prob / maxv
    elif normalize_output == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalize_output: {normalize_output}")

    return beat_prob.astype(np.float32), pd.DataFrame(rows)


def load_named_tempo_sources() -> dict[str, tuple[np.ndarray, Path]]:
    sources: dict[str, tuple[np.ndarray, Path]] = {}
    k331 = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "vienna_atepp_k331_tempo_compare" / "k331_score_aligned_mean_tempo.csv"
    if k331.exists():
        df = pd.read_csv(k331)
        sources["k331_vienna_mean"] = (df["vienna_mean_bpm"].to_numpy(dtype=np.float32), k331)
        sources["k331_atepp_mean"] = (df["atepp_score_aligned_mean_bpm"].to_numpy(dtype=np.float32), k331)
    op38 = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "chopin_op38_atepp_vienna_score_align" / "chopin_op38_vienna_atepp_aligned_mean_tempo.csv"
    if op38.exists():
        df = pd.read_csv(op38)
        sources["op38_vienna_mean"] = (df["vienna_mean_bpm"].to_numpy(dtype=np.float32), op38)
        sources["op38_atepp_mean"] = (df["atepp_aligned_mean_bpm"].to_numpy(dtype=np.float32), op38)
    return sources


def plot_probability(name: str, tempo: np.ndarray, probability: np.ndarray, out_dir: Path, top_k: int = 20) -> Path:
    x = np.arange(len(tempo), dtype=np.int32)
    top = np.argsort(-probability)[: min(int(top_k), len(probability))]
    top = np.asarray(sorted(top.tolist()), dtype=np.int32)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.9]})
    axes[0].plot(x, tempo, color="#1f4e8c", linewidth=2.0)
    axes[0].scatter(top, tempo[top], color="#d62728", s=24, label=f"top {len(top)} probability beats")
    axes[0].set_title(f"{name}: tempo curve with multiscale probability peaks")
    axes[0].set_ylabel("Tempo BPM")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].plot(x, probability, color="#222222", linewidth=1.8)
    axes[1].scatter(top, probability[top], color="#d62728", s=24)
    axes[1].set_ylabel("Boundary probability")
    axes[1].set_xlabel("Beat index")
    axes[1].grid(alpha=0.22)
    fig.tight_layout()
    path = out_dir / f"{name}_multiscale_probability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate beat-level multiscale tempo-probability boundary labels.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--levels", default="1,3,6,12,24")
    parser.add_argument("--e_mode", choices=["mean", "min", "rms_minus_std"], default="mean")
    parser.add_argument("--normalize_output", choices=["max", "sum", "none"], default="max")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(x.strip()) for x in str(args.levels).split(",") if x.strip())
    sources = load_named_tempo_sources()
    if not sources:
        raise FileNotFoundError("No known tempo mean CSVs found for K331 or Op38.")

    summary_rows = []
    for name, (tempo, source_path) in sources.items():
        probability, window_df = multiscale_boundary_probability(
            tempo,
            levels=levels,
            e_mode=str(args.e_mode),
            normalize_output=str(args.normalize_output),
        )
        label_df = pd.DataFrame(
            {
                "beat_idx": np.arange(len(probability), dtype=np.int32),
                "tempo_bpm": tempo,
                "boundary_probability": probability,
                "rank": pd.Series(probability).rank(method="first", ascending=False).astype(int).to_numpy(),
                "top20": (pd.Series(probability).rank(method="first", ascending=False) <= int(args.top_k)).astype(np.int8).to_numpy(),
            }
        )
        label_path = out_dir / f"{name}_multiscale_probability_labels.csv"
        window_path = out_dir / f"{name}_window_probabilities.csv"
        label_df.to_csv(label_path, index=False)
        window_df.to_csv(window_path, index=False)
        fig_path = plot_probability(name, tempo, probability, out_dir=out_dir, top_k=int(args.top_k))
        top_beats = label_df.nsmallest(int(args.top_k), "rank")["beat_idx"].tolist()
        summary_rows.append(
            {
                "source": name,
                "source_path": str(source_path),
                "beats": int(len(probability)),
                "levels": ",".join(str(x) for x in levels),
                "e_mode": str(args.e_mode),
                "normalize_output": str(args.normalize_output),
                "prob_min": float(np.min(probability)),
                "prob_max": float(np.max(probability)),
                "prob_mean": float(np.mean(probability)),
                "top_beats": " ".join(str(int(x)) for x in top_beats),
                "label_csv": str(label_path),
                "window_csv": str(window_path),
                "plot": str(fig_path),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "multiscale_probability_label_summary.csv", index=False)
    print(summary[["source", "beats", "levels", "e_mode", "top_beats"]].to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
