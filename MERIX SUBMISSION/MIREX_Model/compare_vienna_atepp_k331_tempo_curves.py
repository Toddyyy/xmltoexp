from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "vienna_atepp_k331_tempo_compare"
VIENNA_MATCH_DIR = ROOT / "datasets" / "Vienna4x4" / "vienna4x22_rematched-master" / "match"
ATEPP_DIR = (
    ROOT
    / "ATEPP-1.2"
    / "ATEPP-1.2"
    / "Wolfgang_Amadeus_Mozart"
    / "Piano_Sonata_No._11_in_A_Major,_K._331"
    / "1._Tema_(Andante_grazioso)_con_variazioni"
)

sys.path.insert(0, str(ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto"))
from run_atepp_auto_meter_crf_transfer import tempo_curve_from_match  # noqa: E402


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    buf = []
    depth = 0
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def parse_info_number(text: str, key: str, default: float) -> float:
    match = re.search(rf"info\({re.escape(key)},([0-9.]+)\)\.", text)
    return float(match.group(1)) if match else float(default)


def vienna_tempo_curve(match_path: Path, num_beats: int = 216, beat_unit: float = 0.5) -> np.ndarray | None:
    text = match_path.read_text(encoding="utf-8", errors="replace")
    clock_units = parse_info_number(text, "midiClockUnits", 4000.0)
    clock_rate_us = parse_info_number(text, "midiClockRate", 500000.0)
    tick_to_sec = clock_rate_us / 1_000_000.0 / clock_units
    rows = []
    for line in text.splitlines():
        if not line.startswith("snote(") or ")-note(" not in line:
            continue
        match = re.match(r"snote\((.*)\)-note\((.*)\)\.", line.strip())
        if not match:
            continue
        snote_parts = split_top_level_commas(match.group(1))
        note_parts = split_top_level_commas(match.group(2))
        if len(snote_parts) < 9 or len(note_parts) < 4:
            continue
        try:
            score_onset = float(snote_parts[-3])
            onset_tick = float(note_parts[3])
        except ValueError:
            continue
        beat_idx = int(np.floor(score_onset))
        if 0 <= beat_idx < int(num_beats):
            rows.append((beat_idx, onset_tick * tick_to_sec))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["beat_idx", "onset"])
    full_idx = pd.Index(range(int(num_beats)), name="beat_idx")
    beat_time = df.groupby("beat_idx")["onset"].median().sort_index()
    beat_time = beat_time.reindex(full_idx).interpolate("linear", limit_direction="both")
    dt = beat_time.diff()
    tempo = (60.0 * float(beat_unit)) / dt
    tempo = tempo[(tempo > 0.0) & (tempo < 400.0)]
    tempo = tempo.rolling(window=3, center=True, min_periods=1).mean()
    tempo = tempo.clip(upper=400.0)
    tempo = tempo.reindex(full_idx).interpolate("linear", limit_direction="both")
    return tempo.to_numpy(dtype=np.float32)


def load_vienna_curves() -> dict[str, np.ndarray]:
    curves = {}
    for path in sorted(VIENNA_MATCH_DIR.glob("Mozart_K331_1st-mov_p*.match")):
        curve = vienna_tempo_curve(path)
        if curve is not None and np.isfinite(curve).any():
            curves[path.stem.replace("Mozart_K331_1st-mov_", "")] = curve
    return curves


def load_atepp_curves() -> dict[str, np.ndarray]:
    curves = {}
    fmt3x = ATEPP_DIR / "score_fmt3x.txt"
    for path in sorted(ATEPP_DIR.glob("*_match.txt")):
        if not path.stem.replace("_match", "").isdigit():
            continue
        curve = tempo_curve_from_match(
            match_path=path,
            fmt3x_path=fmt3x,
            num_beats=947,
            beat_unit=0.5,
            smooth_window=3,
            bpm_max=400.0,
        )
        if curve is not None and np.isfinite(curve).any():
            curves[path.stem.replace("_match", "")] = curve
    return curves


def stack_mean(curves: dict[str, np.ndarray], length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = list(curves.values())
    if length is not None:
        values = [v[:length] for v in values if len(v) >= length]
    arr = np.vstack(values)
    return arr, np.nanmean(arr, axis=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vienna = load_vienna_curves()
    atepp = load_atepp_curves()
    if not vienna or not atepp:
        raise RuntimeError(f"Missing curves: vienna={len(vienna)} atepp={len(atepp)}")

    # Full ATEPP movement is longer; first 216 score beats correspond to the Vienna excerpt length.
    v_arr, v_mean = stack_mean(vienna, length=216)
    a_full_arr, a_full_mean = stack_mean(atepp)
    a_clip_arr, a_clip_mean = stack_mean(atepp, length=216)

    pd.DataFrame({"beat_idx": np.arange(216), "vienna_mean_bpm": v_mean, "atepp_first216_mean_bpm": a_clip_mean}).to_csv(
        OUT_DIR / "k331_mean_tempo_first216.csv", index=False
    )
    pd.DataFrame({"beat_idx": np.arange(len(a_full_mean)), "atepp_full_mean_bpm": a_full_mean}).to_csv(
        OUT_DIR / "k331_atepp_full_mean_tempo.csv", index=False
    )

    theme_stats = {
        "mode": "theme_only_first216_beats",
        "vienna_curves": len(vienna),
        "atepp_curves": len(atepp),
        "compared_beats": int(v_arr.shape[1]),
        "removed_atepp_tail_beats": int(a_full_arr.shape[1] - v_arr.shape[1]),
        "mean_abs_diff_bpm": float(np.nanmean(np.abs(v_mean - a_clip_mean))),
        "rmse_bpm": float(np.sqrt(np.nanmean((v_mean - a_clip_mean) ** 2))),
        "corr": float(np.corrcoef(v_mean, a_clip_mean)[0, 1]),
        "vienna_mean_bpm": float(np.nanmean(v_mean)),
        "atepp_theme_mean_bpm": float(np.nanmean(a_clip_mean)),
    }
    pd.DataFrame([theme_stats]).to_csv(OUT_DIR / "k331_theme_only_derepeated_stats.csv", index=False)

    fig_theme, ax = plt.subplots(1, 1, figsize=(13, 4.8))
    x = np.arange(216)
    for curve in v_arr:
        ax.plot(x, curve, color="#4C78A8", alpha=0.16, linewidth=0.7)
    for curve in a_clip_arr:
        ax.plot(x, curve, color="#F58518", alpha=0.22, linewidth=0.8)
    ax.plot(x, v_mean, color="#1f4e8c", linewidth=2.4, label=f"Vienna4x4 mean (n={len(vienna)})")
    ax.plot(x, a_clip_mean, color="#c45a00", linewidth=2.4, label=f"ATEPP theme only mean (n={len(atepp)})")
    for beat in [48, 108, 156]:
        ax.axvline(beat, color="#444444", alpha=0.16, linewidth=1.0)
    ax.set_title("Mozart K331 tempo curves after removing non-corresponding ATEPP tail")
    ax.set_xlabel("Score beat index in comparable theme section")
    ax.set_ylabel("Tempo (quarter-note BPM)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig_theme.tight_layout()
    fig_theme_path = OUT_DIR / "k331_vienna_vs_atepp_theme_only_derepeated.png"
    fig_theme.savefig(fig_theme_path, dpi=180)
    plt.close(fig_theme)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [1.15, 1.0]}, sharex=False)
    x = np.arange(216)
    for curve in v_arr:
        axes[0].plot(x, curve, color="#4C78A8", alpha=0.18, linewidth=0.7)
    for curve in a_clip_arr:
        axes[0].plot(x, curve, color="#F58518", alpha=0.25, linewidth=0.8)
    axes[0].plot(x, v_mean, color="#1f4e8c", linewidth=2.2, label=f"Vienna4x4 mean (n={len(vienna)})")
    axes[0].plot(x, a_clip_mean, color="#c45a00", linewidth=2.2, label=f"ATEPP mean first 216 beats (n={len(atepp)})")
    axes[0].set_title("Mozart K331 first movement tempo curves: Vienna4x4 vs ATEPP")
    axes[0].set_ylabel("Tempo (quarter-note BPM)")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    xf = np.arange(len(a_full_mean))
    for curve in a_full_arr:
        axes[1].plot(xf, curve, color="#F58518", alpha=0.16, linewidth=0.7)
    axes[1].plot(xf, a_full_mean, color="#c45a00", linewidth=2.0, label="ATEPP full mean")
    axes[1].axvspan(0, 215, color="#4C78A8", alpha=0.12, label="Vienna excerpt length")
    axes[1].set_xlabel("Score beat index")
    axes[1].set_ylabel("Tempo (quarter-note BPM)")
    axes[1].set_title("ATEPP full first movement is longer than the Vienna4x4 excerpt")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig_path = OUT_DIR / "k331_vienna_vs_atepp_tempo_curves.png"
    fig.savefig(fig_path, dpi=180)

    stats = {
        "vienna_curves": len(vienna),
        "atepp_curves": len(atepp),
        "vienna_beats": int(v_arr.shape[1]),
        "atepp_full_beats": int(a_full_arr.shape[1]),
        "first216_mean_abs_diff_bpm": float(np.nanmean(np.abs(v_mean - a_clip_mean))),
        "first216_corr": float(np.corrcoef(v_mean, a_clip_mean)[0, 1]),
        "vienna_mean_bpm": float(np.nanmean(v_mean)),
        "atepp_first216_mean_bpm": float(np.nanmean(a_clip_mean)),
        "atepp_full_mean_bpm": float(np.nanmean(a_full_mean)),
    }
    pd.DataFrame([stats]).to_csv(OUT_DIR / "k331_compare_stats.csv", index=False)
    print(fig_theme_path)
    print(fig_path)
    print(pd.Series(theme_stats).to_string())
    print(pd.Series(stats).to_string())


if __name__ == "__main__":
    main()
