from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import music21 as m21
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "chopin_op38_atepp_vienna_score_align"
WORK_DIR = OUT_DIR / "work"
VIENNA_MATCH_DIR = ROOT / "datasets" / "Vienna4x4" / "vienna4x22_rematched-master" / "match"
VIENNA_SCORE = ROOT / "datasets" / "Vienna4x4" / "vienna4x22_rematched-master" / "musicxml" / "Chopin_op38.musicxml"

NUM_BEATS = 274
BEAT_UNIT = 0.5
SMOOTH_WINDOW = 3
BPM_MAX = 600.0
MIN_FRAGMENT_BEATS = 12
MIN_NONEMPTY_ONSETS = 4


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


def vienna_tempo_curve(match_path: Path) -> np.ndarray | None:
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
        if 0 <= beat_idx < NUM_BEATS:
            rows.append((beat_idx, onset_tick * tick_to_sec))
    return beat_time_rows_to_tempo(rows)


def read_tpqn(fmt3x_path: Path, default: float = 24.0) -> float:
    text = fmt3x_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"TPQN:\s*(\d+)", text)
    return float(match.group(1)) if match else default


def atepp_tempo_curve(match_path: Path, fmt3x_path: Path) -> np.ndarray | None:
    tpqn = read_tpqn(fmt3x_path)
    rows = []
    for line in match_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("//") or line.startswith("Missing"):
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 12:
            continue
        try:
            onset = float(parts[1])
            score_time = float(parts[8])
        except ValueError:
            continue
        if parts[9] == "*" or parts[10] != "0":
            continue
        score_quarter = score_time / float(tpqn)
        beat_idx = int(np.floor(score_quarter / BEAT_UNIT))
        if 0 <= beat_idx < NUM_BEATS:
            rows.append((beat_idx, onset))
    return beat_time_rows_to_tempo(rows)


def beat_time_rows_to_tempo(rows: list[tuple[int, float]]) -> np.ndarray | None:
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["beat_idx", "onset"])
    full_idx = pd.Index(range(NUM_BEATS), name="beat_idx")
    beat_time = df.groupby("beat_idx")["onset"].median().sort_index()
    beat_time = beat_time.reindex(full_idx).interpolate("linear", limit_direction="both")
    dt = beat_time.diff()
    tempo = (60.0 * BEAT_UNIT) / dt
    tempo = tempo[(tempo > 0.0) & (tempo < BPM_MAX)]
    tempo = tempo.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
    tempo = tempo.clip(upper=BPM_MAX)
    tempo = tempo.reindex(full_idx).interpolate("linear", limit_direction="both")
    return tempo.to_numpy(dtype=np.float32)


def load_vienna_curves() -> dict[str, np.ndarray]:
    curves = {}
    for path in sorted(VIENNA_MATCH_DIR.glob("Chopin_op38_p*.match")):
        curve = vienna_tempo_curve(path)
        if curve is not None and np.isfinite(curve).any():
            curves[path.stem.replace("Chopin_op38_", "")] = curve
    return curves


def load_atepp_curves() -> dict[str, np.ndarray]:
    curves = {}
    fmt3x = WORK_DIR / "score_fmt3x.txt"
    for path in sorted(WORK_DIR.glob("*_match.txt")):
        if not path.stem.replace("_match", "").isdigit():
            continue
        curve = atepp_tempo_curve(path, fmt3x)
        if curve is not None and np.isfinite(curve).any():
            curves[path.stem.replace("_match", "")] = curve
    return curves


def score_tokens(path: Path, grid_per_quarter: int = 2) -> list[tuple[int, ...]]:
    score = m21.converter.parse(str(path))
    n = int(round(float(score.highestTime) * grid_per_quarter))
    by_grid: dict[int, set[int]] = {}
    for part in score.parts:
        for note in part.recurse().notes:
            idx = int(round(float(note.getOffsetInHierarchy(score)) * grid_per_quarter))
            pitches = note.pitches if note.isChord else [note.pitch]
            by_grid.setdefault(idx, set()).update(int(p.midi) for p in pitches)
    return [tuple(sorted(by_grid.get(i, set()))) for i in range(n)]


def find_self_common_fragments(tokens: list[tuple[int, ...]]) -> pd.DataFrame:
    rows = []
    for v_start in range(len(tokens)):
        for a_start in range(v_start + 1, len(tokens)):
            length = 0
            nonempty = 0
            while (
                v_start + length < len(tokens)
                and a_start + length < len(tokens)
                and tokens[v_start + length] == tokens[a_start + length]
            ):
                if tokens[v_start + length]:
                    nonempty += 1
                length += 1
            if length >= MIN_FRAGMENT_BEATS and nonempty >= MIN_NONEMPTY_ONSETS:
                if v_start > 0 and a_start > 0 and tokens[v_start - 1] == tokens[a_start - 1]:
                    continue
                rows.append(
                    {
                        "fragment_id": f"F{len(rows) + 1:02d}",
                        "vienna_start_beat": v_start,
                        "vienna_end_beat_exclusive": v_start + length,
                        "atepp_start_beat": a_start,
                        "atepp_end_beat_exclusive": a_start + length,
                        "length_beats": length,
                        "nonempty_onset_beats": nonempty,
                        "start_measure_approx": int(v_start // 6 + 1),
                        "end_measure_approx": int((v_start + length - 1) // 6 + 1),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["length_beats", "nonempty_onset_beats"], ascending=False).reset_index(drop=True)


def make_curve_table(matches: pd.DataFrame, vienna_mean: np.ndarray, atepp_mean: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, match in matches.iterrows():
        v0 = int(match["vienna_start_beat"])
        a0 = int(match["atepp_start_beat"])
        length = int(match["length_beats"])
        if v0 + length > NUM_BEATS or a0 + length > NUM_BEATS:
            continue
        for local_idx in range(length):
            rows.append(
                {
                    "fragment_id": match["fragment_id"],
                    "local_beat_idx": local_idx,
                    "vienna_beat_idx": v0 + local_idx,
                    "atepp_beat_idx": a0 + local_idx,
                    "vienna_mean_bpm": float(vienna_mean[v0 + local_idx]),
                    "atepp_mean_bpm": float(atepp_mean[a0 + local_idx]),
                }
            )
    return pd.DataFrame(rows)


def add_stats(matches: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fragment_id, group in curves.groupby("fragment_id", sort=False):
        v = group["vienna_mean_bpm"].to_numpy(dtype=np.float32)
        a = group["atepp_mean_bpm"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "fragment_id": fragment_id,
                "tempo_corr": float(np.corrcoef(v, a)[0, 1]) if len(v) > 1 else np.nan,
                "tempo_mae_bpm": float(np.mean(np.abs(v - a))),
                "vienna_mean_bpm": float(np.mean(v)),
                "atepp_mean_bpm": float(np.mean(a)),
            }
        )
    return matches.merge(pd.DataFrame(rows), on="fragment_id", how="left")


def plot_top(matches: pd.DataFrame, curves: pd.DataFrame, top_n: int = 6) -> Path:
    top = matches.head(top_n)["fragment_id"].tolist()
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.45 * len(top)), sharex=False)
    if len(top) == 1:
        axes = [axes]
    for ax, fragment_id in zip(axes, top):
        meta = matches[matches["fragment_id"] == fragment_id].iloc[0]
        group = curves[curves["fragment_id"] == fragment_id].sort_values("local_beat_idx")
        ax.plot(group["local_beat_idx"], group["vienna_mean_bpm"], color="#1f4e8c", linewidth=2.1, label="Vienna mean")
        ax.plot(group["local_beat_idx"], group["atepp_mean_bpm"], color="#c45a00", linewidth=2.1, label="ATEPP aligned mean")
        ax.set_title(
            f"{fragment_id}: score beats {int(meta.vienna_start_beat)}-{int(meta.vienna_end_beat_exclusive)} "
            f"({int(meta.length_beats)} beats, corr {float(meta.tempo_corr):.2f}, MAE {float(meta.tempo_mae_bpm):.1f} BPM)"
        )
        ax.set_ylabel("BPM")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Local beat index inside exact note/onset fragment")
    fig.tight_layout()
    path = OUT_DIR / "chopin_op38_common_score_fragments_tempo_compare_top6.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vienna = load_vienna_curves()
    atepp = load_atepp_curves()
    if not vienna or not atepp:
        raise RuntimeError(f"Missing curves: vienna={len(vienna)} atepp={len(atepp)}")
    v_arr = np.vstack(list(vienna.values()))
    a_arr = np.vstack(list(atepp.values()))
    v_mean = np.nanmean(v_arr, axis=0)
    a_mean = np.nanmean(a_arr, axis=0)
    pd.DataFrame(
        {
            "beat_idx": np.arange(NUM_BEATS, dtype=np.int32),
            "vienna_mean_bpm": v_mean,
            "atepp_aligned_mean_bpm": a_mean,
        }
    ).to_csv(OUT_DIR / "chopin_op38_vienna_atepp_aligned_mean_tempo.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["Vienna", "ATEPP"],
            "curves": [len(vienna), len(atepp)],
            "mean_bpm": [float(np.nanmean(v_mean)), float(np.nanmean(a_mean))],
        }
    ).to_csv(OUT_DIR / "chopin_op38_curve_inventory.csv", index=False)

    tokens = score_tokens(VIENNA_SCORE)
    matches = find_self_common_fragments(tokens)
    curves = make_curve_table(matches, v_mean, a_mean)
    matches = add_stats(matches, curves)
    matches.to_csv(OUT_DIR / "chopin_op38_common_exact_note_onset_fragments.csv", index=False)
    curves.to_csv(OUT_DIR / "chopin_op38_common_exact_note_onset_fragment_tempo_curves.csv", index=False)
    fig_path = plot_top(matches, curves, top_n=min(6, len(matches)))
    print(fig_path)
    print(pd.DataFrame({"dataset": ["Vienna", "ATEPP"], "curves": [len(vienna), len(atepp)]}).to_string(index=False))
    print(matches.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
