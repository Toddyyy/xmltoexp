from __future__ import annotations

from pathlib import Path

import matplotlib
import music21 as m21
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "vienna_atepp_k331_tempo_compare"
VIENNA_SCORE = ROOT / "datasets" / "Vienna4x4" / "vienna4x22_rematched-master" / "musicxml" / "Mozart_K331_1st-mov.musicxml"
ATEPP_SCORE = (
    ROOT
    / "ATEPP-1.2"
    / "ATEPP-1.2"
    / "Wolfgang_Amadeus_Mozart"
    / "Piano_Sonata_No._11_in_A_Major,_K._331"
    / "1._Tema_(Andante_grazioso)_con_variazioni"
    / "musicxml_cleaned.musicxml"
)
MEAN_TEMPO = OUT_DIR / "k331_score_aligned_mean_tempo.csv"
ATEPP_FULL_TEMPO = OUT_DIR / "k331_atepp_full_mean_tempo.csv"

MIN_FRAGMENT_BEATS = 12
MIN_NONEMPTY_ONSETS = 4


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


def find_exact_diagonal_runs(
    vienna: list[tuple[int, ...]],
    atepp: list[tuple[int, ...]],
) -> pd.DataFrame:
    rows = []
    for v_start in range(len(vienna)):
        for a_start in range(len(atepp)):
            if v_start > 0 and a_start > 0 and vienna[v_start - 1] == atepp[a_start - 1]:
                continue
            length = 0
            nonempty = 0
            while (
                v_start + length < len(vienna)
                and a_start + length < len(atepp)
                and vienna[v_start + length] == atepp[a_start + length]
            ):
                if vienna[v_start + length]:
                    nonempty += 1
                length += 1
            if length >= MIN_FRAGMENT_BEATS and nonempty >= MIN_NONEMPTY_ONSETS:
                rows.append(
                    {
                        "fragment_id": f"F{len(rows) + 1:02d}",
                        "vienna_start_beat": v_start,
                        "vienna_end_beat_exclusive": v_start + length,
                        "atepp_start_beat": a_start,
                        "atepp_end_beat_exclusive": a_start + length,
                        "length_beats": length,
                        "nonempty_onset_beats": nonempty,
                        "vienna_start_measure_approx": int(v_start // 6 + 1),
                        "vienna_end_measure_approx": int((v_start + length - 1) // 6 + 1),
                        "atepp_start_measure_approx": int(a_start // 6 + 1),
                        "atepp_end_measure_approx": int((a_start + length - 1) // 6 + 1),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["length_beats", "nonempty_onset_beats"], ascending=False).reset_index(drop=True)


def load_tempo() -> tuple[np.ndarray, np.ndarray]:
    aligned = pd.read_csv(MEAN_TEMPO)
    atepp_full = pd.read_csv(ATEPP_FULL_TEMPO)
    return (
        aligned["vienna_mean_bpm"].to_numpy(dtype=np.float32),
        atepp_full["atepp_full_mean_bpm"].to_numpy(dtype=np.float32),
    )


def make_curve_table(matches: pd.DataFrame, vienna_tempo: np.ndarray, atepp_tempo: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, match in matches.iterrows():
        v0 = int(match["vienna_start_beat"])
        a0 = int(match["atepp_start_beat"])
        length = int(match["length_beats"])
        if v0 + length > len(vienna_tempo) or a0 + length > len(atepp_tempo):
            continue
        for local_idx in range(length):
            rows.append(
                {
                    "fragment_id": match["fragment_id"],
                    "local_beat_idx": local_idx,
                    "vienna_beat_idx": v0 + local_idx,
                    "atepp_beat_idx": a0 + local_idx,
                    "vienna_mean_bpm": float(vienna_tempo[v0 + local_idx]),
                    "atepp_mean_bpm": float(atepp_tempo[a0 + local_idx]),
                }
            )
    return pd.DataFrame(rows)


def add_tempo_stats(matches: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
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


def plot_top_fragments(matches: pd.DataFrame, curves: pd.DataFrame, top_n: int = 6) -> Path:
    top = matches.head(top_n)["fragment_id"].tolist()
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.4 * len(top)), sharex=False)
    if len(top) == 1:
        axes = [axes]
    for ax, fragment_id in zip(axes, top):
        meta = matches[matches["fragment_id"] == fragment_id].iloc[0]
        group = curves[curves["fragment_id"] == fragment_id].sort_values("local_beat_idx")
        ax.plot(group["local_beat_idx"], group["vienna_mean_bpm"], color="#1f4e8c", linewidth=2.1, label="Vienna mean")
        ax.plot(group["local_beat_idx"], group["atepp_mean_bpm"], color="#c45a00", linewidth=2.1, label="ATEPP mean")
        ax.set_title(
            f"{fragment_id}: Vienna beats {int(meta.vienna_start_beat)}-{int(meta.vienna_end_beat_exclusive)} "
            f"vs ATEPP beats {int(meta.atepp_start_beat)}-{int(meta.atepp_end_beat_exclusive)} "
            f"({int(meta.length_beats)} beats, MAE {float(meta.tempo_mae_bpm):.1f} BPM)"
        )
        ax.set_ylabel("BPM")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Local beat index in exact-matching note/onset fragment")
    fig.tight_layout()
    path = OUT_DIR / "k331_common_score_fragments_tempo_compare_top6.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vienna_tokens = score_tokens(VIENNA_SCORE)
    atepp_tokens = score_tokens(ATEPP_SCORE)
    matches = find_exact_diagonal_runs(vienna_tokens, atepp_tokens)
    vienna_tempo, atepp_tempo = load_tempo()
    curves = make_curve_table(matches, vienna_tempo, atepp_tempo)
    matches = add_tempo_stats(matches, curves)
    matches.to_csv(OUT_DIR / "k331_common_exact_note_onset_fragments.csv", index=False)
    curves.to_csv(OUT_DIR / "k331_common_exact_note_onset_fragment_tempo_curves.csv", index=False)
    fig_path = plot_top_fragments(matches, curves, top_n=min(6, len(matches)))
    print(fig_path)
    print(matches.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
