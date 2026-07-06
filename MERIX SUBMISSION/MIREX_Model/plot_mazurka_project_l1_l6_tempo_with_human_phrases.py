from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "build_mazurka_beat_npz_performer_levels.py"
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurka_project_l1_l6_tempo_human_phrase_plots"
STR_VEC = [3, 2, 2, 2, 2, 2]
TARGET_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}

spec = importlib.util.spec_from_file_location("mazurka_level_builder", BUILD_SCRIPT)
builder = importlib.util.module_from_spec(spec)
sys.modules["mazurka_level_builder_for_xls_plot"] = builder
assert spec.loader is not None
spec.loader.exec_module(builder)


def normalize_piece_id(path: Path) -> str:
    m = re.search(r"mazurka(\d+)-(\d+)", path.stem, flags=re.I)
    if not m:
        return path.stem
    return f"M{int(m.group(1)):02d}-{int(m.group(2))}"


def is_phrase_label(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    low = text.lower()
    skip = {
        "nan",
        ".",
        "expressions/notes",
        "measure",
        "beat",
        "non-event",
        "minrhy",
    }
    if low in skip:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return False
    # Keep structural/phrase labels and section names, not expression markings.
    phrase_words = ("phrase", "section", "verse", "intro", "cod", "coda")
    if any(w in low for w in phrase_words):
        return True
    if re.fullmatch(r"[A-H](?:['’])?", text):
        return True
    return False


def load_summary(path: Path) -> dict:
    raw = pd.read_excel(path, sheet_name="Summary", header=None)
    header = raw.iloc[1].astype(str).str.strip().str.lower()
    measure_col = int(np.flatnonzero(header == "measure")[0])
    beat_col = int(np.flatnonzero(header == "beat")[0])
    non_event_matches = np.flatnonzero(header == "non-event")
    tempo_start = int(non_event_matches[0] + 1) if len(non_event_matches) else beat_col + 1
    phrase_col = 3

    data = raw.iloc[2:].reset_index(drop=True)
    measure = pd.to_numeric(data.iloc[:, measure_col], errors="coerce")
    beat = pd.to_numeric(data.iloc[:, beat_col], errors="coerce")
    valid = measure.notna() & beat.notna()
    data = data.loc[valid].reset_index(drop=True)

    tempo_df = data.iloc[:, tempo_start:].apply(pd.to_numeric, errors="coerce")
    tempo_df = tempo_df.dropna(axis=1, how="all")
    tempo_df = tempo_df.interpolate(axis=0, limit_direction="both").clip(lower=1, upper=600)
    curves = {
        str(raw.iloc[0, col_idx]) if not pd.isna(raw.iloc[0, col_idx]) else f"perf_{j+1:03d}": tempo_df.iloc[:, j].to_numpy(dtype=float)
        for j, col_idx in enumerate(tempo_df.columns)
    }
    mean_tempo = tempo_df.mean(axis=1).to_numpy(dtype=float)

    phrase_rows = []
    phrase_values = data.iloc[:, phrase_col] if phrase_col < data.shape[1] else pd.Series([], dtype=object)
    for row_idx, value in phrase_values.items():
        if is_phrase_label(value):
            phrase_rows.append({"beat_index": int(row_idx), "label": str(value).strip()})

    return {
        "piece_id": normalize_piece_id(path),
        "path": path,
        "mean_tempo": mean_tempo,
        "curves": curves,
        "phrase_rows": phrase_rows,
        "num_beats": len(mean_tempo),
    }


def consensus_level_probs(curves: dict[str, np.ndarray], n_beats: int) -> dict[int, np.ndarray]:
    counts = {level: np.zeros(n_beats, dtype=np.float32) for level in range(1, 7)}
    for curve in curves.values():
        if len(curve) != n_beats:
            continue
        _, level_sets = builder.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=True)
        for level in range(1, 7):
            idx = level_sets.get(level, np.array([], dtype=int))
            idx = idx[(idx >= 0) & (idx < n_beats)]
            counts[level][idx] += 1.0
    denom = max(len(curves), 1)
    return {level: counts[level] / denom for level in range(1, 7)}


def draw_piece(item: dict) -> list[dict]:
    piece_id = item["piece_id"]
    mean_tempo = item["mean_tempo"]
    curves = item["curves"]
    phrase_rows = item["phrase_rows"]
    n_beats = item["num_beats"]
    beats = np.arange(n_beats) + 1
    level_probs = consensus_level_probs(curves, n_beats)
    target = np.maximum.reduce(
        [TARGET_WEIGHTS[level] * level_probs[level] for level in sorted(TARGET_WEIGHTS)]
    )

    fig, (ax_tempo, ax_target, ax_marks) = plt.subplots(
        3,
        1,
        figsize=(16, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.05, 1.35, 2.35]},
    )

    ax_tempo.plot(beats, mean_tempo, color="#222222", linewidth=1.5, label="mean tempo curve")
    for phrase in phrase_rows:
        x = phrase["beat_index"] + 1
        ax_tempo.axvline(x, color="#d62728", linewidth=1.0, alpha=0.55)
    ax_tempo.set_title(f"{piece_id}: computed L1-L6 boundaries vs human phrase labels")
    ax_tempo.set_ylabel("BPM")
    ax_tempo.grid(True, axis="y", alpha=0.25)
    ax_tempo.legend(loc="upper right")

    ax_target.plot(beats, target, color="#6a3d9a", linewidth=1.45, label="weighted L2+ target")
    if phrase_rows:
        for phrase in phrase_rows:
            ax_target.axvline(phrase["beat_index"] + 1, color="#d62728", linewidth=0.9, alpha=0.45)
    ax_target.set_ylabel("target")
    ax_target.set_ylim(-0.02, max(1.02, float(np.nanmax(target)) + 0.05))
    ax_target.grid(True, axis="y", alpha=0.25)
    ax_target.legend(loc="upper right")

    colors = {
        1: "#4c78a8",
        2: "#f58518",
        3: "#54a24b",
        4: "#b279a2",
        5: "#e45756",
        6: "#72b7b2",
    }
    rows = []
    for level in range(1, 7):
        probs = level_probs[level]
        idx = np.flatnonzero(probs > 0)
        rows.append(
            {
                "piece_id": piece_id,
                "level": level,
                "boundary_count_any_performer": int(idx.size),
                "boundary_count_ge_0p10": int(np.count_nonzero(probs >= 0.10)),
                "boundary_count_ge_0p50": int(np.count_nonzero(probs >= 0.50)),
            }
        )
        if idx.size:
            ax_marks.scatter(
                idx + 1,
                np.full_like(idx, 7 - level, dtype=float),
                s=8 + 44 * probs[idx],
                color=colors[level],
                alpha=0.82,
                edgecolors="none",
                label=f"L{level}",
            )

    if phrase_rows:
        x = np.array([p["beat_index"] + 1 for p in phrase_rows], dtype=int)
        ax_marks.scatter(
            x,
            np.zeros_like(x, dtype=float),
            s=90,
            marker="|",
            linewidths=2.4,
            color="#d62728",
            label="human phrase labels",
        )
        for p in phrase_rows:
            ax_marks.text(
                p["beat_index"] + 1,
                -0.33,
                p["label"],
                rotation=65,
                fontsize=7,
                ha="right",
                va="top",
                color="#8c1d18",
            )

    ax_marks.set_yticks([6, 5, 4, 3, 2, 1, 0])
    ax_marks.set_yticklabels(["L1", "L2", "L3", "L4", "L5", "L6", "human"])
    ax_marks.set_ylim(-1.2, 6.8)
    ax_marks.set_xlabel("score beat")
    ax_marks.grid(True, axis="x", alpha=0.16)
    ax_marks.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()

    png_path = OUT_DIR / f"{piece_id}_tempo_L1-L6_human_phrases.png"
    pdf_path = OUT_DIR / f"{piece_id}_tempo_L1-L6_human_phrases.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    phrase_csv = OUT_DIR / f"{piece_id}_human_phrase_boundaries.csv"
    pd.DataFrame(phrase_rows).to_csv(phrase_csv, index=False)
    target_csv = OUT_DIR / f"{piece_id}_beat_target_values.csv"
    target_frame = pd.DataFrame(
        {
            "beat": beats,
            "target_weighted_l2plus": target,
            **{f"consensus_L{level}": level_probs[level] for level in range(1, 7)},
            **{
                f"weighted_L{level}": TARGET_WEIGHTS[level] * level_probs[level]
                for level in sorted(TARGET_WEIGHTS)
            },
            "human_phrase_label": "",
        }
    )
    for phrase in phrase_rows:
        target_frame.loc[target_frame["beat"] == phrase["beat_index"] + 1, "human_phrase_label"] = phrase["label"]
    target_frame.to_csv(target_csv, index=False)

    return [
        {
            **row,
            "num_beats": n_beats,
            "num_performers": len(curves),
            "human_phrase_count": len(phrase_rows),
            "target_max": float(np.nanmax(target)) if target.size else 0.0,
            "target_sum": float(np.nansum(target)),
            "target_ge_0p05": int(np.count_nonzero(target >= 0.05)),
            "png": str(png_path),
            "pdf": str(pdf_path),
            "target_csv": str(target_csv),
        }
        for row in rows
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    plot_rows = []
    for path in sorted(DATA_DIR.glob("mazurka*.xls")):
        item = load_summary(path)
        rows = draw_piece(item)
        all_rows.extend(rows)
        plot_rows.append(
            {
                "piece_id": item["piece_id"],
                "xls": str(path),
                "num_beats": item["num_beats"],
                "num_performers": len(item["curves"]),
                "human_phrase_count": len(item["phrase_rows"]),
                "target_max": rows[0]["target_max"] if rows else 0.0,
                "target_sum": rows[0]["target_sum"] if rows else 0.0,
                "target_ge_0p05": rows[0]["target_ge_0p05"] if rows else 0,
                "human_phrase_labels": " | ".join(f"{p['beat_index'] + 1}:{p['label']}" for p in item["phrase_rows"]),
                "png": rows[0]["png"] if rows else "",
                "pdf": rows[0]["pdf"] if rows else "",
                "target_csv": rows[0]["target_csv"] if rows else "",
            }
        )
    pd.DataFrame(plot_rows).to_csv(OUT_DIR / "plot_summary.csv", index=False)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "level_boundary_count_summary.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "source": "datasets/mazurka*.xls Summary sheets",
                "human_phrase_rule": "non-empty structural labels in Summary column 4, e.g. Intro, A phrase, B phrase, A, A', section, verse",
                "tempo_curve": "mean of performer tempo columns in Summary sheet",
                "computed_boundaries": "L1-L6 from group_analysis_hierarchy with STR_VEC=[1,2,4,8,16,32]; marker size is performer consensus frequency",
                "target_rule": "target(b)=max_{L2..L6}(weight_L * consensus_L(b)), weights={2:0.28,3:0.46,4:0.64,5:0.82,6:1.0}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR}")
    print(
        pd.DataFrame(plot_rows)[
            [
                "piece_id",
                "num_beats",
                "num_performers",
                "human_phrase_count",
                "target_max",
                "target_sum",
                "target_ge_0p05",
                "human_phrase_labels",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
