from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app.py"
OUTPUT_DIR = SCRIPT_DIR / "exports"
OUTPUT_PATH = OUTPUT_DIR / "weighted_topdown_seed44_merge56_clean_outer.pdf"


def load_app_module():
    spec = importlib.util.spec_from_file_location("webviz_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load app module from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_metric(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_overall_summary(app, groups: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for group in groups:
        summary = app.load_weighted_group_summary(group, seed)
        metrics = summary.get("union_metrics", {})
        rows.append(
            {
                "group": group,
                "events": int(metrics.get("pred_events", 0)),
                "threshold": format_metric(summary.get("frozen_threshold", metrics.get("threshold"))),
                "precision": format_metric(metrics.get("union_precision")),
                "freq_precision": format_metric(metrics.get("frequency_weighted_precision")),
                "weighted_recall": format_metric(metrics.get("weighted_recall")),
                "consensus_recall": format_metric(metrics.get("consensus_recall")),
                "epochs": int(summary.get("frozen_epochs", 0)),
            }
        )
    return pd.DataFrame(rows)


def draw_table(ax, df: pd.DataFrame, title: str, fontsize: int = 9, scale_y: float = 1.35) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, scale_y)


def make_cover_page(pdf: PdfPages, app, groups: list[str], seed: int) -> None:
    summary_df = build_overall_summary(app, groups, seed)

    fig = plt.figure(figsize=(11.69, 8.27))
    ax_title = fig.add_axes([0.06, 0.80, 0.88, 0.14])
    ax_title.axis("off")
    ax_title.text(0.0, 0.90, "Weighted Topdown Seed44 PDF", fontsize=20, fontweight="bold", va="top")
    ax_title.text(
        0.0,
        0.58,
        "Merged L5+6 clean outer test",
        fontsize=13,
        va="top",
    )
    ax_title.text(
        0.0,
        0.26,
        "Outer test: M06-1, M06-2, M06-3 | Model: TCN direct | Train floor: 0.05 | Display: high-level priority with +/-1 beat suppression",
        fontsize=10,
        va="top",
    )

    ax_table = fig.add_axes([0.06, 0.08, 0.88, 0.66])
    draw_table(ax_table, summary_df, "Overall Summary", fontsize=10, scale_y=1.45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_piece_page(pdf: PdfPages, app, piece_id: str, groups: list[str], seed: int) -> None:
    raw = app.build_weighted_breakpoint_table(piece_id, groups, seed=seed)
    filtered = app.suppress_lower_breakpoints_within_tolerance(
        raw,
        priority_order=[group for group in app.WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in groups],
        tolerance=1,
    )
    summary = app.build_weighted_summary_table(piece_id, groups, filtered, seed=seed).copy()
    summary = summary[["group", "events", "threshold", "precision", "freq_precision", "weighted_recall", "consensus_recall", "frozen_epochs"]]
    summary.columns = ["group", "events", "threshold", "precision", "freq_prec", "weighted_rec", "cons_rec", "epochs"]

    beat_map = app.load_beat_map(piece_id)
    measure_starts = (
        beat_map.reset_index()
        .groupby("measure_number", as_index=False)
        .first()[["measure_number", "index"]]
        .rename(columns={"index": "beat_idx"})
    )

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"{piece_id} | Seed44 merged L5+6 clean outer", fontsize=16, fontweight="bold", y=0.98)

    ax_plot = fig.add_axes([0.06, 0.12, 0.56, 0.76])
    ax_plot.set_title("Displayed Breakpoints", fontsize=11, fontweight="bold", loc="left")
    y_positions = {group: idx for idx, group in enumerate(reversed(groups))}

    for _, row in measure_starts.iterrows():
        if int(row["measure_number"]) % 4 == 1:
            ax_plot.axvline(float(row["beat_idx"]), color="#d9d9d9", lw=0.6, zorder=0)

    for group in groups:
        group_df = filtered[filtered["group"] == group].copy()
        if group_df.empty:
            continue
        color = app.WEIGHTED_TOPDOWN_GROUP_SPECS[group]["color"]
        matched = group_df[group_df["matched_union"] == True]
        unmatched = group_df[group_df["matched_union"] != True]
        if not matched.empty:
            ax_plot.scatter(
                matched["beat_idx"],
                [y_positions[group]] * len(matched),
                s=40 + matched["detector_score"].fillna(0.0) * 110,
                c=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
        if not unmatched.empty:
            ax_plot.scatter(
                unmatched["beat_idx"],
                [y_positions[group]] * len(unmatched),
                s=34 + unmatched["detector_score"].fillna(0.0) * 100,
                c=color,
                marker="x",
                linewidths=1.1,
                zorder=4,
            )

    ax_plot.set_yticks([y_positions[group] for group in reversed(groups)])
    ax_plot.set_yticklabels(list(reversed(groups)))
    ax_plot.set_xlabel("Beat index")
    ax_plot.set_ylabel("Level")
    ax_plot.set_xlim(-2, max(int(beat_map.index.max()) + 2, 10))
    ax_plot.set_ylim(-0.8, len(groups) - 0.2)
    ax_plot.grid(axis="x", color="#efefef", linewidth=0.5)

    ax_table = fig.add_axes([0.66, 0.48, 0.30, 0.40])
    draw_table(ax_table, summary, "Summary", fontsize=8, scale_y=1.28)

    preview = filtered[["group", "measure", "beat_in_measure", "detector_score", "matched_union"]].copy()
    preview["detector_score"] = preview["detector_score"].map(lambda value: format_metric(value, 3))
    preview["matched_union"] = preview["matched_union"].map(lambda value: "Y" if bool(value) else "N")
    preview = preview.head(18)
    ax_preview = fig.add_axes([0.66, 0.12, 0.30, 0.26])
    draw_table(ax_preview, preview, "First Events", fontsize=8, scale_y=1.18)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    app = load_app_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = app.get_group_order("weighted_topdown_seed44")
    seed = 44

    with PdfPages(OUTPUT_PATH) as pdf:
        make_cover_page(pdf, app, groups, seed)
        for piece_id in app.WEIGHTED_TOPDOWN_PIECES:
            make_piece_page(pdf, app, piece_id, groups, seed)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
