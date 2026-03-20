from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ModuleNotFoundError:  # pragma: no cover - local utility fallback
    st = None
    components = None


def _cache_data_fallback(*_args, **_kwargs):
    def decorator(func):
        return func

    return decorator


CACHE_DATA = st.cache_data if st is not None else _cache_data_fallback


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "MazurkaBL-master").exists() and (candidate / "MERIX SUBMISSION").exists():
            return candidate
    raise FileNotFoundError("Unable to locate project root")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SUPPORTED_PIECES = ["M06-1", "M06-2", "M06-3", "M17-1", "M30-1"]
WEIGHTED_TOPDOWN_PIECES = ["M06-1", "M06-2", "M06-3"]
SEED_ORDER = [42, 44]
STRATEGY_ORDER = ["baseline", "consensus_guarded"]
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "consensus_guarded": "Consensus Guarded",
}
CLEAN_GROUP_SPECS = {
    "L1": {"slug": "level1", "color": "#0a6cff"},
    "L2": {"slug": "level2", "color": "#00a35c"},
    "L3": {"slug": "level3", "color": "#ff8a00"},
    "L4": {"slug": "level4", "color": "#7b1fa2"},
    "L5+6": {"slug": "level56", "color": "#c2185b"},
}
STRATEGY_GROUP_SPECS = {
    "L1": {"slug": "level1", "color": "#0a6cff"},
    "L2": {"slug": "level2", "color": "#00a35c"},
    "L3": {"slug": "level3", "color": "#ff8a00"},
    "L4": {"slug": "level4", "color": "#7b1fa2"},
    "L5": {"slug": "level5", "color": "#c2185b"},
    "L6": {"slug": "level6", "color": "#6d4c41"},
}
WEIGHTED_TOPDOWN_GROUP_SPECS = {
    "L1+": {"slug": "level1plus_split56_boundary", "color": "#0a6cff"},
    "L2+": {"slug": "level2plus_split56_boundary", "color": "#00a35c"},
    "L3+": {"slug": "level3plus_split56_boundary", "color": "#ff8a00"},
    "L4+": {"slug": "level4plus_split56_boundary", "color": "#7b1fa2"},
    "L5+": {"slug": "level5plus_split56_boundary", "color": "#c2185b"},
    "L6": {"slug": "level6_boundary", "color": "#6d4c41"},
}
WEIGHTED_TOPDOWN_DISPLAY_PRIORITY = ["L6", "L5+", "L4+", "L3+", "L2+", "L1+"]


def get_group_specs(view_mode: str) -> dict[str, dict[str, str]]:
    if view_mode == "clean_outer_seeds":
        return CLEAN_GROUP_SPECS
    if view_mode == "weighted_topdown_seed44":
        return WEIGHTED_TOPDOWN_GROUP_SPECS
    return STRATEGY_GROUP_SPECS


def get_group_order(view_mode: str) -> list[str]:
    return list(get_group_specs(view_mode).keys())


def get_xml_path(piece_id: str) -> Path:
    return PROJECT_ROOT / f"MazurkaBL-master/xml_scores/Mazurka{piece_id[1:]}.xml"


def get_beat_map_path(piece_id: str) -> Path:
    return PROJECT_ROOT / f"MazurkaBL-master/beat_time/{piece_id}beat_time.csv"


def get_predicted_events_path(level_slug: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/clean_outer_test/M06_outer_{level_slug}_seed{seed}/predicted_events.csv.gz"
    )


def get_summary_path(level_slug: str, seed: int) -> Path:
    return PROJECT_ROOT / f"MERIX SUBMISSION/Boundary_Restart/reports/clean_outer_test/M06_outer_{level_slug}_seed{seed}/summary.json"


def get_weighted_predicted_events_path(detector_target_slug: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/clean_outer_test/weighted_topdown_{detector_target_slug}_seed{seed}/predicted_events.csv.gz"
    )


def get_weighted_summary_path(detector_target_slug: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/reports/clean_outer_test/weighted_topdown_{detector_target_slug}_seed{seed}/summary.json"
    )


def get_strategy_specs(group_label: str) -> dict[str, str]:
    spec = STRATEGY_GROUP_SPECS[group_label]
    run_root = "strategy_compare_l5l6_u70" if spec["slug"] in {"level5", "level6"} else "strategy_compare_alllevels"
    return {"slug": spec["slug"], "color": spec["color"], "run_root": run_root}


def get_strategy_target(group_label: str) -> str:
    return f"{get_strategy_specs(group_label)['slug']}_boundary"


def get_strategy_predicted_events_path(piece_id: str, group_label: str, variant: str, seed: int = 42) -> Path:
    strategy_specs = get_strategy_specs(group_label)
    target = f"{strategy_specs['slug']}_boundary"
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/{strategy_specs['run_root']}/{piece_id}_{target}_{variant}_seed{seed}/predicted_events.csv.gz"
    )


def get_strategy_summary_path(piece_id: str, group_label: str, variant: str, seed: int = 42) -> Path:
    strategy_specs = get_strategy_specs(group_label)
    target = f"{strategy_specs['slug']}_boundary"
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/{strategy_specs['run_root']}/{piece_id}_{target}_{variant}_seed{seed}/summary.json"
    )


@CACHE_DATA(show_spinner=False)
def load_beat_map(piece_id: str) -> pd.DataFrame:
    df = pd.read_csv(get_beat_map_path(piece_id))
    return df[["measure_number", "beat_number"]].reset_index(drop=True)


@CACHE_DATA(show_spinner=False)
def load_group_events(piece_id: str, group_label: str, seed: int) -> pd.DataFrame:
    level_slug = CLEAN_GROUP_SPECS[group_label]["slug"]
    path = get_predicted_events_path(level_slug, seed)
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    frame = pd.read_csv(path)
    frame = frame[frame["piece_id"] == piece_id].copy()
    keep_cols = [
        col
        for col in [
            "beat_idx",
            "detector_score",
            "matched_union",
            "frequency_target_at_beat",
            "matched_true_beat_idx",
            "match_offset",
        ]
        if col in frame.columns
    ]
    return frame[keep_cols].copy()


@CACHE_DATA(show_spinner=False)
def load_weighted_group_events(piece_id: str, group_label: str, seed: int = 44) -> pd.DataFrame:
    detector_target_slug = WEIGHTED_TOPDOWN_GROUP_SPECS[group_label]["slug"]
    path = get_weighted_predicted_events_path(detector_target_slug, seed)
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    frame = pd.read_csv(path)
    frame = frame[frame["piece_id"] == piece_id].copy()
    keep_cols = [
        col
        for col in [
            "beat_idx",
            "detector_score",
            "matched_union",
            "frequency_target_at_beat",
            "matched_true_beat_idx",
            "match_offset",
        ]
        if col in frame.columns
    ]
    return frame[keep_cols].copy()


@CACHE_DATA(show_spinner=False)
def load_strategy_events(piece_id: str, group_label: str, variant: str, seed: int = 42) -> pd.DataFrame:
    path = get_strategy_predicted_events_path(piece_id, group_label, variant, seed=seed)
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    frame = pd.read_csv(path)
    keep_cols = [
        col
        for col in [
            "beat_idx",
            "detector_score",
            "matched_union",
            "frequency_target_at_beat",
            "matched_true_beat_idx",
            "match_offset",
        ]
        if col in frame.columns
    ]
    return frame[keep_cols].copy()


@CACHE_DATA(show_spinner=False)
def load_group_summary(group_label: str, seed: int) -> dict:
    level_slug = CLEAN_GROUP_SPECS[group_label]["slug"]
    path = get_summary_path(level_slug, seed)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@CACHE_DATA(show_spinner=False)
def load_weighted_group_summary(group_label: str, seed: int = 44) -> dict:
    detector_target_slug = WEIGHTED_TOPDOWN_GROUP_SPECS[group_label]["slug"]
    path = get_weighted_summary_path(detector_target_slug, seed)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@CACHE_DATA(show_spinner=False)
def load_strategy_summary(piece_id: str, group_label: str, variant: str, seed: int = 42) -> dict:
    path = get_strategy_summary_path(piece_id, group_label, variant, seed=seed)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_measure_number(measure_attr: str) -> int | None:
    match = re.search(r"\d+", str(measure_attr))
    if not match:
        return None
    return int(match.group(0))


def map_events_to_measures(events: pd.DataFrame, beat_map: pd.DataFrame) -> list[dict]:
    mapped: list[dict] = []
    n = len(beat_map)
    for row in events.itertuples(index=False):
        beat_idx = int(row.beat_idx)
        idx = beat_idx if 0 <= beat_idx < n else (beat_idx - 1 if 0 <= beat_idx - 1 < n else None)
        if idx is None:
            continue
        beat_row = beat_map.iloc[idx]
        mapped.append(
            {
                "beat_idx": beat_idx,
                "measure": int(beat_row["measure_number"]),
                "beat_in_measure": int(beat_row["beat_number"]) + 1,
                "detector_score": float(getattr(row, "detector_score", 0.0)),
                "matched_union": bool(getattr(row, "matched_union", False)),
                "frequency_target_at_beat": float(getattr(row, "frequency_target_at_beat", 0.0)),
                "matched_true_beat_idx": getattr(row, "matched_true_beat_idx", None),
                "match_offset": getattr(row, "match_offset", None),
            }
        )
    return mapped


def build_breakpoint_table(piece_id: str, selected_groups: list[str], seed: int) -> pd.DataFrame:
    beat_map = load_beat_map(piece_id)
    rows: list[dict] = []
    for group_label in selected_groups:
        mapped = map_events_to_measures(load_group_events(piece_id, group_label, seed), beat_map)
        for item in mapped:
            rows.append(
                {
                    "group": group_label,
                    "seed": seed,
                    "beat_idx": item["beat_idx"],
                    "measure": item["measure"],
                    "beat_in_measure": item["beat_in_measure"],
                    "detector_score": item["detector_score"],
                    "matched_union": item["matched_union"],
                    "frequency_target_at_beat": item["frequency_target_at_beat"],
                    "matched_true_beat_idx": item["matched_true_beat_idx"],
                    "match_offset": item["match_offset"],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "group",
                "seed",
                "beat_idx",
                "measure",
                "beat_in_measure",
                "detector_score",
                "matched_union",
                "frequency_target_at_beat",
                "matched_true_beat_idx",
                "match_offset",
            ]
        )
    return pd.DataFrame(rows).sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)


def build_strategy_breakpoint_table(piece_id: str, selected_groups: list[str], variant: str, seed: int = 42) -> pd.DataFrame:
    beat_map = load_beat_map(piece_id)
    rows: list[dict] = []
    for group_label in selected_groups:
        mapped = map_events_to_measures(load_strategy_events(piece_id, group_label, variant, seed=seed), beat_map)
        for item in mapped:
            rows.append(
                {
                    "group": group_label,
                    "variant": variant,
                    "seed": seed,
                    "beat_idx": item["beat_idx"],
                    "measure": item["measure"],
                    "beat_in_measure": item["beat_in_measure"],
                    "detector_score": item["detector_score"],
                    "matched_union": item["matched_union"],
                    "frequency_target_at_beat": item["frequency_target_at_beat"],
                    "matched_true_beat_idx": item["matched_true_beat_idx"],
                    "match_offset": item["match_offset"],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "group",
                "variant",
                "seed",
                "beat_idx",
                "measure",
                "beat_in_measure",
                "detector_score",
                "matched_union",
                "frequency_target_at_beat",
                "matched_true_beat_idx",
                "match_offset",
            ]
        )
    return pd.DataFrame(rows).sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)


def build_weighted_breakpoint_table(piece_id: str, selected_groups: list[str], seed: int = 44) -> pd.DataFrame:
    beat_map = load_beat_map(piece_id)
    rows: list[dict] = []
    for group_label in selected_groups:
        mapped = map_events_to_measures(load_weighted_group_events(piece_id, group_label, seed), beat_map)
        for item in mapped:
            rows.append(
                {
                    "group": group_label,
                    "seed": seed,
                    "beat_idx": item["beat_idx"],
                    "measure": item["measure"],
                    "beat_in_measure": item["beat_in_measure"],
                    "detector_score": item["detector_score"],
                    "matched_union": item["matched_union"],
                    "frequency_target_at_beat": item["frequency_target_at_beat"],
                    "matched_true_beat_idx": item["matched_true_beat_idx"],
                    "match_offset": item["match_offset"],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "group",
                "seed",
                "beat_idx",
                "measure",
                "beat_in_measure",
                "detector_score",
                "matched_union",
                "frequency_target_at_beat",
                "matched_true_beat_idx",
                "match_offset",
            ]
        )
    return pd.DataFrame(rows).sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)


def suppress_lower_breakpoints_within_tolerance(
    breakpoints: pd.DataFrame,
    *,
    priority_order: list[str],
    tolerance: int = 1,
) -> pd.DataFrame:
    if breakpoints.empty:
        return breakpoints

    priority_rank = {group: idx for idx, group in enumerate(priority_order)}
    kept_beats: list[int] = []
    kept_rows = []

    ordered = breakpoints.copy()
    ordered["priority_rank"] = ordered["group"].map(lambda value: priority_rank.get(value, len(priority_rank)))
    ordered = ordered.sort_values(["priority_rank", "beat_idx", "detector_score"], ascending=[True, True, False])

    for row in ordered.itertuples(index=False):
        beat_idx = int(row.beat_idx)
        if any(abs(beat_idx - existing) <= tolerance for existing in kept_beats):
            continue
        kept_beats.append(beat_idx)
        kept_rows.append(
            {
                "group": row.group,
                "seed": row.seed,
                "beat_idx": beat_idx,
                "measure": row.measure,
                "beat_in_measure": row.beat_in_measure,
                "detector_score": row.detector_score,
                "matched_union": row.matched_union,
                "frequency_target_at_beat": row.frequency_target_at_beat,
                "matched_true_beat_idx": row.matched_true_beat_idx,
                "match_offset": row.match_offset,
            }
        )

    return pd.DataFrame(kept_rows).sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)


def build_summary_table(piece_id: str, selected_groups: list[str], seed: int, breakpoints: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for group_label in selected_groups:
        group_df = breakpoints[breakpoints["group"] == group_label].copy()
        summary = load_group_summary(group_label, seed)
        union_metrics = summary.get("union_metrics", {})
        summary_rows.append(
            {
                "group": group_label,
                "seed": seed,
                "events": int(len(group_df)) if not group_df.empty else int(union_metrics.get("pred_events", 0)),
                "measures": int(group_df["measure"].nunique()) if not group_df.empty else 0,
                "threshold": union_metrics.get("threshold"),
                "precision": union_metrics.get("union_precision"),
                "union_recall": union_metrics.get("union_recall"),
                "weighted_recall": union_metrics.get("weighted_recall"),
                "consensus_recall": union_metrics.get("consensus_recall"),
                "frozen_epochs": summary.get("frozen_epochs"),
            }
        )
    return pd.DataFrame(summary_rows)


def build_strategy_summary_table(
    piece_id: str,
    selected_groups: list[str],
    variant: str,
    breakpoints: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    summary_rows = []
    for group_label in selected_groups:
        group_df = breakpoints[breakpoints["group"] == group_label].copy()
        summary = load_strategy_summary(piece_id, group_label, variant, seed=seed)
        union_metrics = summary.get("union_metrics", {})
        floors = summary.get("precision_floors", {})
        summary_rows.append(
            {
                "group": group_label,
                "variant": variant,
                "events": int(len(group_df)) if not group_df.empty else int(union_metrics.get("pred_events", 0)),
                "measures": int(group_df["measure"].nunique()) if not group_df.empty else 0,
                "precision_metric": summary.get("precision_metric"),
                "precision_floors": json.dumps(floors, ensure_ascii=False, sort_keys=True),
                "threshold": union_metrics.get("threshold"),
                "precision": union_metrics.get("union_precision"),
                "freq_precision": union_metrics.get("frequency_weighted_precision"),
                "cons_precision": union_metrics.get("consensus_precision"),
                "union_recall": union_metrics.get("union_recall"),
                "weighted_recall": union_metrics.get("weighted_recall"),
                "consensus_recall": union_metrics.get("consensus_recall"),
                "best_epoch": summary.get("best_epoch"),
            }
        )
    return pd.DataFrame(summary_rows)


def build_weighted_summary_table(piece_id: str, selected_groups: list[str], breakpoints: pd.DataFrame, seed: int = 44) -> pd.DataFrame:
    summary_rows = []
    for group_label in selected_groups:
        group_df = breakpoints[breakpoints["group"] == group_label].copy()
        summary = load_weighted_group_summary(group_label, seed)
        union_metrics = summary.get("union_metrics", {})
        summary_rows.append(
            {
                "group": group_label,
                "seed": seed,
                "events": int(len(group_df)) if not group_df.empty else int(union_metrics.get("pred_events", 0)),
                "measures": int(group_df["measure"].nunique()) if not group_df.empty else 0,
                "threshold": summary.get("frozen_threshold", union_metrics.get("threshold")),
                "precision": union_metrics.get("union_precision"),
                "freq_precision": union_metrics.get("frequency_weighted_precision"),
                "union_recall": union_metrics.get("union_recall"),
                "weighted_recall": union_metrics.get("weighted_recall"),
                "consensus_recall": union_metrics.get("consensus_recall"),
                "frozen_epochs": summary.get("frozen_epochs"),
            }
        )
    return pd.DataFrame(summary_rows)


def build_annotated_musicxml(
    xml_path: Path,
    breakpoints: pd.DataFrame,
    selected_groups: list[str],
    group_specs: dict[str, dict[str, str]],
) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    by_measure_group: dict[int, dict[str, list[dict]]] = {}
    for row in breakpoints.itertuples(index=False):
        by_measure_group.setdefault(int(row.measure), {}).setdefault(str(row.group), []).append(
            {
                "beat_idx": int(row.beat_idx),
                "beat_in_measure": int(row.beat_in_measure),
                "detector_score": float(row.detector_score),
            }
        )

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            measure_no = _as_measure_number(measure.attrib.get("number", ""))
            if measure_no is None or measure_no not in by_measure_group:
                continue
            insertions = []
            for group_label in selected_groups:
                items = by_measure_group[measure_no].get(group_label, [])
                if not items:
                    continue
                color = group_specs[group_label]["color"]
                beat_text = ",".join(f"b{item['beat_in_measure']}" for item in items[:4])
                if len(items) > 4:
                    beat_text += ",..."
                direction = ET.Element("direction", {"placement": "above"})
                direction_type = ET.SubElement(direction, "direction-type")
                ET.SubElement(
                    direction_type,
                    "words",
                    {"color": color, "font-weight": "bold", "font-size": "10"},
                ).text = f"{group_label}: {beat_text}"
                insertions.append(direction)
            for direction in reversed(insertions):
                measure.insert(0, direction)

    return ET.tostring(root, encoding="unicode")


def render_score(
    xml_text: str,
    group_counts: dict[str, int],
    group_specs: dict[str, dict[str, str]],
    height: int = 1000,
) -> None:
    if components is None:
        raise RuntimeError("streamlit is not installed; use this module via `streamlit run app.py`.")
    if not xml_text.lstrip().startswith("<?xml"):
        xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
    xml_b64 = base64.b64encode(xml_text.encode("utf-8")).decode("ascii")
    xml_b64_js = json.dumps(xml_b64)

    legend_html = []
    for group_label in group_specs:
        if group_label not in group_counts:
            continue
        color = group_specs[group_label]["color"]
        count = group_counts[group_label]
        legend_html.append(
            f'<span style="display:inline-block;background:{color};color:white;'
            f'border-radius:999px;padding:2px 10px;font-size:12px;margin-right:8px;">'
            f"{group_label}: {count}</span>"
        )
    legend_text = "".join(legend_html)

    html = f"""
    <div style="padding: 8px 0 12px 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;">
      {legend_text}
    </div>
    <div id="status" style="font-size:13px; color:#555; margin-bottom:8px;">正在加载曲谱...</div>
    <div id="score" style="background: white; border: 1px solid #ddd; border-radius: 12px; padding: 8px;"></div>
    <script>
      const status = document.getElementById("status");
      const xmlText = atob({xml_b64_js});

      function setStatus(text, isError=false) {{
        status.textContent = text;
        status.style.color = isError ? "#b00020" : "#555";
      }}

      function renderWithOSMD() {{
        try {{
          const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score", {{
            autoResize: true,
            backend: "svg",
            drawingParameters: "compacttight",
            drawPartNames: false,
          }});
          osmd.load(xmlText)
            .then(() => {{
              osmd.render();
              setStatus("加载完成");
            }})
            .catch(err => setStatus("OSMD 解析失败: " + err, true));
        }} catch (err) {{
          setStatus("OSMD 初始化失败: " + err, true);
        }}
      }}

      function loadScript(src, onOk, onErr) {{
        const s = document.createElement("script");
        s.src = src;
        s.onload = onOk;
        s.onerror = onErr;
        document.head.appendChild(s);
      }}

      if (window.opensheetmusicdisplay) {{
        renderWithOSMD();
      }} else {{
        loadScript(
          "https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.2/build/opensheetmusicdisplay.min.js",
          renderWithOSMD,
          () => setStatus("无法加载 OSMD 脚本（可能是网络/CSP问题）", true)
        );
      }}
    </script>
    """
    components.html(html, height=height, scrolling=True)


def render_seed_panel(piece_id: str, selected_groups: list[str], seed: int) -> None:
    xml_path = get_xml_path(piece_id)
    breakpoints = build_breakpoint_table(piece_id, selected_groups, seed)
    summary_df = build_summary_table(piece_id, selected_groups, seed, breakpoints)

    st.markdown(f"### Seed {seed}")
    if breakpoints.empty:
        st.info("当前本地还没有这个 seed 的事件表，只显示摘要。")
    else:
        annotated_xml = build_annotated_musicxml(xml_path, breakpoints, selected_groups, CLEAN_GROUP_SPECS)
        render_score(
            annotated_xml,
            group_counts={row["group"]: int(row["events"]) for _, row in summary_df.iterrows()},
            group_specs=CLEAN_GROUP_SPECS,
            height=980,
        )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.dataframe(breakpoints, use_container_width=True, hide_index=True)


def render_strategy_panel(piece_id: str, selected_groups: list[str], variant: str, seed: int = 42) -> None:
    xml_path = get_xml_path(piece_id)
    breakpoints = build_strategy_breakpoint_table(piece_id, selected_groups, variant, seed=seed)
    summary_df = build_strategy_summary_table(piece_id, selected_groups, variant, breakpoints, seed=seed)

    st.markdown(f"### {STRATEGY_LABELS.get(variant, variant)}")
    if breakpoints.empty:
        st.info("当前本地还没有这个策略的事件表，只显示摘要。")
    else:
        annotated_xml = build_annotated_musicxml(xml_path, breakpoints, selected_groups, STRATEGY_GROUP_SPECS)
        render_score(
            annotated_xml,
            group_counts={row["group"]: int(row["events"]) for _, row in summary_df.iterrows()},
            group_specs=STRATEGY_GROUP_SPECS,
            height=980,
        )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.dataframe(breakpoints, use_container_width=True, hide_index=True)


def render_weighted_seed44_panel(piece_id: str, selected_groups: list[str]) -> None:
    xml_path = get_xml_path(piece_id)
    raw_breakpoints = build_weighted_breakpoint_table(piece_id, selected_groups, seed=44)
    breakpoints = suppress_lower_breakpoints_within_tolerance(
        raw_breakpoints,
        priority_order=[group for group in WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in selected_groups],
        tolerance=1,
    )
    summary_df = build_weighted_summary_table(piece_id, selected_groups, breakpoints, seed=44)

    st.markdown("### Weighted Topdown Clean Outer Seed44")
    st.caption("显示规则：高层优先；若更低层断点落在更高层断点的 ±1 beat 内，则在谱面和事件表中隐藏。")
    if breakpoints.empty:
        st.info("当前本地还没有这批 weighted-topdown seed44 的事件表，只显示摘要。")
    else:
        annotated_xml = build_annotated_musicxml(xml_path, breakpoints, selected_groups, WEIGHTED_TOPDOWN_GROUP_SPECS)
        render_score(
            annotated_xml,
            group_counts={row["group"]: int(row["events"]) for _, row in summary_df.iterrows()},
            group_specs=WEIGHTED_TOPDOWN_GROUP_SPECS,
            height=980,
        )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.dataframe(breakpoints, use_container_width=True, hide_index=True)


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed; install it before running this app.")
    st.set_page_config(page_title="Mazurka Clean Outer Breakpoint Visualizer", layout="wide")
    st.title("Mazurka Breakpoint Visualizer")
    view_mode = st.sidebar.radio(
        "展示模式",
        options=["weighted_topdown_seed44", "clean_outer_seeds", "strategy_compare"],
        format_func=lambda value: {
            "weighted_topdown_seed44": "Weighted Topdown Seed44",
            "clean_outer_seeds": "Clean Outer Seed42 vs Seed44",
            "strategy_compare": "Baseline vs Consensus Guarded",
        }[value],
    )
    if view_mode == "weighted_topdown_seed44":
        st.caption("展示 weighted top-down wide-gap + TCN direct + train floor 0.05 的 clean outer seed44 六层结果。")
    elif view_mode == "clean_outer_seeds":
        st.caption("展示 clean outer test 下的五层结果；每层分别显示 seed42 和 seed44。")
    else:
        st.caption("展示 5 首曲子、6 个 level 下的 baseline 与新策略对照；策略模式下高层拆成 L5 和 L6，并使用 union precision floor 0.7 的新结果。")

    group_order = get_group_order(view_mode)
    selected_groups = st.sidebar.multiselect("显示层级", options=group_order, default=group_order)
    if not selected_groups:
        st.warning("请至少选择一层。")
        return

    pieces = WEIGHTED_TOPDOWN_PIECES if view_mode == "weighted_topdown_seed44" else SUPPORTED_PIECES
    tabs = st.tabs(pieces)
    for tab, piece_id in zip(tabs, pieces):
        with tab:
            st.subheader(piece_id)
            st.code(str(get_xml_path(piece_id)))
            if view_mode == "weighted_topdown_seed44":
                render_weighted_seed44_panel(piece_id, selected_groups)
            else:
                cols = st.columns(2)
                if view_mode == "clean_outer_seeds":
                    for col, seed in zip(cols, SEED_ORDER):
                        with col:
                            render_seed_panel(piece_id, selected_groups, seed)
                else:
                    for col, variant in zip(cols, STRATEGY_ORDER):
                        with col:
                            render_strategy_panel(piece_id, selected_groups, variant, seed=42)


if __name__ == "__main__":
    main()
