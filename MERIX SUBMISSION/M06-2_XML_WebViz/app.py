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
SUPPORTED_PIECES = ["M06-2", "M17-1", "M30-1"]
GROUP_SPECS = {
    "L1": {
        "detector_target": "level1_boundary",
        "color": "#0a6cff",
        "run_name": "tcn_level1_boundary_union_recall_cpu",
    },
    "L2": {
        "detector_target": "level2_boundary",
        "color": "#00a35c",
        "run_name": "tcn_level2_boundary_union_recall_cpu",
    },
    "L3+4": {
        "detector_target": "level34_boundary",
        "color": "#ff8a00",
        "run_name": "tcn_level34_boundary_union_recall_cpu",
    },
    "L5+6": {
        "detector_target": "level56_boundary",
        "color": "#c2185b",
        "run_name": "tcn_level56_boundary_union_recall_cpu",
    },
}
GROUP_ORDER = list(GROUP_SPECS.keys())


def get_xml_path(piece_id: str) -> Path:
    return PROJECT_ROOT / f"MazurkaBL-master/xml_scores/Mazurka{piece_id[1:]}.xml"


def get_beat_map_path(piece_id: str) -> Path:
    return PROJECT_ROOT / f"MazurkaBL-master/beat_time/{piece_id}beat_time.csv"


def get_predicted_events_path(piece_id: str, group_label: str) -> Path:
    run_name = GROUP_SPECS[group_label]["run_name"]
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/{piece_id}_4groups/{run_name}/predicted_events.csv.gz"
    )


def get_summary_path(piece_id: str, group_label: str) -> Path:
    run_name = GROUP_SPECS[group_label]["run_name"]
    return PROJECT_ROOT / f"MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/{piece_id}_4groups/{run_name}/summary.json"


@CACHE_DATA(show_spinner=False)
def load_beat_map(piece_id: str) -> pd.DataFrame:
    df = pd.read_csv(get_beat_map_path(piece_id))
    return df[["measure_number", "beat_number"]].reset_index(drop=True)


@CACHE_DATA(show_spinner=False)
def load_group_events(piece_id: str, group_label: str) -> pd.DataFrame:
    path = get_predicted_events_path(piece_id, group_label)
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    frame = pd.read_csv(path)
    keep_cols = [col for col in ["beat_idx", "detector_score", "matched_union", "frequency_target_at_beat"] if col in frame.columns]
    return frame[keep_cols].copy()


@CACHE_DATA(show_spinner=False)
def load_group_summary(piece_id: str, group_label: str) -> dict:
    path = get_summary_path(piece_id, group_label)
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
            }
        )
    return mapped


def build_breakpoint_table(piece_id: str, selected_groups: list[str]) -> pd.DataFrame:
    beat_map = load_beat_map(piece_id)
    rows: list[dict] = []
    for group_label in selected_groups:
        mapped = map_events_to_measures(load_group_events(piece_id, group_label), beat_map)
        for item in mapped:
            rows.append(
                {
                    "group": group_label,
                    "beat_idx": item["beat_idx"],
                    "measure": item["measure"],
                    "beat_in_measure": item["beat_in_measure"],
                    "detector_score": item["detector_score"],
                    "matched_union": item["matched_union"],
                    "frequency_target_at_beat": item["frequency_target_at_beat"],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "group",
                "beat_idx",
                "measure",
                "beat_in_measure",
                "detector_score",
                "matched_union",
                "frequency_target_at_beat",
            ]
        )
    return pd.DataFrame(rows).sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)


def build_annotated_musicxml(xml_path: Path, breakpoints: pd.DataFrame, selected_groups: list[str]) -> str:
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
                color = GROUP_SPECS[group_label]["color"]
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


def render_score(xml_text: str, group_counts: dict[str, int], height: int = 1050) -> None:
    if components is None:
        raise RuntimeError("streamlit is not installed; use this module via `streamlit run app.py`.")
    if not xml_text.lstrip().startswith("<?xml"):
        xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
    xml_b64 = base64.b64encode(xml_text.encode("utf-8")).decode("ascii")
    xml_b64_js = json.dumps(xml_b64)

    legend_html = []
    for group_label in GROUP_ORDER:
        if group_label not in group_counts:
            continue
        color = GROUP_SPECS[group_label]["color"]
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


def render_piece_tab(piece_id: str, selected_groups: list[str]) -> None:
    xml_path = get_xml_path(piece_id)
    beat_map_path = get_beat_map_path(piece_id)
    breakpoints = build_breakpoint_table(piece_id, selected_groups)

    summary_rows = []
    for group_label in selected_groups:
        group_df = breakpoints[breakpoints["group"] == group_label].copy()
        summary = load_group_summary(piece_id, group_label)
        union_metrics = summary.get("union_metrics", {})
        summary_rows.append(
            {
                "group": group_label,
                "events": int(len(group_df)),
                "measures": int(group_df["measure"].nunique()) if not group_df.empty else 0,
                "threshold": union_metrics.get("threshold"),
                "precision": union_metrics.get("union_precision"),
                "union_recall": union_metrics.get("union_recall"),
                "weighted_recall": union_metrics.get("weighted_recall"),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    annotated_xml = build_annotated_musicxml(xml_path, breakpoints, selected_groups)
    render_score(
        annotated_xml,
        group_counts={row["group"]: int(row["events"]) for row in summary_rows},
        height=1080,
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("四层结果概览")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with col_right:
        st.subheader("路径")
        st.code(str(xml_path))
        st.code(str(beat_map_path))

    st.subheader("断点列表")
    st.dataframe(breakpoints, use_container_width=True, hide_index=True)


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed; install it before running this app.")
    st.set_page_config(page_title="Mazurka Four-Level Breakpoint Visualizer", layout="wide")
    st.title("Mazurka Four-Level Breakpoint Visualizer")
    st.caption("使用当前 Boundary_Restart 的 TCN direct 四层结果，在乐谱上叠加显示断点")

    selected_groups = st.sidebar.multiselect(
        "显示层级",
        options=GROUP_ORDER,
        default=GROUP_ORDER,
    )
    if not selected_groups:
        st.warning("请至少选择一层。")
        return

    tabs = st.tabs(SUPPORTED_PIECES)
    for tab, piece_id in zip(tabs, SUPPORTED_PIECES):
        with tab:
            render_piece_tab(piece_id, selected_groups)


if __name__ == "__main__":
    main()
