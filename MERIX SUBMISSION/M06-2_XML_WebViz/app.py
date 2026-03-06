from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "MazurkaBL-master").exists() and (candidate / "MERIX SUBMISSION").exists():
            return candidate
    raise FileNotFoundError("无法定位项目根目录（缺少 MazurkaBL-master 或 MERIX SUBMISSION）")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SUPPORTED_PIECES = ["M06-1", "M06-2", "M06-3"]


def get_xml_path(piece_id: str) -> Path:
    return PROJECT_ROOT / f"MazurkaBL-master/xml_scores/Mazurka{piece_id[1:]}.xml"


def get_beat_map_path(piece_id: str) -> Path:
    return PROJECT_ROOT / f"MazurkaBL-master/beat_time/{piece_id}beat_time.csv"


def parse_peaks(piece_id: str, level: int) -> list[int]:
    out_dir = PROJECT_ROOT / "MERIX SUBMISSION/MIREX_Model/out"
    if piece_id == "M06-2":
        peaks_path = out_dir / "M06-2_all_levels_pred_peaks.csv"
        df = pd.read_csv(peaks_path)
        row = df[df["level"] == level]
        if row.empty:
            return []
        values = str(row.iloc[0]["pred_peaks"])
        return [int(x) for x in re.findall(r"\d+", values)]

    peaks_path = out_dir / f"pred_{piece_id}_peaks.csv"
    if not peaks_path.exists():
        return []
    df = pd.read_csv(peaks_path)
    if "is_peak" not in df.columns or "beat_index" not in df.columns:
        return []
    return df.loc[df["is_peak"] == 1, "beat_index"].astype(int).tolist()


def load_beat_map(piece_id: str) -> pd.DataFrame:
    df = pd.read_csv(get_beat_map_path(piece_id))
    return df[["measure_number", "beat_number"]].reset_index(drop=True)


def map_peak_beats_to_measures(peak_beats: list[int], beat_map: pd.DataFrame) -> list[dict]:
    mapped: list[dict] = []
    n = len(beat_map)
    for b in peak_beats:
        # 兼容 0-based 和 1-based 两种峰值编号
        idx = b if 0 <= b < n else (b - 1 if 0 <= b - 1 < n else None)
        if idx is None:
            continue
        row = beat_map.iloc[idx]
        mapped.append(
            {
                "beat_index": b,
                "measure": int(row["measure_number"]),
                "beat_in_measure": int(row["beat_number"]) + 1,
            }
        )
    return mapped


def _as_measure_number(measure_attr: str) -> int | None:
    m = re.search(r"\d+", str(measure_attr))
    if not m:
        return None
    return int(m.group(0))


def build_annotated_musicxml(xml_path: Path, mapped_breakpoints: list[dict]) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    label_by_measure: dict[int, list[str]] = {}
    for item in mapped_breakpoints:
        label_by_measure.setdefault(item["measure"], []).append(
            f'B{item["beat_index"]}(b{item["beat_in_measure"]})'
        )

    # 给断点小节插入 "BP" 标记，直接显示在谱面上方
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            n = _as_measure_number(measure.attrib.get("number", ""))
            if n is None or n not in label_by_measure:
                continue
            direction = ET.Element("direction", {"placement": "above"})
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(
                direction_type,
                "words",
                {"color": "#cc0000", "font-weight": "bold", "font-size": "11"},
            ).text = "BP " + ", ".join(label_by_measure[n][:2])
            measure.insert(0, direction)

    return ET.tostring(root, encoding="unicode")


def render_score(xml_text: str, mapped_breakpoints: list[dict], height: int = 980) -> None:
    if not xml_text.lstrip().startswith("<?xml"):
        xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
    xml_b64 = base64.b64encode(xml_text.encode("utf-8")).decode("ascii")
    xml_b64_js = json.dumps(xml_b64)
    breakpoints_text = ", ".join(f'B{x["beat_index"]}->M{x["measure"]}' for x in mapped_breakpoints[:20])
    if len(mapped_breakpoints) > 20:
        breakpoints_text += " ..."
    bp_js = json.dumps(mapped_breakpoints)
    html = f"""
    <div style="padding: 8px 0 12px 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;">
      <span style="display:inline-block; background:#ffecec; color:#a30000; border:1px solid #f3b4b4; border-radius:999px; padding:2px 10px; font-size:12px;">
        断点(beat→measure): {breakpoints_text if mapped_breakpoints else '无'}
      </span>
    </div>
    <div id="status" style="font-size:13px; color:#555; margin-bottom:8px;">正在加载曲谱...</div>
    <div id="score" style="background: white; border: 1px solid #ddd; border-radius: 12px; padding: 8px;"></div>
    <script>
      const status = document.getElementById("status");
      const xmlText = atob({xml_b64_js});
      const breakpoints = {bp_js};

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
              setStatus(`加载完成（断点数量: ${{breakpoints.length}}）`);
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


def main() -> None:
    st.set_page_config(page_title="Mazurka XML Breakpoint Visualizer", layout="wide")
    st.title("Mazurka XML Breakpoint Visualizer")
    st.caption("直接显示曲谱，并在断点小节标出 BP")

    piece_id = st.sidebar.selectbox("选择曲子", options=SUPPORTED_PIECES, index=2)
    level = st.sidebar.slider("选择层级 level", min_value=1, max_value=6, value=3, step=1)
    xml_path = get_xml_path(piece_id)
    beat_map_path = get_beat_map_path(piece_id)

    st.sidebar.markdown("**数据路径**")
    st.sidebar.code(str(xml_path))
    st.sidebar.code(str(beat_map_path))

    peak_beats = parse_peaks(piece_id, level)
    beat_map = load_beat_map(piece_id)
    mapped_breakpoints = map_peak_beats_to_measures(peak_beats, beat_map)
    annotated_xml = build_annotated_musicxml(xml_path, mapped_breakpoints)

    st.subheader("谱面")
    render_score(annotated_xml, mapped_breakpoints, height=1050)
    st.subheader("断点列表")
    st.write(mapped_breakpoints)
    st.write(
        {
            "beats_total": int(len(beat_map)),
            "measures_total": int(beat_map["measure_number"].max()),
            "breakpoints": len(mapped_breakpoints),
            "piece": piece_id,
            "level": level,
        }
    )


if __name__ == "__main__":
    main()
