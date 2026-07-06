from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app.py"
PAGED_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_pdf.py"
EXPORT_DIR = SCRIPT_DIR / "exports" / "score_pdf_long"
HTML_DIR = EXPORT_DIR / "html"
PDF_DIR = EXPORT_DIR / "pdf"
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PAGE_WIDTH_MM = 190.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export long single-page score PDFs for weighted topdown seed44.")
    parser.add_argument("--pieces", nargs="+", help="Piece ids to export. Defaults to all weighted-topdown pieces.")
    parser.add_argument("--groups", nargs="+", help="Displayed target groups. Defaults to all weighted-topdown groups.")
    parser.add_argument("--seed", type=int, default=44, help="Seed label for event loading and output metadata.")
    parser.add_argument("--output-tag", help="Custom output stem, without piece id or suffix.")
    parser.add_argument("--subtitle", default="", help="Optional subtitle shown under each piece title.")
    parser.add_argument("--display-l5plus6-as-l5", action="store_true", help="Display the internal L5+6 group label as L5.")
    parser.add_argument(
        "--merge-tolerance",
        type=int,
        default=1,
        help="Suppress lower-priority groups within this many beats. Use 0 for same-beat de-dup only.",
    )
    parser.add_argument(
        "--per-group-keep-json",
        help='Optional JSON mapping group labels to top-score keep fractions, e.g. {"L1+":0.2,"L5+6":0.6}.',
    )
    parser.add_argument(
        "--breakpoint-selection-dir",
        help="Directory containing {piece_id}_selected_level_breakpoints_seed44.csv files.",
    )
    parser.add_argument(
        "--predicted-events-root",
        help="Directory containing {detector_target}/predicted_events.csv.gz files.",
    )
    parser.add_argument("--show-measure-number-in-label", action="store_true")
    parser.add_argument("--hide-level-names-in-label", action="store_true")
    parser.add_argument("--black-labels-with-random-points", action="store_true")
    parser.add_argument("--number-labels", action="store_true")
    return parser.parse_args()


def load_app_module():
    spec = importlib.util.spec_from_file_location("webviz_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load app module from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_paged_export_module():
    spec = importlib.util.spec_from_file_location("webviz_export_pdf", PAGED_EXPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load export module from {PAGED_EXPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def px_to_mm(px: float) -> float:
    return px * 25.4 / 96.0


def estimate_piece_height_mm(app, piece_id: str) -> float:
    score_height_px = float(app.estimate_score_height(piece_id))
    # Keep the page comfortably larger than the rendered score.
    return max(300.0, px_to_mm(score_height_px) * 1.35 + 35.0)


def build_numbered_annotated_musicxml(app, xml_path: Path, breakpoints, group_specs: dict[str, dict[str, str]]) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if hasattr(app, "apply_display_clef_fixes"):
        app.apply_display_clef_fixes(root, xml_path)
    sorted_breakpoints = breakpoints.sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)
    by_measure: dict[int, list[dict[str, str]]] = {}
    for idx, row in enumerate(sorted_breakpoints.itertuples(index=False), start=1):
        by_measure.setdefault(int(row.measure), []).append(
            {
                "label": str(idx),
                "color": "#d40000",
            }
        )

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            measure_no = app._as_measure_number(measure.attrib.get("number", ""))
            if measure_no is None or measure_no not in by_measure:
                continue
            insertions = []
            for item in by_measure[measure_no]:
                direction = ET.Element("direction", {"placement": "above"})
                direction_type = ET.SubElement(direction, "direction-type")
                ET.SubElement(
                    direction_type,
                    "words",
                    {"color": item["color"], "font-weight": "bold", "font-size": "10"},
                ).text = item["label"]
                insertions.append(direction)
            for direction in reversed(insertions):
                measure.insert(0, direction)

    return ET.tostring(root, encoding="unicode")


def build_selected_breakpoint_table(app, piece_id: str, selected_groups: list[str], selection_dir: Path, seed: int):
    path = selection_dir / f"{piece_id}_selected_level_breakpoints_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selected breakpoint file not found: {path}")

    table = app.pd.read_csv(path)
    beat_map = app.load_beat_map(piece_id)
    rows: list[dict] = []
    for row in table.itertuples(index=False):
        group_label = str(row.level)
        if group_label not in selected_groups:
            continue
        selected_beats = ast.literal_eval(str(row.selected_beats))
        for beat_idx_raw in selected_beats:
            beat_idx = int(beat_idx_raw)
            map_idx = beat_idx if 0 <= beat_idx < len(beat_map) else beat_idx - 1
            if map_idx < 0 or map_idx >= len(beat_map):
                continue
            beat_row = beat_map.iloc[map_idx]
            rows.append(
                {
                    "group": group_label,
                    "seed": seed,
                    "beat_idx": beat_idx,
                    "measure": int(beat_row["measure_number"]),
                    "beat_in_measure": int(beat_row["beat_number"]) + 1,
                    "detector_score": 0.0,
                    "matched_union": False,
                    "frequency_target_at_beat": 0.0,
                    "matched_true_beat_idx": None,
                    "match_offset": None,
                }
            )
    if not rows:
        return app.pd.DataFrame(
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
    return (
        app.pd.DataFrame(rows)
        .sort_values(["measure", "beat_in_measure", "group", "beat_idx"])
        .reset_index(drop=True)
    )


def filter_top_fraction(app, frame, keep_fraction: float):
    keep_fraction = float(keep_fraction)
    if frame.empty or keep_fraction >= 1.0:
        return frame
    if keep_fraction <= 0.0:
        return frame.iloc[0:0].copy()
    keep_count = int(math.ceil(len(frame) * keep_fraction))
    keep_count = min(max(keep_count, 0), len(frame))
    if keep_count <= 0:
        return frame.iloc[0:0].copy()
    return frame.sort_values(["detector_score", "beat_idx"], ascending=[False, True]).head(keep_count).copy()


def build_predicted_events_breakpoint_table(
    app,
    piece_id: str,
    selected_groups: list[str],
    events_root: Path,
    seed: int,
    group_keep_fractions: dict[str, float] | None = None,
):
    beat_map = app.load_beat_map(piece_id)
    rows: list[dict] = []
    for group_label in selected_groups:
        detector_target = app.WEIGHTED_TOPDOWN_GROUP_SPECS[group_label]["slug"]
        path = events_root / detector_target / "predicted_events.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Predicted event file not found: {path}")
        frame = app.pd.read_csv(path)
        frame = frame[frame["piece_id"] == piece_id].copy()
        if group_keep_fractions is not None and group_label in group_keep_fractions:
            frame = filter_top_fraction(app, frame, float(group_keep_fractions[group_label]))
        for row in frame.itertuples(index=False):
            beat_idx = int(row.beat_idx)
            map_idx = beat_idx if 0 <= beat_idx < len(beat_map) else beat_idx - 1
            if map_idx < 0 or map_idx >= len(beat_map):
                continue
            beat_row = beat_map.iloc[map_idx]
            rows.append(
                {
                    "group": group_label,
                    "seed": seed,
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
    if not rows:
        return app.pd.DataFrame(
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
    return (
        app.pd.DataFrame(rows)
        .sort_values(["measure", "beat_in_measure", "group", "beat_idx"])
        .reset_index(drop=True)
    )


def render_long_section_html(
    title: str,
    subtitle: str,
    xml_text: str,
    legend: str,
    container_id: str,
) -> str:
    if not xml_text.lstrip().startswith("<?xml"):
        xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
    xml_b64 = base64.b64encode(xml_text.encode("utf-8")).decode("ascii")
    return f"""
    <section class="score-section">
      <div class="header">
        <h1>{title}</h1>
        {'<div class="subtitle">' + subtitle + '</div>' if subtitle else ''}
        <div class="legend">{legend}</div>
      </div>
      <div id="{container_id}" class="score-box"></div>
      <div id="{container_id}-status" class="status">Loading score...</div>
    </section>
    <script>
      window.__scoreJobs = window.__scoreJobs || [];
      window.__scoreJobs.push({{
        containerId: {repr(container_id)},
        statusId: {repr(container_id + "-status")},
        xmlText: atob({repr(xml_b64)}),
      }});
    </script>
    """


def wrap_long_document(body_html: str, page_height_mm: float, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    @page {{
      size: {PAGE_WIDTH_MM}mm {page_height_mm:.2f}mm;
      margin: 0;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: white;
      width: {PAGE_WIDTH_MM}mm;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111;
    }}
    .score-section {{
      width: {PAGE_WIDTH_MM}mm;
      padding: 3mm 0 5mm 0;
      break-after: auto;
      page-break-after: auto;
    }}
    h1 {{
      margin: 0 0 2mm 0;
      font-size: 14pt;
      font-weight: 600;
    }}
    .subtitle {{
      font-size: 10pt;
      color: #555;
      margin-bottom: 2mm;
    }}
    .legend {{
      margin-bottom: 3mm;
      line-height: 1.5;
    }}
    .score-box {{
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      background: white;
      min-height: 200px;
    }}
    .score-box svg {{
      width: {PAGE_WIDTH_MM}mm;
      height: auto;
      display: block;
    }}
    .status {{
      display: none;
    }}
  </style>
</head>
<body>
{body_html}
<script>
  function loadScript(src) {{
    return new Promise((resolve, reject) => {{
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    }});
  }}

  async function ensureOSMD() {{
    if (window.opensheetmusicdisplay) return;
    await loadScript("https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.2/build/opensheetmusicdisplay.min.js");
  }}

  async function renderAll() {{
    await ensureOSMD();
    for (const job of (window.__scoreJobs || [])) {{
      const status = document.getElementById(job.statusId);
      try {{
        const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(job.containerId, {{
          autoResize: false,
          backend: "svg",
          drawingParameters: "compacttight",
          drawPartNames: false,
        }});
        await osmd.load(job.xmlText);
        osmd.render();
        status.textContent = "Loaded";
      }} catch (err) {{
        status.style.display = "block";
        status.textContent = "Render failed: " + err;
        status.style.color = "#b00020";
      }}
    }}
    window.__scoresRendered = true;
  }}

  renderAll();
</script>
</body>
</html>
"""


def print_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(CHROME_BIN),
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--virtual-time-budget=20000",
        "--run-all-compositor-stages-before-draw",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not CHROME_BIN.exists():
        raise FileNotFoundError(f"Chrome binary not found: {CHROME_BIN}")

    app = load_app_module()
    paged_export = load_paged_export_module()
    group_order = args.groups or app.get_group_order("weighted_topdown_seed44")
    seed = int(args.seed)
    pieces = args.pieces or app.WEIGHTED_TOPDOWN_PIECES
    selection_dir = Path(args.breakpoint_selection_dir).expanduser().resolve() if args.breakpoint_selection_dir else None
    predicted_events_root = Path(args.predicted_events_root).expanduser().resolve() if args.predicted_events_root else None
    group_keep_fractions = json.loads(args.per_group_keep_json) if args.per_group_keep_json else None
    filename_suffix = ""
    if args.show_measure_number_in_label and args.hide_level_names_in_label and args.black_labels_with_random_points:
        filename_suffix = "_measure_only_black_random"
    elif args.show_measure_number_in_label and args.hide_level_names_in_label:
        filename_suffix = "_measure_only"
    elif args.show_measure_number_in_label:
        filename_suffix = "_with_measure"
    elif args.hide_level_names_in_label:
        filename_suffix = "_no_level"
    if args.number_labels:
        filename_suffix += "_numbered"

    combined_sections: list[str] = []
    combined_height_mm = 0.0
    for piece_id in pieces:
        if selection_dir is not None:
            raw_breakpoints = build_selected_breakpoint_table(app, piece_id, group_order, selection_dir, seed)
        elif predicted_events_root is not None:
            raw_breakpoints = build_predicted_events_breakpoint_table(
                app,
                piece_id,
                group_order,
                predicted_events_root,
                seed,
                group_keep_fractions=group_keep_fractions,
            )
        else:
            raw_breakpoints = app.build_weighted_breakpoint_table(piece_id, group_order, seed=seed)
        breakpoints = app.suppress_lower_breakpoints_within_tolerance(
            raw_breakpoints,
            priority_order=[group for group in app.WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in group_order],
            tolerance=max(int(args.merge_tolerance), 0),
        )
        display_groups = list(group_order)
        display_group_specs = {group: dict(spec) for group, spec in app.WEIGHTED_TOPDOWN_GROUP_SPECS.items()}
        if args.display_l5plus6_as_l5:
            breakpoints = breakpoints.copy()
            breakpoints["group"] = breakpoints["group"].replace({"L5+6": "L5"})
            display_groups = ["L5" if group == "L5+6" else group for group in display_groups]
            display_group_specs["L5"] = {**display_group_specs["L5+6"], "slug": "level5_boundary"}
        if args.black_labels_with_random_points:
            for spec in display_group_specs.values():
                spec["color"] = "#000000"
            random_breakpoints = app.build_random_breakpoints(piece_id, breakpoints, count=10, tolerance=2)
            if not random_breakpoints.empty:
                breakpoints = app.pd.concat([breakpoints, random_breakpoints], ignore_index=True)
                display_groups.append(app.RANDOM_DISPLAY_GROUP)
                display_group_specs[app.RANDOM_DISPLAY_GROUP] = {"slug": "random_display", "color": "#000000"}
            breakpoints = breakpoints.sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)
        if args.number_labels:
            xml_text = build_numbered_annotated_musicxml(
                app,
                app.get_xml_path(piece_id),
                breakpoints,
                display_group_specs,
            )
        else:
            xml_text = app.build_annotated_musicxml(
                app.get_xml_path(piece_id),
                breakpoints,
                display_groups,
                display_group_specs,
                show_measure_in_label=args.show_measure_number_in_label,
                hide_group_in_label=args.hide_level_names_in_label,
            )
        counts = {str(group): int(count) for group, count in breakpoints.groupby("group").size().items()}
        legend = paged_export.legend_html(counts, display_group_specs, display_groups)
        section_html = render_long_section_html(piece_id, args.subtitle, xml_text, legend, f"score-{piece_id}")
        piece_height_mm = estimate_piece_height_mm(app, piece_id)
        combined_sections.append(section_html)
        combined_height_mm += piece_height_mm

        output_tag = args.output_tag or "weighted_topdown_seed44_merge56"
        long_html = wrap_long_document(
            section_html,
            piece_height_mm,
            f"{piece_id} {output_tag} Long Score PDF",
        )
        long_html_path = HTML_DIR / f"{piece_id}_{output_tag}{filename_suffix}_long.html"
        long_pdf_path = PDF_DIR / f"{piece_id}_{output_tag}{filename_suffix}_long.pdf"
        write_text(long_html_path, long_html)
        print_html_to_pdf(long_html_path, long_pdf_path)

    output_tag = args.output_tag or "weighted_topdown_seed44_merge56"
    combined_html = wrap_long_document(
        "\n".join(combined_sections),
        max(combined_height_mm, 300.0),
        f"{output_tag} Long Score PDF",
    )
    combined_html_path = HTML_DIR / f"{output_tag}_all{filename_suffix}_long.html"
    combined_pdf_path = PDF_DIR / f"{output_tag}_all{filename_suffix}_long.pdf"
    write_text(combined_html_path, combined_html)
    print_html_to_pdf(combined_html_path, combined_pdf_path)

    print(str(PDF_DIR))


if __name__ == "__main__":
    main()
