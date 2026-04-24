from __future__ import annotations

import argparse
import base64
import importlib.util
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
    group_order = app.get_group_order("weighted_topdown_seed44")
    seed = 44
    pieces = app.WEIGHTED_TOPDOWN_PIECES
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
        raw_breakpoints = app.build_weighted_breakpoint_table(piece_id, group_order, seed=seed)
        breakpoints = app.suppress_lower_breakpoints_within_tolerance(
            raw_breakpoints,
            priority_order=[group for group in app.WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in group_order],
            tolerance=1,
        )
        display_groups = list(group_order)
        display_group_specs = {group: dict(spec) for group, spec in app.WEIGHTED_TOPDOWN_GROUP_SPECS.items()}
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
        section_html = render_long_section_html(piece_id, "", xml_text, legend, f"score-{piece_id}")
        piece_height_mm = estimate_piece_height_mm(app, piece_id)
        combined_sections.append(section_html)
        combined_height_mm += piece_height_mm

        long_html = wrap_long_document(
            section_html,
            piece_height_mm,
            f"{piece_id} Weighted Topdown Seed44 Long Score PDF",
        )
        long_html_path = HTML_DIR / f"{piece_id}_weighted_topdown_seed44_merge56{filename_suffix}_long.html"
        long_pdf_path = PDF_DIR / f"{piece_id}_weighted_topdown_seed44_merge56{filename_suffix}_long.pdf"
        write_text(long_html_path, long_html)
        print_html_to_pdf(long_html_path, long_pdf_path)

    combined_html = wrap_long_document(
        "\n".join(combined_sections),
        max(combined_height_mm, 300.0),
        "Weighted Topdown Seed44 Long Score PDF",
    )
    combined_html_path = HTML_DIR / f"weighted_topdown_seed44_merge56_all{filename_suffix}_long.html"
    combined_pdf_path = PDF_DIR / f"weighted_topdown_seed44_merge56_all{filename_suffix}_long.pdf"
    write_text(combined_html_path, combined_html)
    print_html_to_pdf(combined_html_path, combined_pdf_path)

    print(str(PDF_DIR))


if __name__ == "__main__":
    main()
