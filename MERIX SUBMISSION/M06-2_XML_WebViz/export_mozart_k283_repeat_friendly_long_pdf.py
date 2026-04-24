from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app.py"
PAGED_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_pdf.py"
LONG_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_long_pdf.py"
EXPORT_DIR = SCRIPT_DIR / "exports" / "score_pdf_long"
HTML_DIR = EXPORT_DIR / "html"
PDF_DIR = EXPORT_DIR / "pdf"
PAGE_WIDTH_MM = 200.0


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate_piece_height_mm(app, xml_path: Path) -> float:
    score_height_px = float(app.estimate_score_height_from_xml(str(xml_path)))
    return max(300.0, score_height_px * 25.4 / 96.0 * 1.45 + 45.0)


def render_repeat_friendly_document(body_html: str, page_height_mm: float, title: str) -> str:
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
      padding: 5mm 0 8mm 0;
    }}
    h1 {{
      margin: 0 0 2mm 0;
      font-size: 14pt;
      font-weight: 600;
    }}
    .score-box {{
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 18px 24px;
      background: white;
      min-height: 200px;
    }}
    .score-box svg {{
      width: {PAGE_WIDTH_MM - 12:.2f}mm;
      height: auto;
      display: block;
    }}
    .legend {{
      margin-bottom: 3mm;
      line-height: 1.5;
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
          drawingParameters: "default",
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


def main() -> None:
    app = _load_module(APP_PATH, "webviz_app")
    paged_export = _load_module(PAGED_EXPORT_PATH, "webviz_paged_export")
    long_export = _load_module(LONG_EXPORT_PATH, "webviz_long_export")

    piece_id = "mozart_k283_i"
    group_order = app.get_group_order("new_scores_seed44")
    display_group_specs = {group: dict(spec) for group, spec in app.WEIGHTED_TOPDOWN_GROUP_SPECS.items()}

    raw_breakpoints = app.build_new_score_breakpoint_table(piece_id, group_order)
    breakpoints = app.suppress_lower_breakpoints_within_tolerance(
        raw_breakpoints,
        priority_order=[group for group in app.WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in group_order],
        tolerance=1,
    )

    source_frame = app.load_new_score_events(piece_id, group_order[0])
    if source_frame.empty or "source_score_path" not in source_frame.columns:
        raise FileNotFoundError(f"Missing source score path for {piece_id}")
    xml_path = Path(source_frame["source_score_path"].dropna().iloc[0])

    xml_text = app.build_annotated_musicxml(
        xml_path,
        breakpoints,
        group_order,
        display_group_specs,
        show_measure_in_label=False,
        hide_group_in_label=False,
    )

    counts = {str(group): int(count) for group, count in breakpoints.groupby("group").size().items()}
    legend = paged_export.legend_html(counts, display_group_specs, group_order)
    title = app.NEW_SCORE_LABELS.get(piece_id, piece_id)
    section_html = long_export.render_long_section_html(title, "", xml_text, legend, f"score-{piece_id}-repeat-friendly")
    piece_height_mm = estimate_piece_height_mm(app, xml_path)
    html = render_repeat_friendly_document(section_html, piece_height_mm, f"{title} repeat-friendly long score PDF")

    html_path = HTML_DIR / f"{piece_id}_repeat_friendly_long.html"
    pdf_path = PDF_DIR / f"{piece_id}_repeat_friendly_long.pdf"
    long_export.write_text(html_path, html)
    long_export.print_html_to_pdf(html_path, pdf_path)
    print(str(pdf_path))


if __name__ == "__main__":
    main()
