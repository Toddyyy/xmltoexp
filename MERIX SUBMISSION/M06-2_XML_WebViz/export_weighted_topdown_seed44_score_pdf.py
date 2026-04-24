from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app.py"
EXPORT_DIR = SCRIPT_DIR / "exports" / "score_pdf"
HTML_DIR = EXPORT_DIR / "html"
PDF_DIR = EXPORT_DIR / "pdf"
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def load_app_module():
    spec = importlib.util.spec_from_file_location("webviz_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load app module from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legend_html(group_counts: dict[str, int], group_specs: dict[str, dict[str, str]], group_order: list[str]) -> str:
    chips: list[str] = []
    for group_label in group_order:
        count = int(group_counts.get(group_label, 0))
        color = group_specs[group_label]["color"]
        chips.append(
            f'<span style="display:inline-block;background:{color};color:white;'
            f'border-radius:999px;padding:3px 10px;font-size:12px;margin-right:8px;margin-bottom:6px;">'
            f"{group_label}: {count}</span>"
        )
    return "".join(chips)


def render_section_html(
    title: str,
    subtitle: str,
    xml_text: str,
    legend: str,
    container_id: str,
    breakpoint_positions: list[dict] | None = None,
) -> str:
    if not xml_text.lstrip().startswith("<?xml"):
        xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
    xml_b64 = base64.b64encode(xml_text.encode("utf-8")).decode("ascii")
    xml_b64_js = json.dumps(xml_b64)
    title_js = json.dumps(title)
    subtitle_js = json.dumps(subtitle)
    legend_js = json.dumps(legend)
    container_id_js = json.dumps(container_id)
    return f"""
    <section class="score-section">
      <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
        <div class="legend">{legend}</div>
      </div>
      <div id="{container_id}" class="score-box"></div>
      <div id="{container_id}-status" class="status">Loading score...</div>
    </section>
    <script>
      window.__scoreJobs = window.__scoreJobs || [];
      window.__scoreJobs.push({{
        containerId: {container_id_js},
        statusId: {json.dumps(container_id + "-status")},
        title: {title_js},
        subtitle: {subtitle_js},
        xmlText: atob({xml_b64_js}),
        legendHtml: {legend_js},
        breakpointPositions: {json.dumps(breakpoint_positions or [])},
      }});
    </script>
    """


def wrap_document(body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Weighted Topdown Seed44 Score PDF</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 8mm;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111;
      background: white;
    }}
    .score-section {{
      page-break-after: always;
      break-after: page;
      padding: 4mm 0 2mm 0;
    }}
    .score-section:last-child {{
      page-break-after: auto;
      break-after: auto;
    }}
    .header h1 {{
      margin: 0 0 4px 0;
      font-size: 20px;
    }}
    .subtitle {{
      font-size: 12px;
      color: #555;
      margin-bottom: 8px;
    }}
    .legend {{
      margin-bottom: 10px;
      line-height: 1.6;
    }}
    .score-box {{
      position: relative;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      background: white;
      min-height: 400px;
    }}
    .score-pages {{
      display: flex;
      flex-direction: column;
      gap: 10mm;
    }}
    .score-page {{
      break-after: page;
      page-break-after: always;
    }}
    .score-page:last-child {{
      break-after: auto;
      page-break-after: auto;
    }}
    .score-page svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .breakpoint-overlay {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 5;
    }}
    .breakpoint-line {{
      position: absolute;
      width: 2px;
      border-radius: 999px;
      opacity: 0.95;
      transform: translateX(-50%);
      background: #d40000;
    }}
    .status {{
      margin-top: 8px;
      font-size: 11px;
      color: #666;
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

  function chunkArray(items, size) {{
    const chunks = [];
    for (let i = 0; i < items.length; i += size) {{
      chunks.push(items.slice(i, i + size));
    }}
    return chunks;
  }}

  function paginateRenderedScore(containerId, systemsPerPage) {{
    const host = document.getElementById(containerId);
    const sourceSvg = host.querySelector("svg");
    if (!sourceSvg) return 0;

    const stafflines = Array.from(sourceSvg.querySelectorAll("g.staffline"));
    if (!stafflines.length) return 1;

    const viewBoxParts = (sourceSvg.getAttribute("viewBox") || "").trim().split(/\\s+/).map(Number);
    const width = viewBoxParts[2] || Number(sourceSvg.getAttribute("width")) || 1000;
    const height = viewBoxParts[3] || Number(sourceSvg.getAttribute("height")) || 1000;
    const originX = viewBoxParts[0] || 0;
    const originY = viewBoxParts[1] || 0;
    const svgMarkup = sourceSvg.innerHTML;
    const pages = chunkArray(stafflines, systemsPerPage);

    if (pages.length <= 1) {{
      sourceSvg.style.width = "100%";
      sourceSvg.style.height = "auto";
      return 1;
    }}

    const wrapper = document.createElement("div");
    wrapper.className = "score-pages";
    const margin = 28;

    for (const pageSystems of pages) {{
      const boxes = pageSystems.map(node => node.getBBox());
      const minY = Math.max(originY, Math.min(...boxes.map(box => box.y)) - margin);
      const maxY = Math.min(originY + height, Math.max(...boxes.map(box => box.y + box.height)) + margin);

      const page = document.createElement("div");
      page.className = "score-page";

      const pageSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      pageSvg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      pageSvg.setAttribute("viewBox", `${{originX}} ${{minY}} ${{width}} ${{maxY - minY}}`);
      pageSvg.setAttribute("preserveAspectRatio", "xMinYMin meet");
      pageSvg.innerHTML = svgMarkup;

      page.appendChild(pageSvg);
      wrapper.appendChild(page);
    }}

    host.innerHTML = "";
    host.appendChild(wrapper);
    return pages.length;
  }}

  function decorateBreakpointLines(containerId, breakpointPositions) {{
    if (!breakpointPositions || !breakpointPositions.length) return;
    const host = document.getElementById(containerId);
    const svgs = Array.from(host.querySelectorAll("svg"));
    if (!svgs.length) return;

    for (const existing of Array.from(host.querySelectorAll(".breakpoint-overlay"))) {{
      existing.remove();
    }}

    const hostRect = host.getBoundingClientRect();
    const overlay = document.createElement("div");
    overlay.className = "breakpoint-overlay";
    host.appendChild(overlay);

    for (const svg of svgs) {{
      const measureNodes = Array.from(svg.querySelectorAll("g.vf-measure"));
      const measureMap = new Map();
      for (const node of measureNodes) {{
        const key = String(node.id);
        if (!measureMap.has(key)) {{
          measureMap.set(key, []);
        }}
        measureMap.get(key).push(node);
      }}

      for (const bp of breakpointPositions) {{
        const key = String(bp.measure_index);
        const targets = measureMap.get(key) || [];
        for (const node of targets) {{
          const rect = node.getBoundingClientRect();
          const beatsInMeasure = Math.max(1, Number(bp.beats_in_measure || 1));
          const beatInMeasure = Math.max(1, Number(bp.beat_in_measure || 1));
          const fraction = Math.min(0.94, Math.max(0.06, (beatInMeasure - 0.5) / beatsInMeasure));
          const left = rect.left - hostRect.left + rect.width * fraction;
          const lineHeight = Math.max(22, rect.height * 0.62);
          const top = rect.top - hostRect.top + (rect.height - lineHeight) / 2;
          const line = document.createElement("div");
          line.className = "breakpoint-line";
          line.style.left = String(left) + "px";
          line.style.top = String(top) + "px";
          line.style.height = String(lineHeight) + "px";
          overlay.appendChild(line);
        }}
      }}
    }}
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
        const pageCount = paginateRenderedScore(job.containerId, 7);
        await new Promise(resolve => setTimeout(resolve, 120));
        decorateBreakpointLines(job.containerId, job.breakpointPositions);
        status.textContent = "Loaded (" + pageCount + " page" + (pageCount === 1 ? "" : "s") + ")";
      }} catch (err) {{
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    if not CHROME_BIN.exists():
        raise FileNotFoundError(f"Chrome binary not found: {CHROME_BIN}")

    app = load_app_module()
    group_order = app.get_group_order("weighted_topdown_seed44")
    seed = 44
    pieces = app.WEIGHTED_TOPDOWN_PIECES

    combined_sections: list[str] = []

    for piece_id in pieces:
        raw_breakpoints = app.build_weighted_breakpoint_table(piece_id, group_order, seed=seed)
        breakpoints = app.suppress_lower_breakpoints_within_tolerance(
            raw_breakpoints,
            priority_order=[group for group in app.WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in group_order],
            tolerance=1,
        )
        summary_df = app.build_weighted_summary_table(piece_id, group_order, breakpoints, seed=seed)
        xml_text = app.build_annotated_musicxml(
            app.get_xml_path(piece_id),
            breakpoints,
            group_order,
            app.WEIGHTED_TOPDOWN_GROUP_SPECS,
        )
        counts = {row["group"]: int(row["events"]) for _, row in summary_df.iterrows()}
        legend = legend_html(counts, app.WEIGHTED_TOPDOWN_GROUP_SPECS, group_order)
        subtitle = "Merged L5+6 clean outer | Seed44 | High-level priority with +/-1 beat lower-level suppression"
        breakpoint_positions = app.build_breakpoint_positions(piece_id, breakpoints, app.WEIGHTED_TOPDOWN_GROUP_SPECS)
        section_html = render_section_html(
            piece_id,
            subtitle,
            xml_text,
            legend,
            f"score-{piece_id}",
            breakpoint_positions=breakpoint_positions,
        )
        combined_sections.append(section_html)

        piece_html_path = HTML_DIR / f"{piece_id}_weighted_topdown_seed44_merge56.html"
        piece_pdf_path = PDF_DIR / f"{piece_id}_weighted_topdown_seed44_merge56.pdf"
        write_text(piece_html_path, wrap_document(section_html))
        print_html_to_pdf(piece_html_path, piece_pdf_path)

    combined_html_path = HTML_DIR / "weighted_topdown_seed44_merge56_all.html"
    combined_pdf_path = PDF_DIR / "weighted_topdown_seed44_merge56_all.pdf"
    write_text(combined_html_path, wrap_document("\n".join(combined_sections)))
    print_html_to_pdf(combined_html_path, combined_pdf_path)

    print(str(PDF_DIR))


if __name__ == "__main__":
    main()
