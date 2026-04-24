from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app.py"
LONG_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_long_pdf.py"
PAGED_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_pdf.py"
EXPORT_DIR = SCRIPT_DIR / "exports" / "score_pdf_long"
HTML_DIR = EXPORT_DIR / "html"
PDF_DIR = EXPORT_DIR / "pdf"
PAGE_WIDTH_MM = 190.0


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate_piece_height_mm(app, xml_path: Path) -> float:
    score_height_px = float(app.estimate_score_height_from_xml(str(xml_path)))
    return max(300.0, score_height_px * 25.4 / 96.0 * 1.35 + 35.0)


def main() -> None:
    app = _load_module(APP_PATH, "webviz_app")
    long_export = _load_module(LONG_EXPORT_PATH, "webviz_long_export")
    paged_export = _load_module(PAGED_EXPORT_PATH, "webviz_paged_export")

    group_order = app.get_group_order("new_scores_seed44")
    display_group_specs = {group: dict(spec) for group, spec in app.WEIGHTED_TOPDOWN_GROUP_SPECS.items()}
    seed = 44
    filename_suffix = "_with_measure_long"

    combined_sections: list[str] = []
    combined_height_mm = 0.0

    for piece_id in app.NEW_SCORE_PIECES:
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
            show_measure_in_label=True,
            hide_group_in_label=False,
        )

        counts = {str(group): int(count) for group, count in breakpoints.groupby("group").size().items()}
        legend = paged_export.legend_html(counts, display_group_specs, group_order)
        title = app.NEW_SCORE_LABELS.get(piece_id, piece_id)
        section_html = long_export.render_long_section_html(title, "", xml_text, legend, f"score-{piece_id}")
        piece_height_mm = estimate_piece_height_mm(app, xml_path)
        combined_sections.append(section_html)
        combined_height_mm += piece_height_mm

        long_html = long_export.wrap_long_document(
            section_html,
            piece_height_mm,
            f"{title} New Scores Seed44 Long Score PDF",
        )
        long_html_path = HTML_DIR / f"{piece_id}_new_scores_seed44{filename_suffix}.html"
        long_pdf_path = PDF_DIR / f"{piece_id}_new_scores_seed44{filename_suffix}.pdf"
        long_export.write_text(long_html_path, long_html)
        long_export.print_html_to_pdf(long_html_path, long_pdf_path)

    combined_html = long_export.wrap_long_document(
        "\n".join(combined_sections),
        max(combined_height_mm, 300.0),
        "New Scores Seed44 Long Score PDF",
    )
    combined_html_path = HTML_DIR / f"new_scores_seed44_all{filename_suffix}.html"
    combined_pdf_path = PDF_DIR / f"new_scores_seed44_all{filename_suffix}.pdf"
    long_export.write_text(combined_html_path, combined_html)
    long_export.print_html_to_pdf(combined_html_path, combined_pdf_path)

    print(str(PDF_DIR))


if __name__ == "__main__":
    main()
