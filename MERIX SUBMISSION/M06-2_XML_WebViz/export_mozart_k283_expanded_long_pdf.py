from __future__ import annotations

import importlib.util
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app.py"
PAGED_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_pdf.py"
LONG_EXPORT_PATH = SCRIPT_DIR / "export_weighted_topdown_seed44_score_long_pdf.py"
EXPORT_DIR = SCRIPT_DIR / "exports" / "score_pdf_long"
HTML_DIR = EXPORT_DIR / "html"
PDF_DIR = EXPORT_DIR / "pdf"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_measure_occurrence(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("beat_idx").reset_index(drop=True).copy()
    occurrences: list[int] = []
    per_measure_seen: dict[int, int] = {}
    prev_key: tuple[int, int] | None = None
    current_occurrence = 0
    for row in frame.itertuples(index=False):
        key = (int(row.measure_number), int(row.beat_in_measure))
        measure_no = int(row.measure_number)
        if key != prev_key and int(row.beat_in_measure) == 1:
            per_measure_seen[measure_no] = per_measure_seen.get(measure_no, 0) + 1
            current_occurrence = per_measure_seen[measure_no]
        elif measure_no not in per_measure_seen:
            per_measure_seen[measure_no] = 1
            current_occurrence = 1
        occurrences.append(current_occurrence)
        prev_key = key
    frame["measure_occurrence"] = occurrences
    return frame


def build_expanded_annotated_musicxml(
    app,
    expanded_xml_path: Path,
    breakpoints: pd.DataFrame,
    beat_frame: pd.DataFrame,
    selected_groups: list[str],
    group_specs: dict[str, dict[str, str]],
    *,
    show_measure_in_label: bool = False,
    hide_group_in_label: bool = False,
) -> str:
    root = app._load_musicxml_root(expanded_xml_path)
    app.collapse_globally_empty_staves(root)

    beat_frame = _add_measure_occurrence(beat_frame)

    beat_with_occ = breakpoints.merge(
        beat_frame[["beat_idx", "measure_occurrence"]],
        on="beat_idx",
        how="left",
        validate="many_to_one",
    )

    by_measure_occ_group: dict[tuple[int, int], dict[str, list[dict[str, int | float]]]] = {}
    for row in beat_with_occ.itertuples(index=False):
        occ = int(getattr(row, "measure_occurrence", 1) or 1)
        key = (int(row.measure), occ)
        by_measure_occ_group.setdefault(key, {}).setdefault(str(row.group), []).append(
            {
                "beat_idx": int(row.beat_idx),
                "beat_in_measure": int(row.beat_in_measure),
                "detector_score": float(row.detector_score),
            }
        )

    seen_per_measure: dict[int, int] = {}
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            measure_no = app._as_measure_number(measure.attrib.get("number", ""))
            if measure_no is None:
                continue
            seen_per_measure[measure_no] = seen_per_measure.get(measure_no, 0) + 1
            measure_occ = seen_per_measure[measure_no]
            bucket = by_measure_occ_group.get((measure_no, measure_occ))
            if not bucket:
                continue

            insertions = []
            for group_label in selected_groups:
                items = bucket.get(group_label, [])
                if not items:
                    continue
                color = group_specs.get(group_label, {"color": "#000000"})["color"]
                beat_text = ",".join(f"b{item['beat_in_measure']}" for item in items[:4])
                if len(items) > 4:
                    beat_text += ",..."
                label_parts = []
                if not hide_group_in_label:
                    label_parts.append(str(group_label))
                if show_measure_in_label:
                    label_parts.append(f"m{measure_no}")
                label_parts.append(beat_text)
                label_text = ": ".join(label_parts[:2]) if len(label_parts) >= 2 else label_parts[0]
                if len(label_parts) > 2:
                    label_text = label_text + " " + " ".join(label_parts[2:])
                direction = ET.Element("direction", {"placement": "above"})
                direction_type = ET.SubElement(direction, "direction-type")
                ET.SubElement(
                    direction_type,
                    "words",
                    {"color": color, "font-weight": "bold", "font-size": "10"},
                ).text = label_text
                insertions.append(direction)

            for direction in reversed(insertions):
                measure.insert(0, direction)

    return ET.tostring(root, encoding="unicode")


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
    beat_frame = app.load_new_score_beat_frame(piece_id)
    random_breakpoints = app.build_random_breakpoints_from_frame(
        beat_frame,
        breakpoints,
        key=f"{piece_id}:expanded_random:10",
        count=10,
        tolerance=2,
    )
    display_groups = list(group_order)
    if not random_breakpoints.empty:
        breakpoints = app.pd.concat([breakpoints, random_breakpoints], ignore_index=True)
        breakpoints = breakpoints.sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)
        display_group_specs[app.RANDOM_DISPLAY_GROUP] = {"slug": "random_display", "color": "#000000"}
        display_groups.append(app.RANDOM_DISPLAY_GROUP)

    with tempfile.TemporaryDirectory(prefix="k283_expand_") as tmpdir:
        expanded_xml_path = Path(tmpdir) / "mozart_k283_i_expanded.musicxml"
        score = music21.converter.parse(str(xml_path))
        expanded = score.expandRepeats()
        expanded.write("musicxml", fp=str(expanded_xml_path))

        xml_text = build_expanded_annotated_musicxml(
            app,
            expanded_xml_path,
            breakpoints,
            beat_frame,
            display_groups,
            display_group_specs,
            show_measure_in_label=True,
            hide_group_in_label=False,
        )
        counts = {str(group): int(count) for group, count in breakpoints.groupby("group").size().items()}
        legend = paged_export.legend_html(counts, display_group_specs, display_groups)
        title = "Mozart K.283 I (Expanded Repeats)"
        section_html = long_export.render_long_section_html(title, "", xml_text, legend, "score-k283-expanded")
        piece_height_mm = max(300.0, float(app.estimate_score_height_from_xml(str(expanded_xml_path))) * 25.4 / 96.0 * 1.35 + 35.0)
        long_html = long_export.wrap_long_document(
            section_html,
            piece_height_mm,
            f"{title} Long Score PDF",
        )

        long_html_path = HTML_DIR / "mozart_k283_i_expanded_repeats_with_measure_random_long.html"
        long_pdf_path = PDF_DIR / "mozart_k283_i_expanded_repeats_with_measure_random_long.pdf"
        long_export.write_text(long_html_path, long_html)
        long_export.print_html_to_pdf(long_html_path, long_pdf_path)
        print(str(long_pdf_path))


if __name__ == "__main__":
    main()
