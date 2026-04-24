from __future__ import annotations

import base64
import json
import random
import re
import zipfile
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


def _load_musicxml_bytes(xml_path: Path) -> bytes:
    suffix = xml_path.suffix.lower()
    if suffix in {".xml", ".musicxml"}:
        return xml_path.read_bytes()
    if suffix != ".mxl":
        raise ValueError(f"Unsupported score format: {xml_path.suffix}")

    with zipfile.ZipFile(xml_path) as archive:
        rootfile_path: str | None = None
        try:
            container_root = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container_root.find(".//{*}rootfile")
            if rootfile is not None:
                rootfile_path = rootfile.attrib.get("full-path")
        except KeyError:
            rootfile_path = None

        if rootfile_path is None:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml")) and not name.startswith("META-INF/")
            ]
            if not candidates:
                raise FileNotFoundError(f"No MusicXML rootfile found inside {xml_path.name}")
            rootfile_path = candidates[0]

        return archive.read(rootfile_path)


def _load_musicxml_root(xml_path: Path) -> ET.Element:
    return ET.fromstring(_load_musicxml_bytes(xml_path))


def collapse_globally_empty_staves(root: ET.Element) -> None:
    declared_staff_count = 1
    for staves_el in root.findall(".//attributes/staves"):
        try:
            declared_staff_count = max(declared_staff_count, int((staves_el.text or "1").strip()))
        except Exception:
            continue
    declared_staffs = set(range(1, declared_staff_count + 1))
    active_staffs: set[int] = set()
    for note in root.findall(".//note"):
        if note.find("rest") is not None:
            continue
        try:
            active_staffs.add(int((note.findtext("staff") or "1").strip()))
        except Exception:
            active_staffs.add(1)

    empty_staffs = declared_staffs - active_staffs
    if not empty_staffs:
        return

    kept_staffs = sorted(active_staffs)
    if not kept_staffs:
        return
    remap = {old_staff: idx + 1 for idx, old_staff in enumerate(kept_staffs)}

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            new_children: list[ET.Element] = []
            pending_removed_staff_note = False
            for child in list(measure):
                if child.tag == "note":
                    try:
                        staff_no = int((child.findtext("staff") or "1").strip())
                    except Exception:
                        staff_no = 1
                    if staff_no in empty_staffs:
                        pending_removed_staff_note = True
                        continue
                    staff_el = child.find("staff")
                    if staff_el is not None:
                        staff_el.text = str(remap.get(staff_no, staff_no))
                    new_children.append(child)
                    pending_removed_staff_note = False
                    continue

                if child.tag == "backup":
                    if pending_removed_staff_note:
                        pending_removed_staff_note = False
                        continue
                    new_children.append(child)
                    continue

                if child.tag == "attributes":
                    staves_el = child.find("staves")
                    if staves_el is not None:
                        staves_el.text = str(len(kept_staffs))
                    for clef in list(child.findall("clef")):
                        try:
                            clef_staff = int(clef.get("number", "1"))
                        except Exception:
                            clef_staff = 1
                        if clef_staff in empty_staffs:
                            child.remove(clef)
                        else:
                            clef.set("number", str(remap.get(clef_staff, clef_staff)))
                    new_children.append(child)
                    continue

                if child.tag == "print":
                    for staff_layout in list(child.findall("staff-layout")):
                        try:
                            layout_staff = int(staff_layout.get("number", "1"))
                        except Exception:
                            layout_staff = 1
                        if layout_staff in empty_staffs:
                            child.remove(staff_layout)
                        else:
                            staff_layout.set("number", str(remap.get(layout_staff, layout_staff)))
                    new_children.append(child)
                    continue

                if child.tag == "direction":
                    staff_el = child.find("staff")
                    if staff_el is not None:
                        try:
                            direction_staff = int((staff_el.text or "1").strip())
                        except Exception:
                            direction_staff = 1
                        staff_el.text = str(remap.get(direction_staff, 1))
                    new_children.append(child)
                    continue

                for staff_el in child.findall(".//staff"):
                    try:
                        staff_no = int((staff_el.text or "1").strip())
                    except Exception:
                        staff_no = 1
                    staff_el.text = str(remap.get(staff_no, 1))
                new_children.append(child)

            measure[:] = new_children


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "MazurkaBL-master").exists() and (candidate / "MERIX SUBMISSION").exists():
            return candidate
    raise FileNotFoundError("Unable to locate project root")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SUPPORTED_PIECES = ["M06-1", "M06-2", "M06-3", "M17-1", "M30-1"]
WEIGHTED_TOPDOWN_PIECES = ["M06-1", "M06-2", "M06-3"]
NEW_SCORE_PIECES = [
    "beethoven_pathetique_ii",
    "mozart_k283_i",
    "mozart_k331_i",
]
NEW_SCORE_LABELS = {
    "beethoven_pathetique_ii": "Beethoven Pathetique II",
    "mozart_k283_i": "Mozart K.283 I",
    "mozart_k331_i": "Mozart K.331 I",
}
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
    "L1+": {"slug": "level1plus_boundary", "color": "#0a6cff"},
    "L2+": {"slug": "level2plus_boundary", "color": "#00a35c"},
    "L3+": {"slug": "level3plus_boundary", "color": "#ff8a00"},
    "L4+": {"slug": "level4plus_boundary", "color": "#7b1fa2"},
    "L5+6": {"slug": "level56_boundary", "color": "#c2185b"},
}
WEIGHTED_TOPDOWN_DISPLAY_PRIORITY = ["L5+6", "L4+", "L3+", "L2+", "L1+"]
RANDOM_DISPLAY_GROUP = "Random"


def get_group_specs(view_mode: str) -> dict[str, dict[str, str]]:
    if view_mode == "clean_outer_seeds":
        return CLEAN_GROUP_SPECS
    if view_mode in {"weighted_topdown_seed44", "new_scores_seed44"}:
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
        / f"MERIX SUBMISSION/Boundary_Restart/reports/clean_outer_test/weighted_topdown_merge56_{detector_target_slug}_seed{seed}/predicted_events.csv.gz"
    )


def get_weighted_summary_path(detector_target_slug: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / f"MERIX SUBMISSION/Boundary_Restart/reports/clean_outer_test/weighted_topdown_merge56_{detector_target_slug}_seed{seed}/summary.json"
    )


def get_new_score_root() -> Path:
    return PROJECT_ROOT / "MERIX SUBMISSION/Boundary_Restart/reports/new_score_predictions_merge56_seed44"


def get_new_score_piece_dir(piece_id: str) -> Path:
    return get_new_score_root() / piece_id


def get_new_score_events_path(piece_id: str, group_label: str) -> Path:
    file_stem = group_label.replace("+", "plus").replace("/", "_")
    return get_new_score_piece_dir(piece_id) / f"{file_stem}_events.csv"


def get_new_score_predictions_path(piece_id: str, group_label: str) -> Path:
    file_stem = group_label.replace("+", "plus").replace("/", "_")
    return get_new_score_piece_dir(piece_id) / f"{file_stem}_predictions.csv.gz"


def get_new_score_beat_features_path(piece_id: str) -> Path:
    return get_new_score_piece_dir(piece_id) / "beat_features.csv.gz"


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
def estimate_score_height_from_xml(xml_path: str) -> int:
    xml_path = Path(xml_path)
    measures: set[int] = set()
    root = _load_musicxml_root(xml_path)
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            measure_no = _as_measure_number(measure.attrib.get("number", ""))
            if measure_no is not None:
                measures.add(measure_no)

    measure_count = len(measures)
    # Streamlit embeds the score in an iframe with a fixed pixel height. Give a
    # generous height based on measure count so the full score renders without
    # an inner scrollbar.
    return max(2200, min(7200, 500 + measure_count * 48))


def estimate_score_height(piece_id: str) -> int:
    return estimate_score_height_from_xml(str(get_xml_path(piece_id)))


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
def load_new_score_events(piece_id: str, group_label: str) -> pd.DataFrame:
    path = get_new_score_events_path(piece_id, group_label)
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "measure_number", "beat_in_measure", "detector_score"])
    frame = pd.read_csv(path)
    keep_cols = [
        col
        for col in [
            "beat_idx",
            "measure_number",
            "beat_in_measure",
            "detector_score",
            "matched_union",
            "frequency_target_at_beat",
            "matched_true_beat_idx",
            "match_offset",
            "source_score_path",
        ]
        if col in frame.columns
    ]
    return frame[keep_cols].copy()


@CACHE_DATA(show_spinner=False)
def load_new_score_beat_frame(piece_id: str) -> pd.DataFrame:
    path = get_new_score_beat_features_path(piece_id)
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "measure_number", "beat_in_measure"])
    return pd.read_csv(path, usecols=["beat_idx", "measure_number", "beat_in_measure"])


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


def map_new_score_events(events: pd.DataFrame) -> list[dict]:
    mapped: list[dict] = []
    for row in events.itertuples(index=False):
        if pd.isna(getattr(row, "measure_number", None)) or pd.isna(getattr(row, "beat_in_measure", None)):
            continue
        mapped.append(
            {
                "beat_idx": int(row.beat_idx),
                "measure": int(row.measure_number),
                "beat_in_measure": int(row.beat_in_measure),
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


def build_new_score_breakpoint_table(piece_id: str, selected_groups: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for group_label in selected_groups:
        mapped = map_new_score_events(load_new_score_events(piece_id, group_label))
        for item in mapped:
            rows.append(
                {
                    "group": group_label,
                    "seed": 44,
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


def build_random_breakpoints(
    piece_id: str,
    existing_breakpoints: pd.DataFrame,
    *,
    count: int = 10,
    tolerance: int = 2,
) -> pd.DataFrame:
    beat_map = load_beat_map(piece_id)
    existing_beats = sorted({int(value) for value in existing_breakpoints["beat_idx"].tolist()}) if not existing_breakpoints.empty else []
    candidate_beats = [
        idx for idx in range(len(beat_map)) if all(abs(idx - existing) > tolerance for existing in existing_beats)
    ]
    if not candidate_beats:
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

    rng = random.Random(f"{piece_id}:weighted_topdown_seed44:black_random:{count}")
    rng.shuffle(candidate_beats)
    selected: list[int] = []
    for beat_idx in candidate_beats:
        if any(abs(beat_idx - existing) <= tolerance for existing in selected):
            continue
        selected.append(int(beat_idx))
        if len(selected) >= count:
            break
    if not selected:
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
    selected = sorted(selected)
    rows = []
    for beat_idx in selected:
        beat_row = beat_map.iloc[beat_idx]
        rows.append(
            {
                "group": RANDOM_DISPLAY_GROUP,
                "seed": 44,
                "beat_idx": int(beat_idx),
                "measure": int(beat_row["measure_number"]),
                "beat_in_measure": int(beat_row["beat_number"]) + 1,
                "detector_score": 0.0,
                "matched_union": False,
                "frequency_target_at_beat": 0.0,
                "matched_true_beat_idx": None,
                "match_offset": None,
            }
        )
    return pd.DataFrame(rows).sort_values(["measure", "beat_in_measure", "beat_idx"]).reset_index(drop=True)


def build_random_breakpoints_from_frame(
    beat_frame: pd.DataFrame,
    existing_breakpoints: pd.DataFrame,
    *,
    key: str,
    count: int = 10,
    tolerance: int = 2,
) -> pd.DataFrame:
    beat_frame = beat_frame.sort_values("beat_idx").reset_index(drop=True)
    existing_beats = sorted({int(value) for value in existing_breakpoints["beat_idx"].tolist()}) if not existing_breakpoints.empty else []
    candidate_beats = [
        int(row.beat_idx)
        for row in beat_frame.itertuples(index=False)
        if all(abs(int(row.beat_idx) - existing) > tolerance for existing in existing_beats)
    ]
    if not candidate_beats:
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
    rng = random.Random(key)
    rng.shuffle(candidate_beats)
    selected: list[int] = []
    for beat_idx in candidate_beats:
        if any(abs(beat_idx - existing) <= tolerance for existing in selected):
            continue
        selected.append(int(beat_idx))
        if len(selected) >= count:
            break
    if not selected:
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
    rows = []
    frame_by_beat = beat_frame.set_index("beat_idx")
    for beat_idx in sorted(selected):
        beat_row = frame_by_beat.loc[beat_idx]
        rows.append(
            {
                "group": RANDOM_DISPLAY_GROUP,
                "seed": 44,
                "beat_idx": int(beat_idx),
                "measure": int(beat_row["measure_number"]),
                "beat_in_measure": int(beat_row["beat_in_measure"]),
                "detector_score": 0.0,
                "matched_union": False,
                "frequency_target_at_beat": 0.0,
                "matched_true_beat_idx": None,
                "match_offset": None,
            }
        )
    return pd.DataFrame(rows).sort_values(["measure", "beat_in_measure", "beat_idx"]).reset_index(drop=True)


def build_breakpoint_positions(
    piece_id: str,
    breakpoints: pd.DataFrame,
    group_specs: dict[str, dict[str, str]],
) -> list[dict]:
    if breakpoints.empty:
        return []
    beat_map = load_beat_map(piece_id)
    beats_per_measure = (
        beat_map.groupby("measure_number")["beat_number"].max().add(1).astype(int).to_dict()
    )
    positions: list[dict] = []
    for row in breakpoints.itertuples(index=False):
        measure_no = int(row.measure)
        positions.append(
            {
                "group": str(row.group),
                "measure_index": measure_no - 1,
                "measure": measure_no,
                "beat_in_measure": int(row.beat_in_measure),
                "beats_in_measure": int(beats_per_measure.get(measure_no, int(row.beat_in_measure))),
                "color": group_specs.get(str(row.group), {"color": "#000000"})["color"],
            }
        )
    return positions


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


def apply_display_clef_fixes(root: ET.Element, xml_path: Path) -> None:
    if xml_path.stem != "Mazurka06-3":
        return

    part = root.find("part")
    if part is None:
        return

    def ensure_attributes(measure: ET.Element) -> ET.Element:
        attrs = measure.find("attributes")
        if attrs is not None:
            return attrs
        attrs = ET.Element("attributes")
        insert_at = 0
        children = list(measure)
        while insert_at < len(children) and children[insert_at].tag == "print":
            insert_at += 1
        measure.insert(insert_at, attrs)
        return attrs

    def ensure_clef(measure: ET.Element, number: str, sign: str, line: str) -> None:
        attrs = ensure_attributes(measure)
        for clef in attrs.findall("clef"):
            if clef.get("number") == number:
                sign_el = clef.find("sign")
                line_el = clef.find("line")
                if sign_el is None:
                    sign_el = ET.SubElement(clef, "sign")
                if line_el is None:
                    line_el = ET.SubElement(clef, "line")
                sign_el.text = sign
                line_el.text = line
                return
        clef = ET.SubElement(attrs, "clef", {"number": number})
        ET.SubElement(clef, "sign").text = sign
        ET.SubElement(clef, "line").text = line

    upper_staff_measures = {5, 17, 21, 29, 33, 73, 76, 85, 89}
    lower_staff_measures = {54, 57}

    for measure in part.findall("measure"):
        measure_no = _as_measure_number(measure.attrib.get("number", ""))
        if measure_no is None:
            continue
        clefs = measure.findall(".//attributes/clef")
        if measure_no == 1 and len(clefs) >= 2:
            clefs[0].set("number", "1")
            clefs[1].set("number", "2")
            continue
        if measure_no == 9:
            ensure_clef(measure, "1", "G", "2")
        if measure_no in upper_staff_measures:
            for clef in clefs:
                clef.set("number", "1")
        if measure_no in lower_staff_measures:
            for clef in clefs:
                clef.set("number", "2")


def build_annotated_musicxml(
    xml_path: Path,
    breakpoints: pd.DataFrame,
    selected_groups: list[str],
    group_specs: dict[str, dict[str, str]],
    *,
    show_measure_in_label: bool = False,
    hide_group_in_label: bool = False,
) -> str:
    root = _load_musicxml_root(xml_path)
    collapse_globally_empty_staves(root)
    apply_display_clef_fixes(root, xml_path)

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
    <div id="status" style="font-size:13px; color:#555; margin-bottom:8px;">Loading score...</div>
    <div id="score" style="position: relative; background: white; border: 1px solid #ddd; border-radius: 12px; padding: 8px;"></div>
    <style>
      #score .score-pages {{
        display: flex;
        flex-direction: column;
        gap: 18px;
      }}
      #score .score-page {{
        background: white;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        break-after: page;
        page-break-after: always;
      }}
      #score .score-page:last-child {{
        border-bottom: none;
        break-after: auto;
        page-break-after: auto;
      }}
      #score .score-page svg {{
        width: 100%;
        height: auto;
        display: block;
      }}
    </style>
    <script>
      const status = document.getElementById("status");
      const xmlText = atob({xml_b64_js});
      const SYSTEMS_PER_PAGE = 7;

      function setStatus(text, isError=false) {{
        status.textContent = text;
        status.style.color = isError ? "#b00020" : "#555";
      }}

      function chunkArray(items, size) {{
        const chunks = [];
        for (let i = 0; i < items.length; i += size) {{
          chunks.push(items.slice(i, i + size));
        }}
        return chunks;
      }}

      function paginateRenderedScore() {{
        const host = document.getElementById("score");
        const sourceSvg = host.querySelector("svg");
        if (!sourceSvg) {{
          return 0;
        }}

        const stafflines = Array.from(sourceSvg.querySelectorAll("g.staffline"));
        if (!stafflines.length) {{
          return 1;
        }}

        const viewBoxParts = (sourceSvg.getAttribute("viewBox") || "").trim().split(/\\s+/).map(Number);
        const width = viewBoxParts[2] || Number(sourceSvg.getAttribute("width")) || 1000;
        const height = viewBoxParts[3] || Number(sourceSvg.getAttribute("height")) || 1000;
        const originX = viewBoxParts[0] || 0;
        const originY = viewBoxParts[1] || 0;
        const svgMarkup = sourceSvg.innerHTML;
        const pages = chunkArray(stafflines, SYSTEMS_PER_PAGE);

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

      function renderWithOSMD() {{
        try {{
          const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score", {{
            autoResize: false,
            backend: "svg",
            drawingParameters: "compacttight",
            drawPartNames: false,
          }});
          osmd.load(xmlText)
            .then(() => {{
              osmd.render();
              const pageCount = paginateRenderedScore();
              setStatus("Loaded (" + pageCount + " page" + (pageCount === 1 ? "" : "s") + ")");
            }})
            .catch(err => setStatus("OSMD parse failed: " + err, true));
        }} catch (err) {{
          setStatus("OSMD initialization failed: " + err, true);
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
          () => setStatus("Unable to load OSMD script (network/CSP issue)", true)
        );
      }}
    </script>
    """
    components.html(html, height=height, scrolling=False)


def render_seed_panel(
    piece_id: str,
    selected_groups: list[str],
    seed: int,
    *,
    show_measure_in_label: bool = False,
    hide_group_in_label: bool = False,
) -> None:
    xml_path = get_xml_path(piece_id)
    breakpoints = build_breakpoint_table(piece_id, selected_groups, seed)
    summary_df = build_summary_table(piece_id, selected_groups, seed, breakpoints)
    score_height = estimate_score_height(piece_id)

    st.markdown(f"### Seed {seed}")
    if breakpoints.empty:
        st.info("No local event table for this seed yet. Showing summary only.")
    else:
        annotated_xml = build_annotated_musicxml(
            xml_path,
            breakpoints,
            selected_groups,
            CLEAN_GROUP_SPECS,
            show_measure_in_label=show_measure_in_label,
            hide_group_in_label=hide_group_in_label,
        )
        render_score(
            annotated_xml,
            group_counts={row["group"]: int(row["events"]) for _, row in summary_df.iterrows()},
            group_specs=CLEAN_GROUP_SPECS,
            height=score_height,
        )


def render_strategy_panel(
    piece_id: str,
    selected_groups: list[str],
    variant: str,
    seed: int = 42,
    *,
    show_measure_in_label: bool = False,
    hide_group_in_label: bool = False,
) -> None:
    xml_path = get_xml_path(piece_id)
    breakpoints = build_strategy_breakpoint_table(piece_id, selected_groups, variant, seed=seed)
    summary_df = build_strategy_summary_table(piece_id, selected_groups, variant, breakpoints, seed=seed)
    score_height = estimate_score_height(piece_id)

    st.markdown(f"### {STRATEGY_LABELS.get(variant, variant)}")
    if breakpoints.empty:
        st.info("No local event table for this strategy yet. Showing summary only.")
    else:
        annotated_xml = build_annotated_musicxml(
            xml_path,
            breakpoints,
            selected_groups,
            STRATEGY_GROUP_SPECS,
            show_measure_in_label=show_measure_in_label,
            hide_group_in_label=hide_group_in_label,
        )
        render_score(
            annotated_xml,
            group_counts={row["group"]: int(row["events"]) for _, row in summary_df.iterrows()},
            group_specs=STRATEGY_GROUP_SPECS,
            height=score_height,
        )


def render_weighted_seed44_panel(
    piece_id: str,
    selected_groups: list[str],
    suppress_nearby_lower: bool = True,
    *,
    show_measure_in_label: bool = False,
    hide_group_in_label: bool = False,
    black_with_random_points: bool = False,
) -> None:
    xml_path = get_xml_path(piece_id)
    raw_breakpoints = build_weighted_breakpoint_table(piece_id, selected_groups, seed=44)
    breakpoints = raw_breakpoints
    if suppress_nearby_lower:
        breakpoints = suppress_lower_breakpoints_within_tolerance(
            raw_breakpoints,
            priority_order=[group for group in WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in selected_groups],
            tolerance=1,
        )
    display_groups = list(selected_groups)
    display_group_specs = {group: dict(spec) for group, spec in WEIGHTED_TOPDOWN_GROUP_SPECS.items()}
    if black_with_random_points:
        for spec in display_group_specs.values():
            spec["color"] = "#000000"
        random_breakpoints = build_random_breakpoints(piece_id, breakpoints, count=10)
        if not random_breakpoints.empty:
            breakpoints = pd.concat([breakpoints, random_breakpoints], ignore_index=True)
            display_groups.append(RANDOM_DISPLAY_GROUP)
            display_group_specs[RANDOM_DISPLAY_GROUP] = {"slug": "random_display", "color": "#000000"}
        breakpoints = breakpoints.sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)
    score_height = estimate_score_height(piece_id)
    group_counts = {str(group): int(count) for group, count in breakpoints.groupby("group").size().items()}

    st.markdown("### Weighted Topdown Clean Outer Seed44")
    if suppress_nearby_lower:
        st.caption("Display rule ON: higher levels take priority; lower-level breakpoints within +/-1 beat of a higher-level breakpoint are hidden.")
    else:
        st.caption("Display rule OFF: all selected breakpoint levels are shown without cross-level suppression.")
    if black_with_random_points:
        st.caption("Black-label mode ON: all displayed labels are black, and 10 reproducible random points are added to each score.")
    if breakpoints.empty:
        st.info("No local event table for weighted-topdown seed44 yet. Showing summary only.")
    else:
        annotated_xml = build_annotated_musicxml(
            xml_path,
            breakpoints,
            display_groups,
            display_group_specs,
            show_measure_in_label=show_measure_in_label,
            hide_group_in_label=hide_group_in_label,
        )
        render_score(
            annotated_xml,
            group_counts=group_counts,
            group_specs=display_group_specs,
            height=score_height,
        )


def render_new_score_seed44_panel(
    piece_id: str,
    selected_groups: list[str],
    suppress_nearby_lower: bool = True,
    *,
    show_measure_in_label: bool = False,
    hide_group_in_label: bool = False,
    black_with_random_points: bool = False,
) -> None:
    raw_breakpoints = build_new_score_breakpoint_table(piece_id, selected_groups)
    if raw_breakpoints.empty:
        st.info("No local event table for this new score yet.")
        return
    first_group = selected_groups[0]
    source_frame = load_new_score_events(piece_id, first_group)
    if source_frame.empty or "source_score_path" not in source_frame.columns:
        st.error("Missing source score path for this piece.")
        return
    xml_path = Path(source_frame["source_score_path"].dropna().iloc[0])
    breakpoints = raw_breakpoints
    if suppress_nearby_lower:
        breakpoints = suppress_lower_breakpoints_within_tolerance(
            raw_breakpoints,
            priority_order=[group for group in WEIGHTED_TOPDOWN_DISPLAY_PRIORITY if group in selected_groups],
            tolerance=1,
        )
    display_groups = list(selected_groups)
    display_group_specs = {group: dict(spec) for group, spec in WEIGHTED_TOPDOWN_GROUP_SPECS.items()}
    if black_with_random_points:
        for spec in display_group_specs.values():
            spec["color"] = "#000000"
        beat_frame = load_new_score_beat_frame(piece_id)
        random_breakpoints = build_random_breakpoints_from_frame(
            beat_frame,
            breakpoints,
            key=f"{piece_id}:new_scores_seed44:black_random:10",
            count=10,
        )
        if not random_breakpoints.empty:
            breakpoints = pd.concat([breakpoints, random_breakpoints], ignore_index=True)
            display_groups.append(RANDOM_DISPLAY_GROUP)
            display_group_specs[RANDOM_DISPLAY_GROUP] = {"slug": "random_display", "color": "#000000"}
        breakpoints = breakpoints.sort_values(["measure", "beat_in_measure", "group", "beat_idx"]).reset_index(drop=True)
    score_height = estimate_score_height_from_xml(str(xml_path))
    group_counts = {str(group): int(count) for group, count in breakpoints.groupby("group").size().items()}

    st.markdown("### New Scores Seed44")
    if suppress_nearby_lower:
        st.caption("Display rule ON: higher levels take priority; lower-level breakpoints within +/-1 beat of a higher-level breakpoint are hidden.")
    else:
        st.caption("Display rule OFF: all selected breakpoint levels are shown without cross-level suppression.")
    if black_with_random_points:
        st.caption("Black-label mode ON: all displayed labels are black, and 10 reproducible random points are added to each score.")
    annotated_xml = build_annotated_musicxml(
        xml_path,
        breakpoints,
        display_groups,
        display_group_specs,
        show_measure_in_label=show_measure_in_label,
        hide_group_in_label=hide_group_in_label,
    )
    render_score(
        annotated_xml,
        group_counts=group_counts,
        group_specs=display_group_specs,
        height=score_height,
    )


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed; install it before running this app.")
    st.set_page_config(page_title="Mazurka Clean Outer Breakpoint Visualizer", layout="wide")
    st.title("Mazurka Breakpoint Visualizer")
    view_mode = st.sidebar.radio(
        "View Mode",
        options=["weighted_topdown_seed44", "new_scores_seed44", "clean_outer_seeds", "strategy_compare"],
        format_func=lambda value: {
            "weighted_topdown_seed44": "Weighted Topdown Seed44",
            "new_scores_seed44": "New Scores Seed44",
            "clean_outer_seeds": "Clean Outer Seed42 vs Seed44",
            "strategy_compare": "Baseline vs Consensus Guarded",
        }[value],
    )
    if view_mode == "weighted_topdown_seed44":
        st.caption("Merged L5+6 weighted top-down wide-gap + TCN direct + train floor 0.05 clean outer Seed44 results across five levels.")
    elif view_mode == "new_scores_seed44":
        st.caption("Seed44 merged L5+6 weighted top-down predictions on Beethoven Pathetique II, Mozart K.283 I, and Mozart K.331 I.")
    elif view_mode == "clean_outer_seeds":
        st.caption("Clean outer test results across five levels; Seed42 and Seed44 are shown side by side.")
    else:
        st.caption("Baseline vs new strategy across five pieces and six levels; in strategy mode, upper levels are split into L5 and L6, using the newer results with union precision floor 0.7.")

    group_order = get_group_order(view_mode)
    selected_groups = st.sidebar.multiselect("Visible Levels", options=group_order, default=group_order)
    if not selected_groups:
        st.warning("Select at least one level.")
        return
    suppress_nearby_lower = True
    show_measure_in_label = False
    hide_group_in_label = False
    if view_mode in {"weighted_topdown_seed44", "new_scores_seed44"}:
        suppress_nearby_lower = st.sidebar.toggle(
            "Hide lower breakpoints near higher levels",
            value=True,
            help="If enabled, lower-level breakpoints within +/-1 beat of a higher-level breakpoint are hidden.",
        )
        show_measure_in_label = st.sidebar.toggle(
            "Show measure number in labels",
            value=True,
            help="If enabled, score labels include the current measure number.",
        )
        hide_group_in_label = st.sidebar.toggle(
            "Hide level names in labels",
            value=True,
            help="If enabled, score labels omit the hierarchy name and rely on color only.",
        )
        black_with_random_points = st.sidebar.toggle(
            "Black labels + 10 random points",
            value=True,
            help="If enabled, all displayed breakpoint labels become black and 10 reproducible random points are added to each score.",
        )
    else:
        black_with_random_points = False

    if view_mode == "weighted_topdown_seed44":
        pieces = WEIGHTED_TOPDOWN_PIECES
    elif view_mode == "new_scores_seed44":
        pieces = NEW_SCORE_PIECES
    else:
        pieces = SUPPORTED_PIECES
    tab_labels = [NEW_SCORE_LABELS.get(piece_id, piece_id) for piece_id in pieces]
    tabs = st.tabs(tab_labels)
    for tab, piece_id in zip(tabs, pieces):
        with tab:
            st.subheader(NEW_SCORE_LABELS.get(piece_id, piece_id))
            if view_mode == "weighted_topdown_seed44":
                render_weighted_seed44_panel(
                    piece_id,
                    selected_groups,
                    suppress_nearby_lower=suppress_nearby_lower,
                    show_measure_in_label=show_measure_in_label,
                    hide_group_in_label=hide_group_in_label,
                    black_with_random_points=black_with_random_points,
                )
            elif view_mode == "new_scores_seed44":
                render_new_score_seed44_panel(
                    piece_id,
                    selected_groups,
                    suppress_nearby_lower=suppress_nearby_lower,
                    show_measure_in_label=show_measure_in_label,
                    hide_group_in_label=hide_group_in_label,
                    black_with_random_points=black_with_random_points,
                )
            else:
                cols = st.columns(2)
                if view_mode == "clean_outer_seeds":
                    for col, seed in zip(cols, SEED_ORDER):
                        with col:
                            render_seed_panel(
                                piece_id,
                                selected_groups,
                                seed,
                                show_measure_in_label=show_measure_in_label,
                                hide_group_in_label=hide_group_in_label,
                            )
                else:
                    for col, variant in zip(cols, STRATEGY_ORDER):
                        with col:
                            render_strategy_panel(
                                piece_id,
                                selected_groups,
                                variant,
                                seed=42,
                                show_measure_in_label=show_measure_in_label,
                                hide_group_in_label=hide_group_in_label,
                            )


if __name__ == "__main__":
    main()
