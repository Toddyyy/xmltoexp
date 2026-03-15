from __future__ import annotations

import numpy as np
import pandas as pd


def add_highlevel_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    required = {
        "xml_phrase_stop_proxy",
        "xml_repeat_end",
        "xml_final_barline",
        "xml_harmonic_change",
        "xml_cadence_dom_tonic",
        "xml_part_exit_count_norm",
        "xml_prev_rest_duration_norm",
        "xml_next_rest_duration_norm",
        "xml_rest_duration_norm",
    }
    if not required.issubset(frame.columns):
        return frame

    phrase = frame["xml_phrase_stop_proxy"].to_numpy(dtype=np.float32)
    repeat_end = frame["xml_repeat_end"].to_numpy(dtype=np.float32)
    final_barline = frame["xml_final_barline"].to_numpy(dtype=np.float32)
    harmonic = frame["xml_harmonic_change"].to_numpy(dtype=np.float32)
    cadence = frame["xml_cadence_dom_tonic"].to_numpy(dtype=np.float32)
    part_exit = frame["xml_part_exit_count_norm"].to_numpy(dtype=np.float32)
    prev_rest = frame["xml_prev_rest_duration_norm"].to_numpy(dtype=np.float32)
    next_rest = frame["xml_next_rest_duration_norm"].to_numpy(dtype=np.float32)
    rest = frame["xml_rest_duration_norm"].to_numpy(dtype=np.float32)

    section_break = np.maximum.reduce([phrase, repeat_end, final_barline]).astype(np.float32)
    local_rest = np.maximum.reduce([prev_rest, next_rest, rest]).astype(np.float32)

    frame["xml_section_break_strength"] = section_break
    frame["xml_cadence_phrase_interaction"] = (cadence * phrase).astype(np.float32)
    frame["xml_harmonic_phrase_interaction"] = (harmonic * phrase).astype(np.float32)
    frame["xml_harmonic_section_interaction"] = (harmonic * section_break).astype(np.float32)
    frame["xml_cadence_section_interaction"] = (cadence * section_break).astype(np.float32)
    frame["xml_part_exit_phrase_interaction"] = (part_exit * phrase).astype(np.float32)
    frame["xml_rest_phrase_interaction"] = (local_rest * phrase).astype(np.float32)
    frame["xml_rest_section_interaction"] = (local_rest * section_break).astype(np.float32)
    return frame
