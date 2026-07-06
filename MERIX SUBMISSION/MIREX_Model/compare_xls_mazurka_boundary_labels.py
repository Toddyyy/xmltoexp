import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
MAZURKA_ROOT = ROOT / "MazurkaBL-master"
CURRENT_LEVEL_DIR = Path(__file__).resolve().parent / "beat_data_mazurka_performer_levels"
OUT_DIR = Path(__file__).resolve().parent / "xls_mazurka_boundary_compare"
STR_VEC = [3, 2, 2, 2, 2, 2]


def load_beat_time(file_path: Path):
    df = pd.read_csv(file_path)
    meta_cols = {"Unnamed: 0", "measure_number", "beat_number"}
    performer_cols = [c for c in df.columns if c not in meta_cols]
    return df, performer_cols


def compute_tempo_curves(df, performer_cols, smooth_window=3, bpm_range=(0, 5000), clip_max=600):
    tempo_arrays = {}
    lo, hi = bpm_range
    for col in performer_cols:
        times = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(times, prepend=np.nan)
        tempo = 60.0 / dt
        invalid = (dt <= 0) | (~np.isfinite(tempo))
        tempo[invalid] = np.nan

        s = pd.Series(tempo)
        s = s.where((s > lo) & (s < hi))
        s = s.interpolate("linear", limit_direction="both")
        s = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
        s = s.clip(upper=clip_max)
        tempo_arrays[col] = s.to_numpy()
    return tempo_arrays


def _pad_to_multiple(x, m):
    x = np.asarray(x, dtype=float).reshape(-1)
    pad = (-len(x)) % m
    if pad == 0:
        return x
    return np.concatenate([x, np.zeros(pad, dtype=float)])


def group_analysis_hierarchy(tempo_curve, str_vec, enforce_nested=True):
    tempo = np.asarray(tempo_curve, dtype=float).reshape(-1)
    n = len(tempo)
    str_vec = np.asarray(str_vec, dtype=int).reshape(-1)
    L = len(str_vec)

    if n == 0:
        return np.zeros((L, 0), dtype=bool), {l: np.array([], dtype=int) for l in range(1, L + 1)}

    avg = np.nanmean(tempo)
    if not np.isfinite(avg) or avg == 0:
        avg = 1.0

    energy = [None] * (L + 1)
    energy2 = [None] * (L + 1)
    valleys = [None] * (L + 1)

    s1 = int(str_vec[0])
    pad_raw = _pad_to_multiple(tempo, s1)
    pad_norm = _pad_to_multiple(tempo / avg, s1)

    energy[1] = pad_raw.reshape((s1, -1), order="F")
    energy2[1] = pad_norm.reshape((s1, -1), order="F")

    v1, _ = find_peaks(-tempo)
    valleys[1] = v1 + 1

    for i in range(2, L + 1):
        t_eng = energy[i - 1].copy()
        t_eng[t_eng == 0] = np.nan
        t_norm = t_eng / avg
        eng2 = np.sqrt(np.nanmean(t_norm**2, axis=0)) - np.nanstd(t_norm, axis=0)
        vi, _ = find_peaks(-eng2)
        valleys[i] = vi + 1
        si = int(str_vec[i - 1])
        eng2_pad = _pad_to_multiple(eng2, si)
        energy[i] = _pad_to_multiple(np.sqrt(np.nanmean(t_eng**2, axis=0)), si).reshape((si, -1), order="F")
        energy2[i] = eng2_pad.reshape((si, -1), order="F")

    results_raw = np.zeros((L, n), dtype=bool)
    for top in range(L, 0, -1):
        roots = valleys[top]
        if roots is None or len(roots) == 0:
            continue
        trace = np.zeros((top, len(roots)), dtype=int)
        trace[0, :] = roots
        for rj in range(len(roots)):
            for i in range(2, top + 1):
                lvl = top + 1 - i
                parent_col = trace[i - 2, rj]
                colvec = energy2[lvl][:, parent_col - 1]
                row = int(np.nanargmin(colvec)) + 1
                trace[i - 1, rj] = int(str_vec[lvl - 1]) * (parent_col - 1) + row
        beats_1b = trace[top - 1, :]
        beats_0b = beats_1b - 1
        beats_0b = beats_0b[(beats_0b >= 0) & (beats_0b < n)]
        results_raw[top - 1, beats_0b] = True

    level_sets = {l: np.where(results_raw[l - 1])[0] for l in range(1, L + 1)}
    if enforce_nested:
        cum = np.zeros(n, dtype=bool)
        nested = {}
        for l in range(L, 0, -1):
            cum |= results_raw[l - 1]
            nested[l] = np.where(cum)[0]
        level_sets = nested
    return results_raw, level_sets


def boundaries_to_mask(n_beats, boundary_indices):
    mask = np.zeros(n_beats, dtype=np.float32)
    boundary_indices = np.asarray(boundary_indices, dtype=int)
    boundary_indices = boundary_indices[(boundary_indices >= 0) & (boundary_indices < n_beats)]
    mask[boundary_indices] = 1.0
    return mask


def piece_id_from_xls(path: Path) -> str:
    match = re.search(r"mazurka(\d+)-(\d+)", path.stem, flags=re.I)
    if not match:
        raise ValueError(f"Cannot parse Mazurka id from {path}")
    return f"M{int(match.group(1)):02d}-{match.group(2)}"


def align_times_to_mazurkabl(times: np.ndarray, num_beats: int) -> np.ndarray | None:
    times = np.asarray(times, dtype=float).reshape(-1)
    if len(times) == num_beats:
        return times
    if len(times) == num_beats + 1:
        return times[:num_beats]
    return None


def load_xls_time_curves(xls_path: Path, num_beats: int) -> tuple[dict[str, np.ndarray], list[dict]]:
    xl = pd.ExcelFile(xls_path, engine="xlrd")
    curves: dict[str, np.ndarray] = {}
    skipped: list[dict] = []
    for sheet in xl.sheet_names:
        if sheet in {"Summary", "Correlation"}:
            continue
        frame = xl.parse(sheet, header=None)
        header = frame.iloc[0].astype(str).str.strip().str.lower().tolist()
        if "time" not in header or "absbeat" not in header:
            skipped.append({"sheet": sheet, "reason": "missing_time_or_absbeat"})
            continue
        time_col = header.index("time")
        abs_col = header.index("absbeat")
        mask = pd.to_numeric(frame.iloc[:, abs_col], errors="coerce").notna()
        times = pd.to_numeric(frame.loc[mask, time_col], errors="coerce").to_numpy(dtype=float)
        aligned = align_times_to_mazurkabl(times, num_beats)
        if aligned is None:
            skipped.append(
                {
                    "sheet": sheet,
                    "reason": "length_mismatch",
                    "xls_len": int(len(times)),
                    "num_beats": int(num_beats),
                }
            )
            continue
        curves[sheet] = aligned
    return curves, skipped


def xls_frequency_by_level(xls_path: Path, beat_grid: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict]:
    num_beats = len(beat_grid)
    curves, skipped = load_xls_time_curves(xls_path, num_beats)
    if not curves:
        raise RuntimeError(f"No usable performer curves in {xls_path}")
    df = pd.DataFrame({"measure_number": beat_grid["measure_number"], "beat_number": beat_grid["beat_number"]})
    for name, times in curves.items():
        df[name] = times
    tempo_arrays = compute_tempo_curves(df, list(curves), smooth_window=3, bpm_range=(0, 5000), clip_max=600)

    sums = {level: np.zeros(num_beats, dtype=np.float64) for level in range(1, len(STR_VEC) + 1)}
    for curve in tempo_arrays.values():
        _, level_sets = group_analysis_hierarchy(curve, STR_VEC, enforce_nested=True)
        for level in sums:
            sums[level] += boundaries_to_mask(num_beats, level_sets.get(level, np.array([], dtype=int)))
    freqs = {level: sums[level] / float(len(tempo_arrays)) for level in sums}
    meta = {"performers": len(tempo_arrays), "skipped": skipped}
    return freqs, meta


def current_frequency_by_level(piece_id: str) -> tuple[dict[int, np.ndarray], int]:
    freqs: dict[int, np.ndarray] = {}
    performer_count = 0
    for level in range(1, len(STR_VEC) + 1):
        files = sorted(CURRENT_LEVEL_DIR.glob(f"{piece_id}_*_L{level}.npz"))
        if not files:
            raise FileNotFoundError(f"No current level files for {piece_id} L{level}")
        arrays = [np.load(path, allow_pickle=True)["boundary_probs"].astype(float) for path in files]
        if level == 1:
            performer_count = len(arrays)
        freqs[level] = np.mean(np.stack(arrays, axis=0), axis=0)
    return freqs, performer_count


def event_support(freq: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.flatnonzero(np.asarray(freq) > threshold)


def tolerant_counts(pred: np.ndarray, ref: np.ndarray, tol: int = 1) -> tuple[int, int, int]:
    pred = list(map(int, pred))
    ref = list(map(int, ref))
    used = set()
    matched = 0
    for p in pred:
        best = None
        best_dist = tol + 1
        for j, r in enumerate(ref):
            if j in used:
                continue
            dist = abs(p - r)
            if dist <= tol and dist < best_dist:
                best = j
                best_dist = dist
        if best is not None:
            used.add(best)
            matched += 1
    return matched, len(pred), len(ref)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_rows = []

    for xls_path in sorted(DATASETS_DIR.glob("mazurka*.xls")):
        piece_id = piece_id_from_xls(xls_path)
        beat_time_path = MAZURKA_ROOT / "beat_time" / f"{piece_id}beat_time.csv"
        beat_grid, _ = load_beat_time(beat_time_path)

        xls_freqs, xls_meta = xls_frequency_by_level(xls_path, beat_grid)
        cur_freqs, cur_performers = current_frequency_by_level(piece_id)

        for item in xls_meta["skipped"]:
            skipped_rows.append({"piece_id": piece_id, "xls_file": xls_path.name, **item})

        for level in range(1, len(STR_VEC) + 1):
            x = xls_freqs[level]
            c = cur_freqs[level]
            x_events = event_support(x)
            c_events = event_support(c)
            matched, x_count, c_count = tolerant_counts(x_events, c_events, tol=1)
            precision = matched / x_count if x_count else float("nan")
            recall = matched / c_count if c_count else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
            rows.append(
                {
                    "piece_id": piece_id,
                    "level": level,
                    "num_beats": int(len(beat_grid)),
                    "xls_performers_used": int(xls_meta["performers"]),
                    "current_performers": int(cur_performers),
                    "xls_expected_boundary_count": float(x.sum()),
                    "current_expected_boundary_count": float(c.sum()),
                    "count_diff_xls_minus_current": float(x.sum() - c.sum()),
                    "freq_mae": float(np.mean(np.abs(x - c))),
                    "freq_rmse": float(np.sqrt(np.mean((x - c) ** 2))),
                    "freq_pearson": pearson(x, c),
                    "xls_support_count": int(x_count),
                    "current_support_count": int(c_count),
                    "support_matched_tol1": int(matched),
                    "support_precision_tol1": precision,
                    "support_recall_tol1": recall,
                    "support_f1_tol1": f1,
                }
            )

        piece_frame = pd.DataFrame({"beat_index": np.arange(len(beat_grid), dtype=int)})
        for level in range(1, len(STR_VEC) + 1):
            piece_frame[f"xls_L{level}"] = xls_freqs[level]
            piece_frame[f"current_L{level}"] = cur_freqs[level]
        piece_frame.to_csv(OUT_DIR / f"{piece_id}_frequency_compare.csv", index=False)

    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "summary_by_piece_level.csv", index=False)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(OUT_DIR / "skipped_xls_sheets.csv", index=False)

    pd.set_option("display.max_columns", 30)
    print(report.to_string(index=False))
    print(f"\nWrote reports to {OUT_DIR}")
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} xls sheets; see skipped_xls_sheets.csv")


if __name__ == "__main__":
    main()
