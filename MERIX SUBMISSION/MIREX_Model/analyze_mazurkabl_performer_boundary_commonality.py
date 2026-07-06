from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "beat_data_mazurka_performer_levels"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_performer_boundary_commonality"
LEVELS = [1, 2, 3, 4, 5, 6]
RNG = np.random.default_rng(20260616)


def parse_label_path(path: Path) -> tuple[str, str, int]:
    m = re.match(r"(M\d+-\d+)_(.+)_L(\d+)\.npz$", path.name)
    if not m:
        raise ValueError(path.name)
    return m.group(1), m.group(2), int(m.group(3))


def match_count(a_idx: np.ndarray, b_idx: np.ndarray, tolerance: int) -> int:
    if len(a_idx) == 0 or len(b_idx) == 0:
        return 0
    used = np.zeros(len(b_idx), dtype=bool)
    count = 0
    for a in a_idx:
        d = np.abs(b_idx - a)
        d[used] = tolerance + 1
        j = int(np.argmin(d))
        if d[j] <= tolerance:
            used[j] = True
            count += 1
    return count


def pairwise_metrics(masks: np.ndarray, max_pairs: int = 80) -> dict:
    n_perf = masks.shape[0]
    pairs = list(itertools.combinations(range(n_perf), 2))
    if len(pairs) > max_pairs:
        pairs = [tuple(x) for x in RNG.choice(np.array(pairs), size=max_pairs, replace=False)]
    exact_j = []
    f1_tol1 = []
    f1_tol2 = []
    for i, j in pairs:
        a = masks[i] > 0.5
        b = masks[j] > 0.5
        union = np.count_nonzero(a | b)
        inter = np.count_nonzero(a & b)
        exact_j.append(inter / union if union else np.nan)
        ai = np.flatnonzero(a)
        bi = np.flatnonzero(b)
        for tol, store in [(1, f1_tol1), (2, f1_tol2)]:
            m_ab = match_count(ai, bi, tol)
            p = m_ab / len(ai) if len(ai) else 0.0
            r = m_ab / len(bi) if len(bi) else 0.0
            store.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "pair_count_sampled": len(pairs),
        "pairwise_exact_jaccard_mean": float(np.nanmean(exact_j)) if exact_j else np.nan,
        "pairwise_f1_tol1_mean": float(np.nanmean(f1_tol1)) if f1_tol1 else np.nan,
        "pairwise_f1_tol2_mean": float(np.nanmean(f1_tol2)) if f1_tol2 else np.nan,
    }


def random_pairwise_baseline(masks: np.ndarray, repeats: int = 20) -> dict:
    n_perf, n_beats = masks.shape
    counts = masks.sum(axis=1).astype(int)
    f1_tol1 = []
    exact_j = []
    for _ in range(repeats):
        sim = np.zeros_like(masks)
        for i, k in enumerate(counts):
            if k > 0:
                sim[i, RNG.choice(n_beats, size=min(k, n_beats), replace=False)] = 1
        m = pairwise_metrics(sim, max_pairs=50)
        f1_tol1.append(m["pairwise_f1_tol1_mean"])
        exact_j.append(m["pairwise_exact_jaccard_mean"])
    return {
        "random_exact_jaccard_mean": float(np.nanmean(exact_j)),
        "random_f1_tol1_mean": float(np.nanmean(f1_tol1)),
    }


def load_beat_positions(piece: str, n_beats: int) -> np.ndarray:
    path = BEAT_TIME_DIR / f"{piece}beat_time.csv"
    if not path.exists():
        return np.full(n_beats, np.nan)
    df = pd.read_csv(path)
    col = "beat_number" if "beat_number" in df.columns else None
    if col is None:
        return np.full(n_beats, np.nan)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if len(arr) >= n_beats:
        return arr[:n_beats]
    return np.pad(arr, (0, n_beats - len(arr)), constant_values=np.nan)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(LABEL_DIR.glob("*_L*.npz")):
        piece, performer, level = parse_label_path(path)
        arr = np.load(path, allow_pickle=True)["boundary_probs"].astype(np.float32)
        records.append((piece, performer, level, arr))

    by_key: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for piece, performer, level, arr in records:
        by_key.setdefault((piece, level), []).append((performer, arr))

    piece_rows = []
    beat_rows = []
    beatpos_rows = []
    for (piece, level), items in sorted(by_key.items()):
        performers = [x[0] for x in items]
        masks = np.stack([x[1] for x in items], axis=0)
        n_perf, n_beats = masks.shape
        consensus = masks.mean(axis=0)
        total_events = int(masks.sum())
        top10_n = max(1, int(round(0.10 * n_beats)))
        top_idx = np.argsort(-consensus)[:top10_n]
        top10_event_share = float(masks[:, top_idx].sum() / total_events) if total_events else 0.0
        pair = pairwise_metrics(masks)
        rand = random_pairwise_baseline(masks)
        piece_rows.append(
            {
                "piece": piece,
                "level": level,
                "num_performers": n_perf,
                "num_beats": n_beats,
                "total_boundary_events": total_events,
                "mean_events_per_performer": float(masks.sum(axis=1).mean()),
                "std_events_per_performer": float(masks.sum(axis=1).std()),
                "beats_with_any_boundary": int(np.count_nonzero(consensus > 0)),
                "beats_consensus_ge_0p10": int(np.count_nonzero(consensus >= 0.10)),
                "beats_consensus_ge_0p25": int(np.count_nonzero(consensus >= 0.25)),
                "beats_consensus_ge_0p50": int(np.count_nonzero(consensus >= 0.50)),
                "max_consensus": float(consensus.max()) if consensus.size else 0.0,
                "top10pct_beats_event_share": top10_event_share,
                **pair,
                **rand,
            }
        )
        for beat in np.flatnonzero(consensus > 0):
            beat_rows.append(
                {
                    "piece": piece,
                    "level": level,
                    "beat": int(beat),
                    "consensus": float(consensus[beat]),
                    "performer_count": int(masks[:, beat].sum()),
                    "num_performers": n_perf,
                }
            )

        beat_pos = load_beat_positions(piece, n_beats)
        for pos in sorted(set(beat_pos[np.isfinite(beat_pos)].astype(int).tolist())):
            all_count = int(np.count_nonzero(beat_pos == pos))
            boundary_count = int(masks[:, beat_pos == pos].sum()) if all_count else 0
            non_boundary_count = int(n_perf * all_count - boundary_count)
            other_beats = np.isfinite(beat_pos) & (beat_pos != pos)
            other_boundary = int(masks[:, other_beats].sum())
            other_non_boundary = int(n_perf * int(other_beats.sum()) - other_boundary)
            if min(boundary_count + non_boundary_count, other_boundary + other_non_boundary) <= 0:
                continue
            table = np.array([[boundary_count, non_boundary_count], [other_boundary, other_non_boundary]])
            try:
                _chi, p_value, _dof, _expected = chi2_contingency(table, correction=False)
            except ValueError:
                p_value = np.nan
            boundary_rate = boundary_count / max(n_perf * all_count, 1)
            other_rate = other_boundary / max(n_perf * int(other_beats.sum()), 1)
            beatpos_rows.append(
                {
                    "piece": piece,
                    "level": level,
                    "beat_number_in_measure": pos,
                    "boundary_rate": boundary_rate,
                    "other_boundary_rate": other_rate,
                    "rate_ratio": boundary_rate / other_rate if other_rate > 0 else np.nan,
                    "chi2_p": float(p_value),
                    "boundary_count": boundary_count,
                    "all_slots": int(n_perf * all_count),
                }
            )

    piece_df = pd.DataFrame(piece_rows)
    beat_df = pd.DataFrame(beat_rows)
    beatpos_df = pd.DataFrame(beatpos_rows)
    piece_df.to_csv(OUT_DIR / "piece_level_commonality.csv", index=False)
    beat_df.to_csv(OUT_DIR / "beat_consensus_nonzero.csv", index=False)
    beatpos_df.to_csv(OUT_DIR / "beat_position_bias.csv", index=False)

    level_summary = (
        piece_df.groupby("level")
        .agg(
            pieces=("piece", "nunique"),
            mean_performers=("num_performers", "mean"),
            mean_events_per_performer=("mean_events_per_performer", "mean"),
            mean_beats_consensus_ge_0p10=("beats_consensus_ge_0p10", "mean"),
            mean_beats_consensus_ge_0p25=("beats_consensus_ge_0p25", "mean"),
            mean_beats_consensus_ge_0p50=("beats_consensus_ge_0p50", "mean"),
            mean_max_consensus=("max_consensus", "mean"),
            mean_top10pct_event_share=("top10pct_beats_event_share", "mean"),
            mean_pairwise_f1_tol1=("pairwise_f1_tol1_mean", "mean"),
            mean_random_f1_tol1=("random_f1_tol1_mean", "mean"),
            mean_pairwise_exact_jaccard=("pairwise_exact_jaccard_mean", "mean"),
            mean_random_exact_jaccard=("random_exact_jaccard_mean", "mean"),
        )
        .reset_index()
    )
    level_summary["f1_tol1_lift_over_random"] = (
        level_summary["mean_pairwise_f1_tol1"] / level_summary["mean_random_f1_tol1"]
    )
    level_summary["jaccard_lift_over_random"] = (
        level_summary["mean_pairwise_exact_jaccard"] / level_summary["mean_random_exact_jaccard"]
    )
    level_summary.to_csv(OUT_DIR / "level_commonality_summary.csv", index=False)

    beatpos_summary = (
        beatpos_df.groupby(["level", "beat_number_in_measure"])
        .agg(
            mean_boundary_rate=("boundary_rate", "mean"),
            mean_other_boundary_rate=("other_boundary_rate", "mean"),
            median_rate_ratio=("rate_ratio", "median"),
            pieces=("piece", "nunique"),
        )
        .reset_index()
    )
    beatpos_summary.to_csv(OUT_DIR / "beat_position_bias_summary.csv", index=False)
    print("Level summary:")
    print(level_summary.round(4).to_string(index=False))
    print("\nBeat-position bias summary:")
    print(beatpos_summary.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
