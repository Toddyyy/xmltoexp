from __future__ import annotations

import numpy as np


def build_rest_span_arrays(empty_mask: np.ndarray, min_len: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty_mask = np.asarray(empty_mask, dtype=bool)
    n = empty_mask.shape[0]
    span_id = np.full(n, -1, dtype=np.int32)
    span_start = np.full(n, -1, dtype=np.int32)
    span_end = np.full(n, -1, dtype=np.int32)

    current_id = 0
    idx = 0
    while idx < n:
        if not empty_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < n and empty_mask[idx + 1]:
            idx += 1
        end = idx
        if end - start + 1 >= int(min_len):
            span_id[start : end + 1] = current_id
            span_start[start : end + 1] = start
            span_end[start : end + 1] = end
            current_id += 1
        idx += 1

    return span_id, span_start, span_end


def rest_aware_distance(pred_idx: int, true_idx: int, span_id: np.ndarray) -> int:
    pred_idx = int(pred_idx)
    true_idx = int(true_idx)
    if 0 <= pred_idx < len(span_id) and 0 <= true_idx < len(span_id):
        pred_span = int(span_id[pred_idx])
        true_span = int(span_id[true_idx])
        if pred_span >= 0 and pred_span == true_span:
            return 0
    return abs(pred_idx - true_idx)


def greedy_match_pairs_rest_aware(
    pred_events: np.ndarray,
    true_events: np.ndarray,
    tolerance: int,
    span_id: np.ndarray,
) -> list[tuple[int, int, int]]:
    pred_events = np.asarray(pred_events, dtype=np.int32)
    true_events = np.asarray(true_events, dtype=np.int32)
    candidates = []
    for pred_i, pred in enumerate(pred_events.tolist()):
        for true_i, true in enumerate(true_events.tolist()):
            diff = rest_aware_distance(int(pred), int(true), span_id=span_id)
            if diff <= int(tolerance):
                offset = 0 if diff == 0 and int(pred) != int(true) else int(pred) - int(true)
                candidates.append((diff, pred_i, true_i, offset))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_pred = set()
    matched_true = set()
    matches = []
    for _, pred_i, true_i, offset in candidates:
        if pred_i in matched_pred or true_i in matched_true:
            continue
        matched_pred.add(pred_i)
        matched_true.add(true_i)
        matches.append((pred_i, true_i, offset))
    return matches


def snap_events_to_rest_spans(
    events: np.ndarray,
    span_start: np.ndarray,
    span_end: np.ndarray,
    mode: str = "center",
) -> np.ndarray:
    events = np.asarray(events, dtype=np.int32)
    snapped = []
    for event in events.tolist():
        event = int(event)
        if 0 <= event < len(span_start) and int(span_start[event]) >= 0:
            start = int(span_start[event])
            end = int(span_end[event])
            if mode == "start":
                target = start
            elif mode == "end":
                target = end
            elif mode == "center":
                target = int(round((start + end) / 2.0))
            else:
                raise ValueError(f"Unsupported snap mode: {mode}")
            snapped.append(target)
        else:
            snapped.append(event)
    return np.asarray(sorted(set(snapped)), dtype=np.int32)


def expand_frequency_over_rest_spans(
    freq_targets: np.ndarray,
    span_id: np.ndarray,
    agg: str = "max",
) -> np.ndarray:
    freq_targets = np.asarray(freq_targets, dtype=np.float32)
    expanded = freq_targets.copy()
    valid_span_ids = sorted(int(v) for v in np.unique(span_id) if int(v) >= 0)
    for current_span in valid_span_ids:
        mask = span_id == current_span
        span_values = freq_targets[mask]
        if not np.any(span_values > 0.0):
            continue
        if agg == "max":
            fill_value = float(np.max(span_values))
        elif agg == "mean":
            fill_value = float(np.mean(span_values[span_values > 0.0]))
        else:
            raise ValueError(f"Unsupported agg: {agg}")
        expanded[mask] = fill_value
    return expanded.astype(np.float32)


def canonicalize_frequency_with_ignore(
    freq_targets: np.ndarray,
    span_id: np.ndarray,
    agg: str = "max",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq_targets = np.asarray(freq_targets, dtype=np.float32)
    train_freq = freq_targets.copy()
    train_union = (freq_targets > 0.0).astype(np.float32)
    loss_factor = np.ones_like(freq_targets, dtype=np.float32)

    valid_span_ids = sorted(int(v) for v in np.unique(span_id) if int(v) >= 0)
    for current_span in valid_span_ids:
        mask = span_id == current_span
        span_indices = np.flatnonzero(mask)
        span_values = freq_targets[mask]
        positive_mask = span_values > 0.0
        if not np.any(positive_mask):
            continue

        positive_indices = span_indices[positive_mask]
        positive_values = freq_targets[positive_indices]
        max_value = float(np.max(positive_values))
        candidate_indices = positive_indices[np.isclose(positive_values, max_value)]
        span_center = 0.5 * float(span_indices[0] + span_indices[-1])
        canonical_idx = int(candidate_indices[np.argmin(np.abs(candidate_indices - span_center))])

        if agg == "max":
            canonical_value = max_value
        elif agg == "sum_clip":
            canonical_value = float(np.clip(np.sum(span_values[positive_mask]), 0.0, 1.0))
        else:
            raise ValueError(f"Unsupported agg: {agg}")

        train_freq[mask] = 0.0
        train_union[mask] = 0.0
        loss_factor[mask] = 0.0

        train_freq[canonical_idx] = canonical_value
        train_union[canonical_idx] = 1.0 if canonical_value > 0.0 else 0.0
        loss_factor[canonical_idx] = 1.0

    return train_freq.astype(np.float32), train_union.astype(np.float32), loss_factor.astype(np.float32)


def build_rest_span_tolerance_weights(
    freq_targets: np.ndarray,
    span_id: np.ndarray,
    negative_weight: float = 0.25,
) -> np.ndarray:
    freq_targets = np.asarray(freq_targets, dtype=np.float32)
    weights = np.ones_like(freq_targets, dtype=np.float32)
    if float(negative_weight) >= 1.0:
        return weights

    valid_span_ids = sorted(int(v) for v in np.unique(span_id) if int(v) >= 0)
    for current_span in valid_span_ids:
        mask = span_id == current_span
        span_indices = np.flatnonzero(mask)
        span_values = freq_targets[mask]
        positive_indices = span_indices[span_values > 0.0]
        if positive_indices.size == 0:
            continue
        max_span_distance = max(int(span_indices[-1] - span_indices[0]), 1)
        for idx in span_indices.tolist():
            if freq_targets[idx] > 0.0:
                continue
            nearest_positive_distance = int(np.min(np.abs(positive_indices - idx)))
            scaled = float(negative_weight) + (1.0 - float(negative_weight)) * (
                max(float(nearest_positive_distance - 1), 0.0) / float(max_span_distance)
            )
            weights[idx] = float(np.clip(scaled, float(negative_weight), 1.0))
    return weights.astype(np.float32)
