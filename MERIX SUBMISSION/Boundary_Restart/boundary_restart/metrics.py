from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score


@dataclass
class EventMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    mean_offset: float | None
    matches: int
    pred_events: int
    true_events: int
    average_precision: float


@dataclass
class LabeledEventMetrics:
    threshold: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    mean_offset: float | None
    class_precision: dict[int, float]
    class_recall: dict[int, float]
    class_f1: dict[int, float]
    class_matches: dict[int, int]
    class_pred_events: dict[int, int]
    class_true_events: dict[int, int]


def extract_events(
    scores: np.ndarray,
    threshold: float,
    min_distance: int = 1,
    prominence: float = 0.0,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return np.zeros(0, dtype=np.int32)
    peaks, _ = find_peaks(
        scores,
        height=float(threshold),
        distance=max(int(min_distance), 1),
        prominence=max(float(prominence), 0.0),
    )
    candidates = list(peaks.astype(int))
    if scores.size == 1 and scores[0] >= threshold:
        candidates.append(0)
    elif scores.size > 1:
        if scores[0] >= threshold and scores[0] >= scores[1]:
            candidates.append(0)
        if scores[-1] >= threshold and scores[-1] >= scores[-2]:
            candidates.append(scores.size - 1)
    if not candidates:
        return np.zeros(0, dtype=np.int32)
    candidates = sorted(set(candidates), key=lambda idx: (-scores[idx], idx))
    kept = []
    for idx in candidates:
        if all(abs(idx - prev) >= max(int(min_distance), 1) for prev in kept):
            kept.append(idx)
    return np.asarray(sorted(kept), dtype=np.int32)


def greedy_match(pred_events: np.ndarray, true_events: np.ndarray, tolerance: int) -> list[int]:
    pred_events = np.asarray(pred_events, dtype=np.int32)
    true_events = np.asarray(true_events, dtype=np.int32)
    candidates = []
    for pred_idx, pred in enumerate(pred_events.tolist()):
        for true_idx, true in enumerate(true_events.tolist()):
            diff = abs(int(pred) - int(true))
            if diff <= tolerance:
                candidates.append((diff, pred_idx, true_idx, int(pred) - int(true)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_pred = set()
    matched_true = set()
    offsets = []
    for _, pred_idx, true_idx, offset in candidates:
        if pred_idx in matched_pred or true_idx in matched_true:
            continue
        matched_pred.add(pred_idx)
        matched_true.add(true_idx)
        offsets.append(offset)
    return offsets


def evaluate_event_sequences(
    sequence_scores: dict[str, np.ndarray],
    sequence_labels: dict[str, np.ndarray],
    threshold: float,
    tolerance: int,
    min_distance: int,
    prominence: float = 0.0,
) -> EventMetrics:
    total_pred = 0
    total_true = 0
    total_match = 0
    all_offsets: list[int] = []
    beat_scores = []
    beat_labels = []

    for sample_id, scores in sequence_scores.items():
        labels = np.asarray(sequence_labels[sample_id], dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        beat_scores.append(scores)
        beat_labels.append(labels)

        pred_events = extract_events(scores, threshold=threshold, min_distance=min_distance, prominence=prominence)
        true_events = np.flatnonzero(labels > 0.5).astype(np.int32)
        offsets = greedy_match(pred_events, true_events, tolerance=tolerance)

        total_pred += int(pred_events.size)
        total_true += int(true_events.size)
        total_match += len(offsets)
        all_offsets.extend(offsets)

    precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    recall = float(total_match / total_true) if total_true > 0 else 0.0
    denom = precision + recall
    f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
    mean_offset = float(np.mean(np.abs(all_offsets))) if all_offsets else None

    all_scores = np.concatenate(beat_scores) if beat_scores else np.zeros(0, dtype=np.float32)
    all_targets = np.concatenate(beat_labels) if beat_labels else np.zeros(0, dtype=np.float32)
    ap_targets = (all_targets > 0.5).astype(np.int32)
    if ap_targets.size > 0 and np.any(ap_targets > 0):
        avg_precision = float(average_precision_score(ap_targets, all_scores))
    else:
        avg_precision = 0.0

    return EventMetrics(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        mean_offset=mean_offset,
        matches=int(total_match),
        pred_events=int(total_pred),
        true_events=int(total_true),
        average_precision=avg_precision,
    )


def search_best_threshold(
    sequence_scores: dict[str, np.ndarray],
    sequence_labels: dict[str, np.ndarray],
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    prominence: float = 0.0,
) -> EventMetrics:
    best = None
    for threshold in thresholds.tolist():
        metrics = evaluate_event_sequences(
            sequence_scores=sequence_scores,
            sequence_labels=sequence_labels,
            threshold=float(threshold),
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )
        if best is None:
            best = metrics
            continue
        current_key = (metrics.f1, metrics.precision, -float(metrics.mean_offset or 1e9))
        best_key = (best.f1, best.precision, -float(best.mean_offset or 1e9))
        if current_key > best_key:
            best = metrics
    if best is None:
        raise ValueError("threshold search received an empty threshold grid")
    return best


def search_threshold_with_min_precision(
    sequence_scores: dict[str, np.ndarray],
    sequence_labels: dict[str, np.ndarray],
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    min_precision: float,
    prominence: float = 0.0,
) -> EventMetrics:
    best_meeting_floor = None
    best_fallback = None
    for threshold in thresholds.tolist():
        metrics = evaluate_event_sequences(
            sequence_scores=sequence_scores,
            sequence_labels=sequence_labels,
            threshold=float(threshold),
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )
        current_recall_key = (
            metrics.recall,
            metrics.precision,
            -float(metrics.mean_offset or 1e9),
            -metrics.threshold,
        )
        current_precision_key = (
            metrics.precision,
            metrics.recall,
            -float(metrics.mean_offset or 1e9),
            -metrics.threshold,
        )
        if metrics.precision >= min_precision:
            if best_meeting_floor is None:
                best_meeting_floor = metrics
            else:
                best_key = (
                    best_meeting_floor.recall,
                    best_meeting_floor.precision,
                    -float(best_meeting_floor.mean_offset or 1e9),
                    -best_meeting_floor.threshold,
                )
                if current_recall_key > best_key:
                    best_meeting_floor = metrics
        if best_fallback is None:
            best_fallback = metrics
        else:
            best_key = (
                best_fallback.precision,
                best_fallback.recall,
                -float(best_fallback.mean_offset or 1e9),
                -best_fallback.threshold,
            )
            if current_precision_key > best_key:
                best_fallback = metrics
    if best_meeting_floor is not None:
        return best_meeting_floor
    if best_fallback is not None:
        return best_fallback
    raise ValueError("threshold search received an empty threshold grid")


def evaluate_labeled_event_sequences(
    sequence_scores: dict[str, np.ndarray],
    sequence_pred_labels: dict[str, np.ndarray],
    sequence_true_labels: dict[str, np.ndarray],
    positive_classes: tuple[int, ...] | list[int],
    threshold: float,
    tolerance: int,
    min_distance: int,
    prominence: float = 0.0,
) -> LabeledEventMetrics:
    positive_classes = tuple(int(v) for v in positive_classes)
    class_matches = {cls: 0 for cls in positive_classes}
    class_pred_events = {cls: 0 for cls in positive_classes}
    class_true_events = {cls: 0 for cls in positive_classes}
    offsets: list[int] = []

    for sample_id, scores in sequence_scores.items():
        scores = np.asarray(scores, dtype=np.float32)
        pred_labels = np.asarray(sequence_pred_labels[sample_id], dtype=np.int32)
        true_labels = np.asarray(sequence_true_labels[sample_id], dtype=np.int32)

        pred_events = extract_events(scores, threshold=threshold, min_distance=min_distance, prominence=prominence)
        pred_events = pred_events[(pred_events >= 0) & (pred_events < pred_labels.shape[0])]
        for cls in positive_classes:
            pred_cls_events = pred_events[pred_labels[pred_events] == cls]
            true_cls_events = np.flatnonzero(true_labels == cls).astype(np.int32)
            matched_offsets = greedy_match(pred_cls_events, true_cls_events, tolerance=tolerance)
            class_pred_events[cls] += int(pred_cls_events.size)
            class_true_events[cls] += int(true_cls_events.size)
            class_matches[cls] += int(len(matched_offsets))
            offsets.extend(matched_offsets)

    class_precision = {}
    class_recall = {}
    class_f1 = {}
    for cls in positive_classes:
        precision = float(class_matches[cls] / class_pred_events[cls]) if class_pred_events[cls] > 0 else 0.0
        recall = float(class_matches[cls] / class_true_events[cls]) if class_true_events[cls] > 0 else 0.0
        denom = precision + recall
        f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
        class_precision[cls] = precision
        class_recall[cls] = recall
        class_f1[cls] = f1

    macro_precision = float(np.mean([class_precision[cls] for cls in positive_classes])) if positive_classes else 0.0
    macro_recall = float(np.mean([class_recall[cls] for cls in positive_classes])) if positive_classes else 0.0
    macro_f1 = float(np.mean([class_f1[cls] for cls in positive_classes])) if positive_classes else 0.0

    total_match = int(sum(class_matches.values()))
    total_pred = int(sum(class_pred_events.values()))
    total_true = int(sum(class_true_events.values()))
    micro_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    micro_recall = float(total_match / total_true) if total_true > 0 else 0.0
    denom = micro_precision + micro_recall
    micro_f1 = float(2.0 * micro_precision * micro_recall / denom) if denom > 0 else 0.0
    mean_offset = float(np.mean(np.abs(offsets))) if offsets else None

    return LabeledEventMetrics(
        threshold=float(threshold),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        mean_offset=mean_offset,
        class_precision=class_precision,
        class_recall=class_recall,
        class_f1=class_f1,
        class_matches=class_matches,
        class_pred_events=class_pred_events,
        class_true_events=class_true_events,
    )
