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
    matches: float
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
    class_matches: dict[int, float]
    class_pred_events: dict[int, int]
    class_true_events: dict[int, int]


@dataclass
class UnionFrequencyMetrics:
    threshold: float
    union_precision: float
    frequency_weighted_precision: float
    consensus_precision: float
    union_recall: float
    union_f1: float
    weighted_recall: float
    consensus_recall: float
    mean_offset: float | None
    matches: float
    pred_events: int
    true_union_events: int
    true_consensus_events: int
    matched_weight: float
    total_weight: float
    matched_pred_weight: float


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


def extract_events_crf(
    scores: np.ndarray,
    threshold: float,
    min_distance: int = 1,
) -> np.ndarray:
    """Decode boundary events with a hard-spacing CRF/Viterbi state model.

    The emission score for selecting a beat is logit(score)-logit(threshold).
    The transition state stores how many beats remain before another boundary
    can be selected, so nearby duplicate boundaries are controlled by the
    decoder rather than by merging target labels.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return np.zeros(0, dtype=np.int32)

    spacing = max(int(min_distance), 1)
    eps = 1e-6
    clipped_scores = np.clip(scores.astype(np.float64), eps, 1.0 - eps)
    clipped_threshold = float(np.clip(float(threshold), eps, 1.0 - eps))
    emissions = np.log(clipped_scores / (1.0 - clipped_scores)) - np.log(
        clipped_threshold / (1.0 - clipped_threshold)
    )

    n = int(scores.size)
    n_states = spacing
    neg_inf = -1e18
    dp = np.full((n + 1, n_states), neg_inf, dtype=np.float64)
    back_state = np.full((n, n_states), -1, dtype=np.int32)
    back_event = np.zeros((n, n_states), dtype=bool)
    dp[0, 0] = 0.0

    for pos in range(n):
        for cooldown in range(n_states):
            base = dp[pos, cooldown]
            if base <= neg_inf / 2:
                continue

            # No boundary at this beat: cooldown moves one step toward zero.
            next_cooldown = max(cooldown - 1, 0)
            if base > dp[pos + 1, next_cooldown]:
                dp[pos + 1, next_cooldown] = base
                back_state[pos, next_cooldown] = cooldown
                back_event[pos, next_cooldown] = False

            # Boundary at this beat: only allowed when no previous event is too close.
            if cooldown == 0:
                next_cooldown = spacing - 1
                candidate = base + float(emissions[pos])
                if candidate > dp[pos + 1, next_cooldown]:
                    dp[pos + 1, next_cooldown] = candidate
                    back_state[pos, next_cooldown] = cooldown
                    back_event[pos, next_cooldown] = True

    state = int(np.argmax(dp[n]))
    events: list[int] = []
    for pos in range(n - 1, -1, -1):
        if back_event[pos, state]:
            events.append(pos)
        prev_state = int(back_state[pos, state])
        if prev_state < 0:
            break
        state = prev_state
    return np.asarray(sorted(events), dtype=np.int32)


def decode_events(
    scores: np.ndarray,
    threshold: float,
    min_distance: int = 1,
    prominence: float = 0.0,
    event_decoder: str = "peak",
) -> np.ndarray:
    decoder = str(event_decoder)
    if decoder == "peak":
        return extract_events(
            scores,
            threshold=threshold,
            min_distance=min_distance,
            prominence=prominence,
        )
    if decoder == "crf":
        return extract_events_crf(
            scores,
            threshold=threshold,
            min_distance=min_distance,
        )
    raise ValueError(f"Unsupported event_decoder: {event_decoder}")


def greedy_match_pairs(pred_events: np.ndarray, true_events: np.ndarray, tolerance: int) -> list[tuple[int, int, int]]:
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
    matches = []
    for _, pred_idx, true_idx, offset in candidates:
        if pred_idx in matched_pred or true_idx in matched_true:
            continue
        matched_pred.add(pred_idx)
        matched_true.add(true_idx)
        matches.append((pred_idx, true_idx, offset))
    return matches


def greedy_match(pred_events: np.ndarray, true_events: np.ndarray, tolerance: int) -> list[int]:
    return [offset for _, _, offset in greedy_match_pairs(pred_events, true_events, tolerance)]


def soft_match_credit(offset: int) -> float:
    return float(0.5 ** abs(int(offset)))


def greedy_soft_match_pairs(
    pred_events: np.ndarray,
    true_events: np.ndarray,
    tolerance: int,
) -> list[tuple[int, int, int, float]]:
    """Match events with geometric distance credit.

    Exact hits receive 1.0. One beat to either side receives 0.5, two beats
    receive 0.25, and so on. A positive tolerance still acts as a maximum
    distance; tolerance <= 0 means no hard cutoff.
    """
    pred_events = np.asarray(pred_events, dtype=np.int32)
    true_events = np.asarray(true_events, dtype=np.int32)
    max_distance = int(tolerance) if int(tolerance) > 0 else None
    candidates = []
    for pred_idx, pred in enumerate(pred_events.tolist()):
        for true_idx, true in enumerate(true_events.tolist()):
            offset = int(pred) - int(true)
            diff = abs(offset)
            if max_distance is not None and diff > max_distance:
                continue
            credit = soft_match_credit(offset)
            candidates.append((-credit, diff, pred_idx, true_idx, offset, credit))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    matched_pred = set()
    matched_true = set()
    matches: list[tuple[int, int, int, float]] = []
    for _, _, pred_idx, true_idx, offset, credit in candidates:
        if pred_idx in matched_pred or true_idx in matched_true:
            continue
        matched_pred.add(pred_idx)
        matched_true.add(true_idx)
        matches.append((pred_idx, true_idx, offset, float(credit)))
    return matches


def evaluate_event_sequences(
    sequence_scores: dict[str, np.ndarray],
    sequence_labels: dict[str, np.ndarray],
    threshold: float,
    tolerance: int,
    min_distance: int,
    prominence: float = 0.0,
    event_decoder: str = "peak",
) -> EventMetrics:
    total_pred = 0
    total_true = 0
    total_match = 0.0
    offset_weight_sum = 0.0
    beat_scores = []
    beat_labels = []

    for sample_id, scores in sequence_scores.items():
        labels = np.asarray(sequence_labels[sample_id], dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        beat_scores.append(scores)
        beat_labels.append(labels)

        pred_events = decode_events(
            scores,
            threshold=threshold,
            min_distance=min_distance,
            prominence=prominence,
            event_decoder=event_decoder,
        )
        true_events = np.flatnonzero(labels > 0.5).astype(np.int32)
        matches = greedy_soft_match_pairs(pred_events, true_events, tolerance=tolerance)

        total_pred += int(pred_events.size)
        total_true += int(true_events.size)
        total_match += float(sum(credit for _, _, _, credit in matches))
        offset_weight_sum += float(sum(abs(offset) * credit for _, _, offset, credit in matches))

    precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    recall = float(total_match / total_true) if total_true > 0 else 0.0
    denom = precision + recall
    f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
    mean_offset = float(offset_weight_sum / total_match) if total_match > 0.0 else None

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
        matches=float(total_match),
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
    event_decoder: str = "peak",
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
            event_decoder=event_decoder,
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
    event_decoder: str = "peak",
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
            event_decoder=event_decoder,
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
    event_decoder: str = "peak",
) -> LabeledEventMetrics:
    positive_classes = tuple(int(v) for v in positive_classes)
    class_matches = {cls: 0.0 for cls in positive_classes}
    class_pred_events = {cls: 0 for cls in positive_classes}
    class_true_events = {cls: 0 for cls in positive_classes}
    offset_weight_sum = 0.0
    total_match_credit = 0.0

    for sample_id, scores in sequence_scores.items():
        scores = np.asarray(scores, dtype=np.float32)
        pred_labels = np.asarray(sequence_pred_labels[sample_id], dtype=np.int32)
        true_labels = np.asarray(sequence_true_labels[sample_id], dtype=np.int32)

        pred_events = decode_events(
            scores,
            threshold=threshold,
            min_distance=min_distance,
            prominence=prominence,
            event_decoder=event_decoder,
        )
        pred_events = pred_events[(pred_events >= 0) & (pred_events < pred_labels.shape[0])]
        for cls in positive_classes:
            pred_cls_events = pred_events[pred_labels[pred_events] == cls]
            true_cls_events = np.flatnonzero(true_labels == cls).astype(np.int32)
            matches = greedy_soft_match_pairs(pred_cls_events, true_cls_events, tolerance=tolerance)
            class_pred_events[cls] += int(pred_cls_events.size)
            class_true_events[cls] += int(true_cls_events.size)
            match_credit = float(sum(credit for _, _, _, credit in matches))
            class_matches[cls] += match_credit
            total_match_credit += match_credit
            offset_weight_sum += float(sum(abs(offset) * credit for _, _, offset, credit in matches))

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

    total_match = float(sum(class_matches.values()))
    total_pred = int(sum(class_pred_events.values()))
    total_true = int(sum(class_true_events.values()))
    micro_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    micro_recall = float(total_match / total_true) if total_true > 0 else 0.0
    denom = micro_precision + micro_recall
    micro_f1 = float(2.0 * micro_precision * micro_recall / denom) if denom > 0 else 0.0
    mean_offset = float(offset_weight_sum / total_match_credit) if total_match_credit > 0.0 else None

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


def evaluate_union_frequency_sequences(
    sequence_scores: dict[str, np.ndarray],
    sequence_union_labels: dict[str, np.ndarray],
    sequence_frequency_targets: dict[str, np.ndarray],
    threshold: float,
    tolerance: int,
    min_distance: int,
    consensus_threshold: float = 0.5,
    prominence: float = 0.0,
    event_decoder: str = "peak",
) -> UnionFrequencyMetrics:
    total_pred = 0
    total_true_union = 0
    total_true_consensus = 0
    total_match = 0.0
    total_consensus_match = 0.0
    total_pred_consensus_match = 0.0
    matched_weight = 0.0
    total_weight = 0.0
    matched_pred_weight = 0.0
    offset_weight_sum = 0.0

    for sample_id, scores in sequence_scores.items():
        scores = np.asarray(scores, dtype=np.float32)
        union_labels = np.asarray(sequence_union_labels[sample_id], dtype=np.float32)
        freq_targets = np.asarray(sequence_frequency_targets[sample_id], dtype=np.float32)

        pred_events = decode_events(
            scores,
            threshold=threshold,
            min_distance=min_distance,
            prominence=prominence,
            event_decoder=event_decoder,
        )
        true_union_events = np.flatnonzero(union_labels > 0.5).astype(np.int32)
        true_consensus_events = np.flatnonzero(freq_targets >= float(consensus_threshold)).astype(np.int32)

        union_matches = greedy_soft_match_pairs(pred_events, true_union_events, tolerance=tolerance)
        consensus_matches = greedy_soft_match_pairs(pred_events, true_consensus_events, tolerance=tolerance)
        union_credit = float(sum(credit for _, _, _, credit in union_matches))
        consensus_credit = float(sum(credit for _, _, _, credit in consensus_matches))

        total_pred += int(pred_events.size)
        total_true_union += int(true_union_events.size)
        total_true_consensus += int(true_consensus_events.size)
        total_match += union_credit
        total_consensus_match += consensus_credit
        total_pred_consensus_match += consensus_credit
        offset_weight_sum += float(sum(abs(offset) * credit for _, _, offset, credit in union_matches))
        sample_match_weight = float(
            sum(freq_targets[true_union_events[true_idx]] * credit for _, true_idx, _, credit in union_matches)
        )
        matched_weight += sample_match_weight
        matched_pred_weight += sample_match_weight
        total_weight += float(freq_targets[true_union_events].sum())

    union_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    frequency_weighted_precision = float(matched_pred_weight / total_pred) if total_pred > 0 else 0.0
    consensus_precision = float(total_pred_consensus_match / total_pred) if total_pred > 0 else 0.0
    union_recall = float(total_match / total_true_union) if total_true_union > 0 else 0.0
    denom = union_precision + union_recall
    union_f1 = float(2.0 * union_precision * union_recall / denom) if denom > 0 else 0.0
    weighted_recall = float(matched_weight / total_weight) if total_weight > 0 else 0.0
    consensus_recall = float(total_consensus_match / total_true_consensus) if total_true_consensus > 0 else 0.0
    mean_offset = float(offset_weight_sum / total_match) if total_match > 0.0 else None

    return UnionFrequencyMetrics(
        threshold=float(threshold),
        union_precision=union_precision,
        frequency_weighted_precision=frequency_weighted_precision,
        consensus_precision=consensus_precision,
        union_recall=union_recall,
        union_f1=union_f1,
        weighted_recall=weighted_recall,
        consensus_recall=consensus_recall,
        mean_offset=mean_offset,
        matches=float(total_match),
        pred_events=int(total_pred),
        true_union_events=int(total_true_union),
        true_consensus_events=int(total_true_consensus),
        matched_weight=float(matched_weight),
        total_weight=float(total_weight),
        matched_pred_weight=float(matched_pred_weight),
    )


def evaluate_union_frequency_event_sets(
    sequence_pred_events: dict[str, np.ndarray],
    sequence_union_labels: dict[str, np.ndarray],
    sequence_frequency_targets: dict[str, np.ndarray],
    tolerance: int,
    threshold: float = 0.5,
    consensus_threshold: float = 0.5,
) -> UnionFrequencyMetrics:
    total_pred = 0
    total_true_union = 0
    total_true_consensus = 0
    total_match = 0.0
    total_consensus_match = 0.0
    total_pred_consensus_match = 0.0
    matched_weight = 0.0
    total_weight = 0.0
    matched_pred_weight = 0.0
    offset_weight_sum = 0.0

    for sample_id, pred_events in sequence_pred_events.items():
        pred_events = np.asarray(pred_events, dtype=np.int32)
        union_labels = np.asarray(sequence_union_labels[sample_id], dtype=np.float32)
        freq_targets = np.asarray(sequence_frequency_targets[sample_id], dtype=np.float32)

        pred_events = pred_events[(pred_events >= 0) & (pred_events < union_labels.shape[0])]
        pred_events = np.asarray(sorted(set(pred_events.tolist())), dtype=np.int32)
        true_union_events = np.flatnonzero(union_labels > 0.5).astype(np.int32)
        true_consensus_events = np.flatnonzero(freq_targets >= float(consensus_threshold)).astype(np.int32)

        union_matches = greedy_soft_match_pairs(pred_events, true_union_events, tolerance=tolerance)
        consensus_matches = greedy_soft_match_pairs(pred_events, true_consensus_events, tolerance=tolerance)
        union_credit = float(sum(credit for _, _, _, credit in union_matches))
        consensus_credit = float(sum(credit for _, _, _, credit in consensus_matches))

        total_pred += int(pred_events.size)
        total_true_union += int(true_union_events.size)
        total_true_consensus += int(true_consensus_events.size)
        total_match += union_credit
        total_consensus_match += consensus_credit
        total_pred_consensus_match += consensus_credit
        offset_weight_sum += float(sum(abs(offset) * credit for _, _, offset, credit in union_matches))
        sample_match_weight = float(
            sum(freq_targets[true_union_events[true_idx]] * credit for _, true_idx, _, credit in union_matches)
        )
        matched_weight += sample_match_weight
        matched_pred_weight += sample_match_weight
        total_weight += float(freq_targets[true_union_events].sum())

    union_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    frequency_weighted_precision = float(matched_pred_weight / total_pred) if total_pred > 0 else 0.0
    consensus_precision = float(total_pred_consensus_match / total_pred) if total_pred > 0 else 0.0
    union_recall = float(total_match / total_true_union) if total_true_union > 0 else 0.0
    denom = union_precision + union_recall
    union_f1 = float(2.0 * union_precision * union_recall / denom) if denom > 0 else 0.0
    weighted_recall = float(matched_weight / total_weight) if total_weight > 0 else 0.0
    consensus_recall = float(total_consensus_match / total_true_consensus) if total_true_consensus > 0 else 0.0
    mean_offset = float(offset_weight_sum / total_match) if total_match > 0.0 else None

    return UnionFrequencyMetrics(
        threshold=float(threshold),
        union_precision=union_precision,
        frequency_weighted_precision=frequency_weighted_precision,
        consensus_precision=consensus_precision,
        union_recall=union_recall,
        union_f1=union_f1,
        weighted_recall=weighted_recall,
        consensus_recall=consensus_recall,
        mean_offset=mean_offset,
        matches=float(total_match),
        pred_events=int(total_pred),
        true_union_events=int(total_true_union),
        true_consensus_events=int(total_true_consensus),
        matched_weight=float(matched_weight),
        total_weight=float(total_weight),
        matched_pred_weight=float(matched_pred_weight),
    )


def search_union_frequency_threshold(
    sequence_scores: dict[str, np.ndarray],
    sequence_union_labels: dict[str, np.ndarray],
    sequence_frequency_targets: dict[str, np.ndarray],
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    min_precision: float,
    consensus_threshold: float = 0.5,
    prominence: float = 0.0,
    primary_metric: str = "weighted_recall",
    precision_metric: str = "union_precision",
    min_union_precision: float | None = None,
    min_frequency_weighted_precision: float | None = None,
    min_consensus_precision: float | None = None,
    event_decoder: str = "peak",
) -> UnionFrequencyMetrics:
    best_meeting_floor = None
    best_fallback = None
    for threshold in thresholds.tolist():
        metrics = evaluate_union_frequency_sequences(
            sequence_scores=sequence_scores,
            sequence_union_labels=sequence_union_labels,
            sequence_frequency_targets=sequence_frequency_targets,
            threshold=float(threshold),
            tolerance=tolerance,
            min_distance=min_distance,
            consensus_threshold=consensus_threshold,
            prominence=prominence,
            event_decoder=event_decoder,
        )
        if primary_metric == "union_recall":
            primary_value = metrics.union_recall
        elif primary_metric == "consensus_recall":
            primary_value = metrics.consensus_recall
        else:
            primary_value = metrics.weighted_recall

        if precision_metric == "frequency_weighted_precision":
            precision_value = metrics.frequency_weighted_precision
            fallback_secondary = metrics.union_precision
        elif precision_metric == "consensus_precision":
            precision_value = metrics.consensus_precision
            fallback_secondary = metrics.union_precision
        else:
            precision_value = metrics.union_precision
            fallback_secondary = metrics.frequency_weighted_precision
        required_union_precision = float(min_union_precision) if min_union_precision is not None else 0.0
        required_frequency_weighted_precision = (
            float(min_frequency_weighted_precision) if min_frequency_weighted_precision is not None else 0.0
        )
        required_consensus_precision = (
            float(min_consensus_precision) if min_consensus_precision is not None else 0.0
        )
        floor_tuple = (
            metrics.union_precision >= required_union_precision,
            metrics.frequency_weighted_precision >= required_frequency_weighted_precision,
            metrics.consensus_precision >= required_consensus_precision,
        )
        floor_count = int(sum(1 for flag in floor_tuple if flag))
        current_weighted_key = (
            primary_value,
            precision_value,
            metrics.frequency_weighted_precision,
            metrics.consensus_precision,
            metrics.union_precision,
            metrics.consensus_recall,
            -float(metrics.mean_offset or 1e9),
            -metrics.threshold,
        )
        current_precision_key = (
            floor_count,
            int(floor_tuple[0]),
            int(floor_tuple[1]),
            int(floor_tuple[2]),
            precision_value,
            primary_value,
            fallback_secondary,
            metrics.frequency_weighted_precision,
            metrics.consensus_precision,
            metrics.union_precision,
            metrics.consensus_recall,
            -float(metrics.mean_offset or 1e9),
            -metrics.threshold,
        )
        if all(floor_tuple):
            if best_meeting_floor is None:
                best_meeting_floor = metrics
            else:
                best_key = (
                    (best_meeting_floor.union_recall if primary_metric == "union_recall" else best_meeting_floor.consensus_recall if primary_metric == "consensus_recall" else best_meeting_floor.weighted_recall),
                    (best_meeting_floor.frequency_weighted_precision if precision_metric == "frequency_weighted_precision" else best_meeting_floor.consensus_precision if precision_metric == "consensus_precision" else best_meeting_floor.union_precision),
                    best_meeting_floor.frequency_weighted_precision,
                    best_meeting_floor.consensus_precision,
                    best_meeting_floor.union_precision,
                    best_meeting_floor.consensus_recall,
                    -float(best_meeting_floor.mean_offset or 1e9),
                    -best_meeting_floor.threshold,
                )
                if current_weighted_key > best_key:
                    best_meeting_floor = metrics
        if best_fallback is None:
            best_fallback = metrics
        else:
            best_key = (
                int(
                    (best_fallback.union_precision >= required_union_precision)
                    + (best_fallback.frequency_weighted_precision >= required_frequency_weighted_precision)
                    + (best_fallback.consensus_precision >= required_consensus_precision)
                ),
                int(best_fallback.union_precision >= required_union_precision),
                int(best_fallback.frequency_weighted_precision >= required_frequency_weighted_precision),
                int(best_fallback.consensus_precision >= required_consensus_precision),
                (best_fallback.frequency_weighted_precision if precision_metric == "frequency_weighted_precision" else best_fallback.consensus_precision if precision_metric == "consensus_precision" else best_fallback.union_precision),
                (best_fallback.union_recall if primary_metric == "union_recall" else best_fallback.consensus_recall if primary_metric == "consensus_recall" else best_fallback.weighted_recall),
                (best_fallback.union_precision if precision_metric != "union_precision" else best_fallback.frequency_weighted_precision),
                best_fallback.frequency_weighted_precision,
                best_fallback.consensus_precision,
                best_fallback.union_precision,
                best_fallback.consensus_recall,
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
