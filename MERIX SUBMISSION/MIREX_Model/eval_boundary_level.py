import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import find_peaks

from dataset_beat import BeatBoundaryDataset
from infer_beat import build_model, load_config
from train_beat import load_piece_split


def piece_id_from_path(path: Path, cfg) -> str:
    stem = path.stem
    regex = cfg.get("data", {}).get("piece_id_regex")
    if regex:
        m = re.search(regex, stem)
        if not m and "\\\\" in regex:
            try:
                m = re.search(regex.encode("utf-8").decode("unicode_escape"), stem)
            except Exception:
                m = None
        if m:
            return m.group(1) if m.groups() else m.group(0)
    delim = cfg.get("data", {}).get("piece_id_delim")
    if delim and delim in stem:
        return stem.split(delim)[0]
    return stem


def build_eval_dataset(cfg, level=None):
    data_cfg = cfg["data"]
    dataset = BeatBoundaryDataset(
        data_dir=data_cfg["data_dir"],
        file_ext=data_cfg["file_ext"],
        max_len=data_cfg.get("max_len"),
        sequence_length=None,
        stride=None,
        beat_sequence_length=None,
        beat_stride=None,
        drop_short=False,
        position_mode=data_cfg.get("position_mode", "absolute"),
        use_base_features_only=data_cfg.get("use_base_features_only", False),
        label_mode="ratio",
        dist_min_dist=data_cfg.get("dist_min_dist", 6),
        dist_height=data_cfg.get("dist_height", 0.15),
        dist_prominence=data_cfg.get("dist_prominence", 0.05),
        dist_tau=data_cfg.get("dist_tau", 4.0),
        add_beat_pos=data_cfg.get("add_beat_pos", False),
        max_samples=data_cfg.get("max_samples"),
        value_ranges=data_cfg.get("value_ranges"),
        label_binarize_threshold=data_cfg.get("label_binarize_threshold"),
        performer_id_regex=data_cfg.get("performer_id_regex"),
    )
    if level is not None:
        level_tag = f"_L{int(level)}"
        dataset.samples = [s for s in dataset.samples if Path(s["path"]).stem.endswith(level_tag)]
        if not dataset.samples:
            raise ValueError(f"No samples found for level {level} in {dataset.data_dir}")
    return dataset


def build_beat_windows(num_beats: int, window_beats: int, stride: int):
    if num_beats <= 0:
        return []
    if window_beats <= 0 or stride <= 0:
        raise ValueError("window_beats and stride must be > 0")
    if num_beats <= window_beats:
        return [(0, num_beats)]
    starts = list(range(0, num_beats - window_beats + 1, stride))
    last_start = num_beats - window_beats
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(s, s + window_beats) for s in starts]


def slice_by_beats(note_feats, beat_ids, beat_start, beat_end, max_len):
    if beat_end <= beat_start:
        return None, None
    mask = (beat_ids >= beat_start) & (beat_ids < beat_end)
    if not np.any(mask):
        return None, None
    feats = note_feats[mask]
    ids = beat_ids[mask] - beat_start
    if max_len is not None and feats.shape[0] > max_len:
        feats = feats[:max_len]
        ids = ids[:max_len]
    return feats, ids


def predict_full_piece(model, cfg, item, device, output_head="dist", window_beats=None, window_stride=None):
    feats = item["note_feats"].cpu().numpy().astype(np.float32)
    beat_ids = item["beat_ids"].cpu().numpy().astype(np.int64)
    num_beats = int(item["num_beats"])
    include_empty_beats = cfg.get("model", {}).get("include_empty_beats", False)
    max_len = cfg.get("data", {}).get("max_len")
    performer_ids = None
    if cfg.get("model", {}).get("performer_cond", False):
        performer_ids = torch.tensor([int(item.get("performer_id", 0))], device=device, dtype=torch.long)

    if window_beats is None:
        with torch.no_grad():
            feats_t = torch.tensor(feats, device=device).unsqueeze(0)
            beat_ids_t = torch.tensor(beat_ids, device=device).unsqueeze(0)
            attn_mask = beat_ids_t >= 0
            logits, _ = model(
                feats_t,
                beat_ids=beat_ids_t,
                num_beats=num_beats,
                performer_ids=performer_ids,
                attn_mask=attn_mask,
                labels=None,
                output_head=output_head,
            )
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        if not include_empty_beats:
            valid_beats = np.zeros(num_beats, dtype=bool)
            valid_beats[np.unique(beat_ids[beat_ids >= 0])] = True
            probs = np.where(valid_beats, probs, 0.0)
        return probs.astype(np.float32)

    if window_stride is None:
        window_stride = window_beats
    windows = build_beat_windows(num_beats, int(window_beats), int(window_stride))
    sum_probs = np.zeros(num_beats, dtype=np.float64)
    count_probs = np.zeros(num_beats, dtype=np.int64)

    for start, end in windows:
        sub_feats, sub_ids = slice_by_beats(feats, beat_ids, start, end, max_len)
        if sub_feats is None or sub_ids is None or sub_feats.shape[0] == 0:
            continue
        feats_t = torch.tensor(sub_feats, device=device).unsqueeze(0)
        beat_ids_t = torch.tensor(sub_ids, device=device).unsqueeze(0)
        attn_mask = beat_ids_t >= 0
        with torch.no_grad():
            logits, _ = model(
                feats_t,
                beat_ids=beat_ids_t,
                num_beats=end - start,
                performer_ids=performer_ids,
                attn_mask=attn_mask,
                labels=None,
                output_head=output_head,
            )
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        idx = np.arange(start, end)
        if include_empty_beats:
            sum_probs[idx] += probs
            count_probs[idx] += 1
        else:
            valid = np.zeros(end - start, dtype=bool)
            if sub_ids.size > 0:
                valid[np.unique(sub_ids[sub_ids >= 0])] = True
            sum_probs[idx[valid]] += probs[valid]
            count_probs[idx[valid]] += 1

    out = np.zeros(num_beats, dtype=np.float32)
    seen = count_probs > 0
    out[seen] = (sum_probs[seen] / count_probs[seen]).astype(np.float32)
    return out


def peak_pick(y, min_dist=6, height=None, prominence=None):
    kwargs = {"distance": max(int(min_dist), 1)}
    if height is not None:
        kwargs["height"] = float(height)
    if prominence is not None:
        kwargs["prominence"] = float(prominence)
    peaks, _ = find_peaks(np.asarray(y, dtype=float), **kwargs)
    return peaks.astype(int)


def true_peaks_from_labels(labels, binary_threshold=0.5, min_dist=6, height=0.15, prominence=0.05):
    labels = np.asarray(labels, dtype=float)
    uniq = np.unique(labels)
    is_binaryish = uniq.size <= 3 and np.all(np.isin(np.round(uniq, 6), [0.0, 1.0]))
    if is_binaryish:
        return np.where(labels > float(binary_threshold))[0].astype(int)
    return peak_pick(labels, min_dist=min_dist, height=height, prominence=prominence)


def match_peaks(pred_peaks, true_peaks, tolerance):
    pred_peaks = np.asarray(pred_peaks, dtype=int)
    true_peaks = np.asarray(true_peaks, dtype=int)
    if pred_peaks.size == 0 and true_peaks.size == 0:
        return 0, 0, 0, []
    candidates = []
    for i, p in enumerate(pred_peaks):
        for j, t in enumerate(true_peaks):
            d = abs(int(p) - int(t))
            if d <= tolerance:
                candidates.append((d, i, j))
    candidates.sort()
    used_pred = set()
    used_true = set()
    matches = []
    for d, i, j in candidates:
        if i in used_pred or j in used_true:
            continue
        used_pred.add(i)
        used_true.add(j)
        matches.append((int(pred_peaks[i]), int(true_peaks[j]), int(d)))
    tp = len(matches)
    fp = int(pred_peaks.size) - tp
    fn = int(true_peaks.size) - tp
    return tp, fp, fn, matches


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Boundary-level evaluation on full-piece predictions.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--split_file", required=True, help="Piece-level split YAML")
    parser.add_argument("--level", type=int, required=True, help="Evaluate only files with suffix _L{level}")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Which split to evaluate")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--output_dir", default=None, help="Directory to save summary and per-file metrics")
    parser.add_argument("--tolerance", type=int, default=2, help="Beat tolerance for matching predicted and true peaks")
    parser.add_argument("--window_beats", type=int, default=None, help="Override eval window length in beats")
    parser.add_argument("--window_stride", type=int, default=None, help="Override eval window stride in beats")
    parser.add_argument("--pred_min_dist", type=int, default=None, help="Pred peak min distance")
    parser.add_argument("--pred_height", type=float, default=None, help="Pred peak height threshold")
    parser.add_argument("--pred_prominence", type=float, default=None, help="Pred peak prominence threshold")
    parser.add_argument("--true_binary_threshold", type=float, default=0.5, help="Threshold for binary true labels")
    parser.add_argument("--true_min_dist", type=int, default=None, help="True peak min distance if non-binary labels")
    parser.add_argument("--true_height", type=float, default=None, help="True peak height if non-binary labels")
    parser.add_argument("--true_prominence", type=float, default=None, help="True peak prominence if non-binary labels")
    parser.add_argument("--head", choices=["dist", "prob"], default="dist", help="Which model head to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of files to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = build_eval_dataset(cfg, level=args.level)
    split_meta = load_piece_split(args.split_file)
    target_pieces = split_meta[args.split]
    selected = [s for s in dataset.samples if piece_id_from_path(Path(s["path"]), cfg) in target_pieces]
    if args.limit is not None:
        selected = selected[: max(int(args.limit), 0)]
    if not selected:
        raise ValueError(f"No files selected for split={args.split}, level={args.level}")
    dataset.samples = selected

    if cfg.get("model", {}).get("performer_cond"):
        if dataset.num_performers <= 0:
            raise ValueError("performer_cond is enabled but no performer IDs were found in filenames.")
        cfg["model"]["performer_vocab_size"] = int(dataset.num_performers) + 1

    model = build_model(cfg, input_dim=dataset.feature_dim).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data_cfg = cfg.get("data", {})
    window_beats = args.window_beats
    if window_beats is None:
        window_beats = data_cfg.get("eval_beat_sequence_length", data_cfg.get("beat_sequence_length"))
    window_stride = args.window_stride
    if window_stride is None:
        window_stride = data_cfg.get("eval_beat_stride", data_cfg.get("beat_stride"))
    if window_beats is not None and window_stride is None:
        window_stride = window_beats

    pred_min_dist = args.pred_min_dist if args.pred_min_dist is not None else data_cfg.get("dist_min_dist", 6)
    pred_height = args.pred_height if args.pred_height is not None else data_cfg.get("dist_height", 0.15)
    pred_prominence = (
        args.pred_prominence if args.pred_prominence is not None else data_cfg.get("dist_prominence", 0.05)
    )
    true_min_dist = args.true_min_dist if args.true_min_dist is not None else data_cfg.get("dist_min_dist", 6)
    true_height = args.true_height if args.true_height is not None else data_cfg.get("dist_height", 0.15)
    true_prominence = (
        args.true_prominence if args.true_prominence is not None else data_cfg.get("dist_prominence", 0.05)
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).resolve().parent / f"boundary_eval_{args.split}_L{args.level}"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_file_rows = []
    total_tp = total_fp = total_fn = 0

    for i in range(len(dataset)):
        item = dataset[i]
        sample = dataset.samples[i]
        path = Path(sample["path"])
        piece = piece_id_from_path(path, cfg)

        probs = predict_full_piece(
            model,
            cfg,
            item,
            device=device,
            output_head=args.head,
            window_beats=window_beats,
            window_stride=window_stride,
        )
        true_labels = item["labels"].cpu().numpy().astype(np.float32)

        pred_peaks = peak_pick(
            probs,
            min_dist=pred_min_dist,
            height=pred_height,
            prominence=pred_prominence,
        )
        true_peaks = true_peaks_from_labels(
            true_labels,
            binary_threshold=args.true_binary_threshold,
            min_dist=true_min_dist,
            height=true_height,
            prominence=true_prominence,
        )
        tp, fp, fn, matches = match_peaks(pred_peaks, true_peaks, args.tolerance)
        precision, recall, f1 = prf(tp, fp, fn)
        mae = float(np.mean([m[2] for m in matches])) if matches else None

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_file_rows.append(
            {
                "file": path.name,
                "piece": piece,
                "performer_id": int(item.get("performer_id", 0)),
                "num_beats": int(item["num_beats"]),
                "pred_count": int(len(pred_peaks)),
                "true_count": int(len(true_peaks)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mae_matched_beats": mae,
                "pred_peaks": json.dumps(pred_peaks.tolist()),
                "true_peaks": json.dumps(true_peaks.tolist()),
            }
        )

    macro_precision = float(np.mean([r["precision"] for r in per_file_rows])) if per_file_rows else 0.0
    macro_recall = float(np.mean([r["recall"] for r in per_file_rows])) if per_file_rows else 0.0
    macro_f1 = float(np.mean([r["f1"] for r in per_file_rows])) if per_file_rows else 0.0
    micro_precision, micro_recall, micro_f1 = prf(total_tp, total_fp, total_fn)

    per_file_csv = output_dir / "per_file_metrics.csv"
    with per_file_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "piece",
                "performer_id",
                "num_beats",
                "pred_count",
                "true_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mae_matched_beats",
                "pred_peaks",
                "true_peaks",
            ],
        )
        writer.writeheader()
        writer.writerows(per_file_rows)

    summary = {
        "config": str(Path(args.config).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "level": int(args.level),
        "split": args.split,
        "num_files": len(per_file_rows),
        "window_beats": None if window_beats is None else int(window_beats),
        "window_stride": None if window_stride is None else int(window_stride),
        "tolerance": int(args.tolerance),
        "head": args.head,
        "pred_peak_params": {
            "min_dist": int(pred_min_dist),
            "height": None if pred_height is None else float(pred_height),
            "prominence": None if pred_prominence is None else float(pred_prominence),
        },
        "true_peak_params": {
            "binary_threshold": float(args.true_binary_threshold),
            "min_dist": int(true_min_dist),
            "height": None if true_height is None else float(true_height),
            "prominence": None if true_prominence is None else float(true_prominence),
        },
        "micro": {
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1": float(micro_f1),
        },
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "per_file_csv": str(per_file_csv.resolve()),
    }

    summary_yaml = output_dir / "summary.yaml"
    with summary_yaml.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"Saved per-file metrics to {per_file_csv}")
    print(
        f"Boundary-level eval | split={args.split} level={args.level} "
        f"| micro_f1={micro_f1:.4f} macro_f1={macro_f1:.4f} files={len(per_file_rows)}"
    )


if __name__ == "__main__":
    main()
