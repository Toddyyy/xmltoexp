import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from model.model_beat import BeatBoundaryConfig, BeatBoundaryModel


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def normalize_feats(feats: np.ndarray, value_ranges: dict) -> np.ndarray:
    names = [
        "pitch_midi",
        "duration",
        "position",
        "part_idx",
        "is_accent",
        "is_staccato",
    ]
    feats = feats.copy()
    for i, name in enumerate(names):
        if name not in value_ranges:
            continue
        min_v, max_v = value_ranges[name]
        denom = max(float(max_v) - float(min_v), 1e-6)
        vals = 2.0 * (feats[:, i] - float(min_v)) / denom - 1.0
        feats[:, i] = np.clip(vals, -1.0, 1.0)
    return feats


def build_model(cfg, input_dim: int) -> BeatBoundaryModel:
    label_mode = cfg.get("data", {}).get("label_mode")
    dual_head = cfg["model"].get("dual_head")
    if dual_head is None:
        dual_head = label_mode == "dual"
    model_cfg = BeatBoundaryConfig(
        input_dim=input_dim,
        d_model=cfg["model"]["d_model"],
        beat_encoder_type=cfg["model"].get("beat_encoder_type", "transformer"),
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_len=cfg["model"]["max_len"],
        fixed_beats=cfg["model"].get("fixed_beats"),
        include_empty_beats=cfg["model"].get("include_empty_beats", False),
        dual_head=dual_head,
        note_rnn_hidden=cfg["model"].get("note_rnn_hidden"),
        note_rnn_layers=cfg["model"].get("note_rnn_layers", 1),
        note_rnn_dropout=cfg["model"].get("note_rnn_dropout", cfg["model"]["dropout"]),
        beat_rnn_hidden=cfg["model"].get("beat_rnn_hidden"),
        beat_rnn_layers=cfg["model"].get("beat_rnn_layers", cfg["model"]["num_layers"]),
        beat_rnn_dropout=cfg["model"].get("beat_rnn_dropout", cfg["model"]["dropout"]),
        performer_cond=cfg["model"].get("performer_cond", False),
        performer_emb_dim=cfg["model"].get("performer_emb_dim", 32),
        performer_vocab_size=cfg["model"].get("performer_vocab_size", 0),
    )
    pos_weight = cfg.get("training", {}).get("pos_weight")
    loss_type = cfg.get("training", {}).get("loss_type", "bce")
    prob_loss_type = cfg.get("training", {}).get("prob_loss_type", "bce")
    prob_pos_weight = cfg.get("training", {}).get("prob_pos_weight")
    prob_loss_weight = cfg.get("training", {}).get("prob_loss_weight", 1.0)
    return BeatBoundaryModel(
        model_cfg,
        pos_weight=pos_weight,
        loss_type=loss_type,
        prob_loss_type=prob_loss_type,
        prob_pos_weight=prob_pos_weight,
        prob_loss_weight=prob_loss_weight,
    )


def apply_position_mode(feats: np.ndarray, position_mode: str) -> np.ndarray:
    if feats.size == 0 or feats.shape[1] < 3:
        return feats
    if position_mode == "absolute":
        return feats
    feats = feats.copy()
    if position_mode == "zero":
        feats[:, 2] = 0.0
        return feats
    pos = feats[:, 2]
    pmin = float(pos.min())
    pmax = float(pos.max())
    denom = max(pmax - pmin, 1e-6)
    feats[:, 2] = (pos - pmin) / denom
    return feats


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


def slice_by_beats(
    note_feats: np.ndarray,
    beat_ids: np.ndarray,
    beat_start: int,
    beat_end: int,
    max_len: int | None,
):
    if beat_end <= beat_start:
        return None, None
    beat_start = max(int(beat_start), 0)
    beat_end = max(int(beat_end), beat_start)
    mask = (beat_ids >= beat_start) & (beat_ids < beat_end)
    if not np.any(mask):
        return None, None
    feats = note_feats[mask]
    ids = beat_ids[mask] - beat_start
    if max_len is not None and feats.shape[0] > max_len:
        feats = feats[:max_len]
        ids = ids[:max_len]
    return feats, ids


def apply_pos_weight_calibration(logits: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if pos_weight is None:
        return logits
    if pos_weight <= 0:
        raise ValueError("pos_weight must be > 0 for calibration.")
    return logits - math.log(float(pos_weight))


def main():
    parser = argparse.ArgumentParser(description="Infer beat boundary probabilities from note-level npz.")
    parser.add_argument("--config", required=True, help="Path to config YAML (model params)")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input_npz", required=True, help="Path to input npz (note_feats, beat_ids)")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument(
        "--window_beats",
        type=int,
        default=None,
        help="Beat window length for sliding inference (default: data.beat_sequence_length).",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Beat window stride (default: data.beat_stride or window_beats).",
    )
    parser.add_argument(
        "--calibrate_pos_weight",
        action="store_true",
        help="Calibrate probabilities by removing pos_weight bias.",
    )
    parser.add_argument(
        "--calibration_weight",
        type=float,
        default=None,
        help="Override pos_weight for calibration (default: training.pos_weight).",
    )
    parser.add_argument(
        "--head",
        choices=["dist", "prob"],
        default=None,
        help="Select which head to output (dist or prob).",
    )
    parser.add_argument(
        "--performer_id",
        type=int,
        default=None,
        help="Optional performer id for conditional logits (0 or None = no conditioning).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    data = np.load(args.input_npz)
    if "note_feats" not in data or "beat_ids" not in data:
        raise KeyError("input_npz must contain note_feats and beat_ids")

    note_feats = data["note_feats"].astype(np.float32)
    beat_ids = data["beat_ids"].astype(np.int64)
    num_beats = int(data["num_beats"]) if "num_beats" in data else int(beat_ids.max() + 1)

    use_base_features_only = cfg.get("data", {}).get("use_base_features_only", False)
    if use_base_features_only:
        if note_feats.shape[1] < 6:
            raise ValueError("note_feats has fewer columns than base features.")
        note_feats = note_feats[:, :6]

    if note_feats.shape[1] != cfg["model"]["input_dim"]:
        raise ValueError(
            f"input_dim mismatch: npz has {note_feats.shape[1]}, config expects {cfg['model']['input_dim']}"
        )

    value_ranges = cfg.get("data", {}).get("value_ranges", {})
    position_mode = cfg.get("data", {}).get("position_mode", "absolute")

    model = build_model(cfg, input_dim=note_feats.shape[1]).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    fixed_beats = cfg.get("model", {}).get("fixed_beats")
    window_beats = args.window_beats
    include_empty_beats = cfg.get("model", {}).get("include_empty_beats", False)
    dual_head = cfg.get("model", {}).get("dual_head")
    if dual_head is None:
        dual_head = cfg.get("data", {}).get("label_mode") == "dual"

    output_head = args.head
    if output_head is None:
        output_head = cfg.get("model", {}).get("output_head")
    if output_head is None:
        output_head = "prob" if dual_head else "dist"
    if output_head == "prob" and not dual_head:
        raise ValueError("Requested prob head but dual_head is disabled in config.")
    performer_ids = None
    if args.performer_id is not None:
        performer_ids = torch.tensor([int(args.performer_id)], device=device, dtype=torch.long)

    loss_type = cfg.get("training", {}).get("loss_type", "bce")
    prob_loss_type = cfg.get("training", {}).get("prob_loss_type", "bce")
    pos_weight = cfg.get("training", {}).get("pos_weight")
    prob_pos_weight = cfg.get("training", {}).get("prob_pos_weight", pos_weight)

    if window_beats is None:
        window_beats = cfg.get("data", {}).get("beat_sequence_length")

    if window_beats is None:
        if fixed_beats is not None:
            if int(fixed_beats) < num_beats:
                print(f"Warning: fixed_beats {fixed_beats} < num_beats {num_beats}, truncating.")
            num_beats = int(fixed_beats)

        with torch.no_grad():
            feats = apply_position_mode(note_feats, position_mode)
            if value_ranges:
                feats = normalize_feats(feats, value_ranges)
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
            if args.calibrate_pos_weight:
                weight = args.calibration_weight
                if weight is None:
                    weight = prob_pos_weight if output_head == "prob" else pos_weight
                if (output_head == "prob" and prob_loss_type == "bce") or (
                    output_head == "dist" and loss_type == "bce"
                ):
                    logits = apply_pos_weight_calibration(logits, weight)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        if not include_empty_beats:
            valid_beats = np.zeros(num_beats, dtype=bool)
            valid_beats[np.unique(beat_ids[beat_ids >= 0])] = True
            probs = np.where(valid_beats, probs, 0.0)
    else:
        window_stride = args.window_stride
        if window_stride is None:
            window_stride = cfg.get("data", {}).get("beat_stride")
        if window_stride is None:
            window_stride = window_beats

        windows = build_beat_windows(num_beats, int(window_beats), int(window_stride))
        max_len = cfg.get("data", {}).get("max_len")
        sum_probs = np.zeros(num_beats, dtype=np.float64)
        count_probs = np.zeros(num_beats, dtype=np.int64)

        for start, end in windows:
            feats, ids = slice_by_beats(note_feats, beat_ids, start, end, max_len)
            if feats is None or ids is None or feats.shape[0] == 0:
                continue
            feats = apply_position_mode(feats, position_mode)
            if value_ranges:
                feats = normalize_feats(feats, value_ranges)
            feats_t = torch.tensor(feats, device=device).unsqueeze(0)
            beat_ids_t = torch.tensor(ids, device=device).unsqueeze(0)
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
                if args.calibrate_pos_weight:
                    weight = args.calibration_weight
                    if weight is None:
                        weight = prob_pos_weight if output_head == "prob" else pos_weight
                    if (output_head == "prob" and prob_loss_type == "bce") or (
                        output_head == "dist" and loss_type == "bce"
                    ):
                        logits = apply_pos_weight_calibration(logits, weight)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            idx = np.arange(start, end)
            if include_empty_beats:
                sum_probs[idx] += probs
                count_probs[idx] += 1
            else:
                valid = np.zeros(end - start, dtype=bool)
                if ids.size > 0:
                    valid[np.unique(ids[ids >= 0])] = True
                sum_probs[idx[valid]] += probs[valid]
                count_probs[idx[valid]] += 1

        probs = np.zeros(num_beats, dtype=np.float64)
        seen = count_probs > 0
        probs[seen] = sum_probs[seen] / count_probs[seen]
        probs = probs.astype(np.float32)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beat_index", "boundary_probability"])
        for i, p in enumerate(probs):
            writer.writerow([i, float(p)])

    print(f"Saved predictions to {out_path} | beats={num_beats}")


if __name__ == "__main__":
    main()
