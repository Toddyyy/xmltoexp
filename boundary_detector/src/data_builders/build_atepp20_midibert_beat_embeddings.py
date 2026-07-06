from __future__ import annotations

import importlib.util
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig


ROOT = Path(__file__).resolve().parents[2]
MIDIBERT_ROOT = ROOT / "MIDI-BERT-CP"
DICT_PATH = ROOT / "models" / "midibert_hf" / "CP.pkl"
CKPT_PATH = ROOT / "models" / "midibert_hf" / "model_best_bert.ckpt"
DATA = ROOT / "data"
BASE_SCRIPT = ROOT / "src" / "experiments" / "run_atepp20_l2plus_weighted_target_experiment.py"
LABEL_DIR = DATA / "labels" / "atepp20_performer_levels"
REGENERATED_NOTE_DIR = DATA / "features" / "atepp20_regenerated_note_feats"
OUT_DIR = DATA / "features" / "atepp20_midibert_beat_embeddings"

MAX_SEQ_LEN = 512
STRIDE = 256
TICKS_PER_QUARTER = 480
POSITION_FRACTION = 16
DURATION_BINS = np.arange(60, 3841, 60, dtype=int)


def load_atepp_base():
    spec = importlib.util.spec_from_file_location("atepp20_base_for_midibert", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["atepp20_base_for_midibert"] = module
    spec.loader.exec_module(module)
    return module


def load_midibert():
    sys.path.insert(0, str(MIDIBERT_ROOT.resolve()))
    from MidiBERT.model import MidiBert

    with DICT_PATH.open("rb") as f:
        e2w, w2e = pickle.load(f)
    config = BertConfig(
        max_position_embeddings=MAX_SEQ_LEN,
        position_embedding_type="relative_key_query",
        hidden_size=768,
    )
    model = MidiBert(config, e2w, w2e)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected MidiBERT state keys: {unexpected[:10]}")
    model.eval()
    return model, e2w, missing


def piece_id_from_npz(path: Path) -> str:
    match = re.match(r"(.+)_\d+_L[1-6]\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse ATEPP piece id from {path.name}")
    return match.group(1)


def load_score_npz_by_piece() -> dict[str, Path]:
    return {
        path.name.removesuffix("_note_feats.npz"): path
        for path in sorted(REGENERATED_NOTE_DIR.glob("*_note_feats.npz"))
    }


def token_id(e2w: dict, kind: str, value: str | int) -> int:
    key = f"{kind} {value}"
    if key in e2w[kind]:
        return int(e2w[kind][key])
    if kind == "Pitch":
        pitch = int(value)
        for fallback in (max(0, min(127, pitch)), max(22, min(109, pitch))):
            key = f"Pitch {fallback}"
            if key in e2w[kind]:
                return int(e2w[kind][key])
    raise KeyError(key)


def _segment_for_beat(data, beat: int) -> dict[str, float | int] | None:
    starts = np.asarray(data["segment_global_start_beat"], dtype=np.int32)
    lengths = np.asarray(data["segment_num_beats"], dtype=np.int32)
    if starts.size == 0:
        return None
    idx = np.where((beat >= starts) & (beat < starts + lengths))[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    return {
        "index": int(np.asarray(data["segment_index"], dtype=np.int32)[i]),
        "numerator": int(np.asarray(data["segment_numerator"], dtype=np.int32)[i]),
        "denominator": int(np.asarray(data["segment_denominator"], dtype=np.int32)[i]),
        "beat_unit": float(np.asarray(data["segment_beat_unit"], dtype=np.float32)[i]),
        "local_start": int(np.asarray(data["segment_local_start_beat"], dtype=np.int32)[i]),
        "global_start": int(starts[i]),
    }


def _note_bar_position(data, feat: np.ndarray, beat: int) -> tuple[tuple[int, int], float]:
    mixed = bool(np.asarray(data["mixed_meter_by_segment"]).item()) if "mixed_meter_by_segment" in data.files else False
    pos_ql = max(float(feat[2]), 0.0)
    if mixed:
        segment = _segment_for_beat(data, beat)
        if segment is None:
            numerator, beat_unit, local_beat, segment_index = 4, 1.0, int(beat), 0
        else:
            numerator = max(int(segment["numerator"]), 1)
            beat_unit = max(float(segment["beat_unit"]), 1e-9)
            local_beat = int(segment["local_start"]) + int(beat) - int(segment["global_start"])
            segment_index = int(segment["index"])
    else:
        numerator = int(np.asarray(data["time_signature_numerator"]).item()) if "time_signature_numerator" in data.files else 4
        denominator = int(np.asarray(data["time_signature_denominator"]).item()) if "time_signature_denominator" in data.files else 4
        if numerator <= 0 or denominator <= 0:
            numerator, denominator = 4, 4
        beat_unit = 4.0 / float(denominator)
        local_beat = int(beat)
        segment_index = 1
    offset_in_beat = pos_ql - math.floor(pos_ql / beat_unit) * beat_unit
    offset_in_beat = min(max(float(offset_in_beat), 0.0), beat_unit)
    in_bar_ql = float(local_beat % numerator) * beat_unit + offset_in_beat
    bar_key = (segment_index, int(local_beat // max(numerator, 1)))
    return bar_key, in_bar_ql / max(float(numerator) * beat_unit, 1e-9)


def notes_to_cp_tokens(note_feats: np.ndarray, beat_ids: np.ndarray, num_beats: int, e2w: dict, data):
    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    note_feats = note_feats[valid]
    beat_ids = beat_ids[valid]
    order = np.lexsort((note_feats[:, 0], note_feats[:, 2]))
    note_feats = note_feats[order]
    beat_ids = beat_ids[order]

    positions = np.linspace(0.0, 1.0, POSITION_FRACTION, endpoint=False)
    tokens = []
    token_beats = []
    prev_bar = None
    for feat, beat in zip(note_feats, beat_ids):
        pitch = int(round(float(feat[0])))
        duration_ql = max(float(feat[1]), 1.0 / 16.0)
        bar, in_bar_frac = _note_bar_position(data, feat, int(beat))
        pos_idx = int(np.argmin(np.abs(positions - in_bar_frac))) + 1
        pos_idx = max(1, min(POSITION_FRACTION, pos_idx))
        dur_ticks = int(round(duration_ql * TICKS_PER_QUARTER))
        dur_idx = int(np.argmin(np.abs(DURATION_BINS - dur_ticks)))
        bar_value = "New" if bar != prev_bar else "Continue"
        prev_bar = bar
        tokens.append(
            [
                token_id(e2w, "Bar", bar_value),
                token_id(e2w, "Position", f"{pos_idx}/16"),
                token_id(e2w, "Pitch", pitch),
                token_id(e2w, "Duration", dur_idx),
            ]
        )
        token_beats.append(int(beat))
    return np.asarray(tokens, dtype=np.int64), np.asarray(token_beats, dtype=np.int32)


@torch.no_grad()
def embed_tokens(model, e2w, tokens: np.ndarray, device: torch.device) -> np.ndarray:
    if tokens.size == 0:
        return np.zeros((0, int(model.hidden_size)), dtype=np.float32)
    model.to(device)
    n = tokens.shape[0]
    hidden_sum = np.zeros((n, int(model.hidden_size)), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    pad = np.array(
        [
            e2w["Bar"]["Bar <PAD>"],
            e2w["Position"]["Position <PAD>"],
            e2w["Pitch"]["Pitch <PAD>"],
            e2w["Duration"]["Duration <PAD>"],
        ],
        dtype=np.int64,
    )
    starts = list(range(0, max(n - 1, 0), STRIDE))
    if not starts:
        starts = [0]
    if starts[-1] + MAX_SEQ_LEN < n:
        starts.append(max(0, n - MAX_SEQ_LEN))
    starts = sorted(set(starts))
    for start in starts:
        end = min(start + MAX_SEQ_LEN, n)
        length = end - start
        window = np.tile(pad[None, :], (MAX_SEQ_LEN, 1))
        window[:length] = tokens[start:end]
        attn = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        attn[:length] = 1
        input_ids = torch.from_numpy(window[None]).to(device)
        attn_mask = torch.from_numpy(attn[None]).to(device)
        out = model(input_ids, attn_mask=attn_mask, output_hidden_states=True)
        last = out.last_hidden_state[0, :length].detach().cpu().numpy().astype(np.float32)
        hidden_sum[start:end] += last
        counts[start:end] += 1.0
    return hidden_sum / np.maximum(counts[:, None], 1.0)


def pool_to_beats(token_hidden: np.ndarray, token_beats: np.ndarray, num_beats: int) -> np.ndarray:
    beat_hidden = np.zeros((num_beats, token_hidden.shape[1]), dtype=np.float32)
    counts = np.zeros(num_beats, dtype=np.float32)
    for hidden, beat in zip(token_hidden, token_beats):
        if 0 <= int(beat) < num_beats:
            beat_hidden[int(beat)] += hidden
            counts[int(beat)] += 1.0
    nonempty = counts > 0
    beat_hidden[nonempty] /= counts[nonempty, None]
    return beat_hidden


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, e2w, missing = load_midibert()
    atepp = load_atepp_base()
    pieces, labels, _ = atepp.load_l2plus_weighted_labels()
    piece_paths = load_score_npz_by_piece()
    rows = []
    for piece in pieces:
        out_path = OUT_DIR / f"{piece}_midibert_beat_embeddings.npz"
        if out_path.exists():
            continue
        num_beats = int(len(labels[piece]))
        path = piece_paths.get(piece)
        if path is None:
            raise FileNotFoundError(f"Missing regenerated note_feats for {piece}")
        with np.load(path, allow_pickle=True) as data:
            note_feats = np.asarray(data["note_feats"], dtype=np.float32)
            beat_ids = np.asarray(data["beat_ids"], dtype=np.int32)
            score_num_beats = int(data["num_beats"])
            if score_num_beats != num_beats:
                raise ValueError(
                    f"Beat length mismatch for {piece}: labels={num_beats}, regenerated={score_num_beats}"
                )
            tokens, token_beats = notes_to_cp_tokens(note_feats, beat_ids, num_beats, e2w, data=data)
        token_hidden = embed_tokens(model, e2w, tokens, device)
        beat_hidden = pool_to_beats(token_hidden, token_beats, num_beats)
        rows.append(
            {
                "piece": piece,
                "num_beats": num_beats,
                "tokens": int(tokens.shape[0]),
                "source_npz": str(path),
                "status": "ok",
            }
        )
        np.savez_compressed(
            out_path,
            piece_id=piece,
            beat_embeddings=beat_hidden.astype(np.float32),
            token_count=int(rows[-1]["tokens"]),
            num_beats=int(num_beats),
            source_npz=str(rows[-1]["source_npz"]),
            status=str(rows[-1]["status"]),
        )
        print(f"{piece}: beats={num_beats} tokens={rows[-1]['tokens']} status={rows[-1]['status']}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "build_summary.csv", index=False)
    (OUT_DIR / "metadata.txt").write_text(
        f"checkpoint={CKPT_PATH}\ndictionary={DICT_PATH}\nmissing_keys={missing}\nmax_seq_len={MAX_SEQ_LEN}\nstride={STRIDE}\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
