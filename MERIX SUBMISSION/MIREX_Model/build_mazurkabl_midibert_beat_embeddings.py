from __future__ import annotations

import importlib.util
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig


ROOT = Path(__file__).resolve().parents[2]
MIDIBERT_ROOT = ROOT / "MIDI-BERT-CP"
DICT_PATH = ROOT / "models" / "midibert_hf" / "CP.pkl"
CKPT_PATH = ROOT / "models" / "midibert_hf" / "model_best_bert.ckpt"
LABEL_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "beat_data_mazurka_performer_levels"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_midibert_beat_embeddings_meter34"

MAX_SEQ_LEN = 512
STRIDE = 256
TICKS_PER_QUARTER = 480
QUARTERS_PER_BAR = 3
TICKS_PER_BAR = QUARTERS_PER_BAR * TICKS_PER_QUARTER
POSITION_FRACTION = 16
DURATION_BINS = np.arange(60, 3841, 60, dtype=int)


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
    match = re.match(r"(M\d+-\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse piece id from {path.name}")
    return match.group(1)


def load_score_npz_by_piece() -> dict[str, Path]:
    out = {}
    for path in sorted(LABEL_DIR.glob("*_L2.npz")):
        piece = piece_id_from_npz(path)
        if piece not in out:
            out[piece] = path
    return out


def token_id(e2w: dict, kind: str, value: str | int) -> int:
    if isinstance(value, int):
        key = f"{kind} {value}"
    else:
        key = f"{kind} {value}"
    if key in e2w[kind]:
        return int(e2w[kind][key])
    # For the HF dictionary, pitch ids include 0..127. For the original dict,
    # piano range starts at 22. Clamp if needed.
    if kind == "Pitch":
        pitch = int(value)
        for fallback in (max(0, min(127, pitch)), max(22, min(109, pitch))):
            key = f"Pitch {fallback}"
            if key in e2w[kind]:
                return int(e2w[kind][key])
    raise KeyError(key)


def notes_to_cp_tokens(note_feats: np.ndarray, beat_ids: np.ndarray, num_beats: int, e2w: dict):
    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    note_feats = note_feats[valid]
    beat_ids = beat_ids[valid]
    order = np.lexsort((note_feats[:, 0], note_feats[:, 2]))
    note_feats = note_feats[order]
    beat_ids = beat_ids[order]

    tokens = []
    token_beats = []
    prev_bar = None
    for feat, beat in zip(note_feats, beat_ids):
        pitch = int(round(float(feat[0])))
        duration_ql = max(float(feat[1]), 1.0 / 16.0)
        # MazurkaBL note onsets are 1-based beat positions. Convert to a
        # zero-based 3/4 score timeline before making CP Bar/Position tokens.
        pos_ql = max(float(feat[2]) - 1.0, 0.0)
        tick = int(round(pos_ql * TICKS_PER_QUARTER))
        bar = tick // TICKS_PER_BAR
        in_bar = tick - bar * TICKS_PER_BAR
        pos_idx = int(np.argmin(np.abs(np.linspace(0, TICKS_PER_BAR, POSITION_FRACTION, endpoint=False) - in_bar))) + 1
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
        [e2w["Bar"]["Bar <PAD>"], e2w["Position"]["Position <PAD>"], e2w["Pitch"]["Pitch <PAD>"], e2w["Duration"]["Duration <PAD>"]],
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, e2w, missing = load_midibert()
    piece_paths = load_score_npz_by_piece()
    rows = []
    for piece, path in piece_paths.items():
        out_path = OUT_DIR / f"{piece}_midibert_beat_embeddings.npz"
        if out_path.exists():
            continue
        with np.load(path) as data:
            note_feats = np.asarray(data["note_feats"], dtype=np.float32)
            beat_ids = np.asarray(data["beat_ids"], dtype=np.int32)
            num_beats = int(data["num_beats"])
        tokens, token_beats = notes_to_cp_tokens(note_feats, beat_ids, num_beats, e2w)
        token_hidden = embed_tokens(model, e2w, tokens, device)
        beat_hidden = pool_to_beats(token_hidden, token_beats, num_beats)
        np.savez_compressed(
            out_path,
            piece_id=piece,
            beat_embeddings=beat_hidden.astype(np.float32),
            token_count=int(tokens.shape[0]),
            num_beats=int(num_beats),
            source_npz=str(path),
        )
        rows.append({"piece": piece, "num_beats": num_beats, "tokens": int(tokens.shape[0]), "out": str(out_path)})
        print(f"{piece}: beats={num_beats} tokens={tokens.shape[0]}")

    import pandas as pd

    pd.DataFrame(rows).to_csv(OUT_DIR / "build_summary.csv", index=False)
    (OUT_DIR / "metadata.txt").write_text(
        f"checkpoint={CKPT_PATH}\ndictionary={DICT_PATH}\nmissing_keys={missing}\nmax_seq_len={MAX_SEQ_LEN}\nstride={STRIDE}\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
