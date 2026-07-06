from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
MIDIBERT_ROOT = ROOT / "MIDI-BERT-CP"
DICT_PATH = ROOT / "models" / "midibert_hf" / "CP.pkl"
CKPT_PATH = ROOT / "models" / "midibert_hf" / "model_best_bert.ckpt"
MEAN_BUILD_SCRIPT = MIREX / "build_mazurkabl_midibert_beat_embeddings.py"
LABEL_DIR = MIREX / "beat_data_mazurka_performer_levels"
OUT_DIR = MIREX / "mazurkabl_midibert_rich_beat_features_meter34"

HIDDEN_DIM = 768
RICH_SCALAR_DIM = 7


def load_mean_builder():
    spec = importlib.util.spec_from_file_location("mazurka_midibert_mean_builder", MEAN_BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MEAN_BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_midibert_mean_builder"] = module
    spec.loader.exec_module(module)
    return module


builder = load_mean_builder()


def load_midibert():
    sys.path.insert(0, str(MIDIBERT_ROOT.resolve()))
    from MidiBERT.model import MidiBert

    with DICT_PATH.open("rb") as f:
        e2w, w2e = pickle.load(f)
    config = BertConfig(
        max_position_embeddings=builder.MAX_SEQ_LEN,
        position_embedding_type="relative_key_query",
        hidden_size=HIDDEN_DIM,
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
    return builder.piece_id_from_npz(path)


def load_score_npz_by_piece() -> dict[str, Path]:
    out = {}
    for path in sorted(LABEL_DIR.glob("*_L2.npz")):
        piece = piece_id_from_npz(path)
        if piece not in out:
            out[piece] = path
    return out


def _mean_or_zero(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((HIDDEN_DIM,), dtype=np.float32)
    return values.mean(axis=0).astype(np.float32)


def _weighted_mean_or_zero(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if values.size == 0 or float(weights.sum()) <= 0:
        return np.zeros((HIDDEN_DIM,), dtype=np.float32)
    return (values * weights[:, None]).sum(axis=0).astype(np.float32) / max(float(weights.sum()), 1e-9)


def build_rich_beat_features(
    note_feats: np.ndarray,
    token_hidden: np.ndarray,
    token_beats: np.ndarray,
    num_beats: int,
) -> tuple[np.ndarray, dict[str, int]]:
    if note_feats.shape[0] != token_hidden.shape[0] or note_feats.shape[0] != token_beats.shape[0]:
        raise ValueError(
            f"note/token length mismatch: notes={note_feats.shape[0]} hidden={token_hidden.shape[0]} beats={token_beats.shape[0]}"
        )
    out = np.zeros((num_beats, HIDDEN_DIM * 6 + RICH_SCALAR_DIM), dtype=np.float32)
    onset_pos = note_feats[:, 2].astype(np.float32)
    duration = np.maximum(note_feats[:, 1].astype(np.float32), 0.0)
    end_pos = onset_pos + duration
    pitch = note_feats[:, 0].astype(np.float32)

    for beat in range(num_beats):
        beat_start = float(beat)
        beat_end = beat_start + 1.0
        onset_mask = token_beats == beat
        overlap = np.maximum(0.0, np.minimum(end_pos, beat_end) - np.maximum(onset_pos, beat_start))
        all_mask = overlap > 1e-7
        sustain_mask = all_mask & ~onset_mask

        offset = 0
        out[beat, offset : offset + HIDDEN_DIM] = _mean_or_zero(token_hidden[onset_mask])
        offset += HIDDEN_DIM
        out[beat, offset : offset + HIDDEN_DIM] = _mean_or_zero(token_hidden[sustain_mask])
        offset += HIDDEN_DIM
        out[beat, offset : offset + HIDDEN_DIM] = _mean_or_zero(token_hidden[all_mask])
        offset += HIDDEN_DIM

        if np.any(all_mask):
            all_indices = np.flatnonzero(all_mask)
            highest = all_indices[int(np.argmax(pitch[all_indices]))]
            lowest = all_indices[int(np.argmin(pitch[all_indices]))]
            out[beat, offset : offset + HIDDEN_DIM] = token_hidden[highest]
            offset += HIDDEN_DIM
            out[beat, offset : offset + HIDDEN_DIM] = token_hidden[lowest]
            offset += HIDDEN_DIM
            pitch_span = float(pitch[all_indices].max() - pitch[all_indices].min())
        else:
            offset += HIDDEN_DIM * 2
            pitch_span = 0.0

        out[beat, offset : offset + HIDDEN_DIM] = _weighted_mean_or_zero(token_hidden[all_mask], overlap[all_mask])
        offset += HIDDEN_DIM

        overlap_count = int(np.count_nonzero(all_mask))
        onset_count = int(np.count_nonzero(onset_mask))
        sustain_count = int(np.count_nonzero(sustain_mask))
        scalars = np.asarray(
            [
                np.log1p(overlap_count),
                np.log1p(onset_count),
                np.log1p(sustain_count),
                pitch_span / 88.0,
                1.0 if overlap_count == 0 else 0.0,
                1.0 if onset_count == 0 else 0.0,
                1.0 if sustain_count > 0 and onset_count == 0 else 0.0,
            ],
            dtype=np.float32,
        )
        out[beat, offset : offset + RICH_SCALAR_DIM] = scalars

    stats = {
        "zero_onset_beats": int(np.count_nonzero(out[:, :HIDDEN_DIM].sum(axis=1) == 0.0)),
        "rest_beats": int(np.count_nonzero(out[:, HIDDEN_DIM * 6 + 4] > 0.5)),
    }
    return out, stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, e2w, missing = load_midibert()
    rows = []
    for piece, path in load_score_npz_by_piece().items():
        out_path = OUT_DIR / f"{piece}_midibert_rich_beat_features.npz"
        with np.load(path) as data:
            raw_note_feats = np.asarray(data["note_feats"], dtype=np.float32)
            raw_beat_ids = np.asarray(data["beat_ids"], dtype=np.int32)
            num_beats = int(data["num_beats"])
        tokens, token_beats = builder.notes_to_cp_tokens(raw_note_feats, raw_beat_ids, num_beats, e2w)
        valid = (raw_beat_ids >= 0) & (raw_beat_ids < num_beats)
        note_feats = raw_note_feats[valid]
        order = np.lexsort((note_feats[:, 0], note_feats[:, 2]))
        note_feats = note_feats[order]
        token_hidden = builder.embed_tokens(model, e2w, tokens, device)
        rich, stats = build_rich_beat_features(note_feats, token_hidden, token_beats, num_beats)
        np.savez_compressed(
            out_path,
            piece_id=piece,
            rich_beat_features=rich.astype(np.float32),
            token_count=int(tokens.shape[0]),
            num_beats=int(num_beats),
            feature_dim=int(rich.shape[1]),
            source_npz=str(path),
        )
        row = {
            "piece": piece,
            "num_beats": int(num_beats),
            "tokens": int(tokens.shape[0]),
            "feature_dim": int(rich.shape[1]),
            "out": str(out_path),
            **stats,
        }
        rows.append(row)
        print(f"{piece}: beats={num_beats} tokens={tokens.shape[0]} rich_dim={rich.shape[1]}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "build_summary.csv", index=False)
    (OUT_DIR / "metadata.txt").write_text(
        f"checkpoint={CKPT_PATH}\ndictionary={DICT_PATH}\nmissing_keys={missing}\nrich_dim={HIDDEN_DIM * 6 + RICH_SCALAR_DIM}\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
