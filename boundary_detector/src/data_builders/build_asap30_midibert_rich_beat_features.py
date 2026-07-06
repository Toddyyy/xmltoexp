from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig


ROOT = Path(__file__).resolve().parents[2]
BUILDERS = ROOT / "src" / "data_builders"
DATA = ROOT / "data"
MIDIBERT_ROOT = ROOT / "MIDI-BERT-CP"
DICT_PATH = ROOT / "models" / "midibert_hf" / "CP.pkl"
CKPT_PATH = ROOT / "models" / "midibert_hf" / "model_best_bert.ckpt"
TOKEN_SCRIPT = BUILDERS / "build_atepp20_midibert_beat_embeddings.py"
RICH_SCRIPT = BUILDERS / "build_mazurkabl_midibert_rich_beat_features.py"
TOP_N = int(os.environ.get("ASAP_TOP_N", "30"))
DATASET_NAME = f"asap{TOP_N}"
NOTE_DIR = DATA / "features" / f"{DATASET_NAME}_score_note_feats"
LABEL_DIR = DATA / "labels" / f"{DATASET_NAME}_tempo_boundary_labels" / f"beat_data_asap_top{TOP_N}_performer_levels"
OUT_DIR = DATA / "features" / f"{DATASET_NAME}_midibert_rich_beat_features"

HIDDEN_DIM = 768


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


token_builder = load_module(f"{DATASET_NAME}_token_builder", TOKEN_SCRIPT)
rich_builder = load_module(f"{DATASET_NAME}_rich_builder", RICH_SCRIPT)


def load_midibert():
    sys.path.insert(0, str(MIDIBERT_ROOT.resolve()))
    from MidiBERT.model import MidiBert

    with DICT_PATH.open("rb") as f:
        e2w, w2e = pickle.load(f)
    config = BertConfig(
        max_position_embeddings=token_builder.MAX_SEQ_LEN,
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


def load_pieces() -> list[str]:
    return sorted({token_builder.piece_id_from_npz(path) for path in LABEL_DIR.glob("*_L2.npz")})


def sorted_valid_notes(note_feats: np.ndarray, beat_ids: np.ndarray, num_beats: int) -> np.ndarray:
    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    notes = note_feats[valid]
    if len(notes) == 0:
        return notes.astype(np.float32)
    order = np.lexsort((notes[:, 0], notes[:, 2]))
    return notes[order].astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not NOTE_DIR.exists():
        raise FileNotFoundError(f"Missing {NOTE_DIR}; run build_asap30_score_features.py first.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, e2w, missing = load_midibert()
    rows = []
    for piece in load_pieces():
        out_path = OUT_DIR / f"{piece}_midibert_rich_beat_features.npz"
        if out_path.exists():
            with np.load(out_path) as existing:
                rows.append(
                    {
                        "piece": piece,
                        "num_beats": int(existing["num_beats"]),
                        "tokens": int(existing["token_count"]),
                        "feature_dim": int(existing["feature_dim"]),
                        "out": str(out_path),
                        "status": "existing",
                    }
                )
            print(f"{piece}: existing {out_path}", flush=True)
            continue
        note_path = NOTE_DIR / f"{piece}_note_feats.npz"
        if not note_path.exists():
            raise FileNotFoundError(note_path)
        with np.load(note_path, allow_pickle=True) as data:
            note_feats = np.asarray(data["note_feats"], dtype=np.float32)
            beat_ids = np.asarray(data["beat_ids"], dtype=np.int32)
            num_beats = int(data["num_beats"])
            tokens, token_beats = token_builder.notes_to_cp_tokens(note_feats, beat_ids, num_beats, e2w, data=data)
        token_hidden = token_builder.embed_tokens(model, e2w, tokens, device)
        notes = sorted_valid_notes(note_feats, beat_ids, num_beats)
        rich, stats = rich_builder.build_rich_beat_features(notes, token_hidden, token_beats, num_beats)
        np.savez_compressed(
            out_path,
            piece_id=piece,
            rich_beat_features=rich.astype(np.float32),
            token_count=int(tokens.shape[0]),
            num_beats=int(num_beats),
            feature_dim=int(rich.shape[1]),
            source_npz=str(note_path),
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
        print(f"{piece}: beats={num_beats} tokens={tokens.shape[0]} rich_dim={rich.shape[1]}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "build_summary.csv", index=False)
    (OUT_DIR / "metadata.txt").write_text(
        f"checkpoint={CKPT_PATH}\ndictionary={DICT_PATH}\nmissing_keys={missing}\nrich_dim={HIDDEN_DIM * 6 + rich_builder.RICH_SCALAR_DIM}\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
