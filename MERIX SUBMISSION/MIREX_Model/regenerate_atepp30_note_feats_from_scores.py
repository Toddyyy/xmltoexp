from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIREX_ROOT = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BASE_SCRIPT = MIREX_ROOT / "regenerate_atepp20_note_feats_from_scores.py"


def load_base():
    spec = importlib.util.spec_from_file_location("atepp20_note_regen_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["atepp20_note_regen_base"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
base.MANIFEST_PATH = base.METER_AUTO_ROOT / "outputs" / "atepp30_manifest.csv"
base.BEAT_TABLE_PATH = base.METER_AUTO_ROOT / "outputs" / "atepp30_beat_table.csv.gz"
base.OUT_DIR = MIREX_ROOT / "atepp30_regenerated_note_feats"


if __name__ == "__main__":
    base.main()
