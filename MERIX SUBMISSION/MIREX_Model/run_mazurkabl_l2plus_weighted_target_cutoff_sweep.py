from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
OUT_ROOT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l2plus_weighted_target_cutoff_sweep"
CUTOFFS = [0.01, 0.005, 0.001]


def run_cutoff(cutoff: float) -> pd.Series:
    module_name = f"mazurkabl_weighted_cutoff_{str(cutoff).replace('.', 'p')}"
    spec = importlib.util.spec_from_file_location(module_name, BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.EVENT_MIN = float(cutoff)
    module.OUT_DIR = OUT_ROOT / f"event_min_{cutoff:g}".replace(".", "p")
    module.main()
    mean = pd.read_csv(module.OUT_DIR / "fold_mean.csv", index_col=0)["mean"]
    mean["event_min"] = float(cutoff)
    mean["out_dir"] = str(module.OUT_DIR)
    return mean


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cutoff in CUTOFFS:
        print(f"\n=== event_min={cutoff:g} ===")
        rows.append(run_cutoff(cutoff))
    summary = pd.DataFrame(rows)
    cols = ["event_min", *[c for c in summary.columns if c not in {"event_min", "out_dir"}], "out_dir"]
    summary = summary[cols]
    summary.to_csv(OUT_ROOT / "cutoff_sweep_summary.csv", index=False)
    print("\nCutoff sweep summary:")
    print(
        summary[
            [
                "event_min",
                "threshold_precision_tol1",
                "threshold_recall_tol1",
                "threshold_f1_tol1",
                "density_precision_tol1",
                "density_recall_tol1",
                "density_f1_tol1",
                "density_pred_events",
                "true_events",
            ]
        ].to_string(index=False)
    )
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
