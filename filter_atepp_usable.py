"""
Filter ATEPP v1.2 metadata to keep entries that:
- have a score file path,
- are not flagged as low-quality/noisy/applause/corrupted,
- and whose MIDI/score files actually exist on disk.

Outputs a CSV with the usable subset for score–performance alignment.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# Paths: adjust if your dataset is elsewhere.
METADATA_CSV = Path("ATEPP-master/ATEPP-metadata-1.2.csv")
MIDI_ROOT = Path("ATEPP-1.2/ATEPP-1.2")
# score files live alongside MIDI paths in ATEPP v1.2
SCORE_ROOT = MIDI_ROOT

# Quality labels to drop
QUALITY_EXCLUDE = {"low quality", "background noise", "applause", "corrupted"}


def has_nonempty(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def main():
    df = pd.read_csv(METADATA_CSV)
    if "quality" in df.columns:
        df["quality"] = df["quality"].fillna("").str.lower()
    else:
        df["quality"] = ""

    # Keep rows with a score path
    mask_score = df["score_path"].apply(has_nonempty)
    # Keep rows not in excluded quality
    mask_quality = ~df["quality"].isin(QUALITY_EXCLUDE)
    df = df[mask_score & mask_quality].copy()

    # Build absolute paths and check existence
    df["midi_abs"] = df["midi_path"].apply(lambda p: str((MIDI_ROOT / str(p)).resolve()))
    df["score_abs"] = df["score_path"].apply(lambda p: str((SCORE_ROOT / str(p)).resolve()))
    df["midi_exists"] = df["midi_abs"].apply(lambda p: Path(p).exists())
    df["score_exists"] = df["score_abs"].apply(lambda p: Path(p).exists())

    usable = df[df["midi_exists"] & df["score_exists"]].copy()

    out_csv = METADATA_CSV.parent / "usable_with_scores_v1.2.csv"
    usable.to_csv(out_csv, index=False)

    print(f"Total rows: {len(pd.read_csv(METADATA_CSV))}")
    print(f"After score+quality filter: {len(df)}")
    print(f"After file existence check: {len(usable)}")
    print(f"Saved to: {out_csv}")


if __name__ == "__main__":
    main()
