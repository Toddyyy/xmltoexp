import argparse
import json
from pathlib import Path
import sys

try:
    from tokenizer.base_score_tokenizer import ScoreNoteToken
    from detokenizer.detokenizer import (
        RenderingPerformanceNoteToken,
        detokenize_to_midi,
    )
except ImportError as e:
    print(
        "CRITICAL ERROR: Could not import necessary modules from 'tokenizer' or 'detokenizer'."
    )
    print(f"  - Details: {e}")
    print(
        "  - Please ensure this script is run from your project's root directory, so that Python can find these packages."
    )
    sys.exit(1)


def load_data_from_json(
    json_path: str,
) -> tuple[list[ScoreNoteToken], list[RenderingPerformanceNoteToken], dict]:
    """
    Loads and parses all required sheet music and performance data from a single JSON file.

    Args:
    json_path: The path to the JSON file containing all the data.

    Returns:
    A tuple containing:
    - score_note_tokens: A list of score note tokens.
    - performance_predictions: A list of performance predictions.
    - metadata: A dictionary of metadata for the file.
    """
    print(f"Loading and parsing data from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    score_note_tokens = []
    performance_predictions = []

    if "note_by_note_results" not in data:
        raise KeyError(
            "The JSON file is missing the required 'note_by_note_results' field."
        )

    note_results = data["note_by_note_results"]

    for i, item in enumerate(note_results):
        score_data = item.get("score", {})
        perf_data = item.get("performance", {})
        seg_info = item.get("segment_info", {})

        score_note_tokens.append(
            ScoreNoteToken(
                pitch=score_data.get("pitch"),
                position=score_data.get("position"),
                duration=score_data.get("duration"),
                part_id=score_data.get("part_id"),
                is_staccato=score_data.get("is_staccato", False),
                is_accent=score_data.get("is_accent", False),
                xml_note_id=score_data.get("xml_note_id"),
                tie=score_data.get("tie"),
                fingering=score_data.get("fingering"),
            )
        )
        performance_predictions.append(
            RenderingPerformanceNoteToken(
                velocity=perf_data.get("velocity"),
                onset_deviation_in_sec=perf_data.get("onset_deviation_in_seconds"),
                duration_deviation_in_sec=perf_data.get(
                    "duration_deviation_in_seconds"
                ),
                local_tempo=perf_data.get("local_tempo"),
                sustain_level=perf_data.get("sustain_level"),
                segment_avg_tempo=seg_info.get("segment_avg_tempo"),
            )
        )

    metadata = data.get("metadata", {})
    return score_note_tokens, performance_predictions, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Render performance predictions from a JSON file into a MIDI file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the input JSON file containing both score and performance data.",
    )

    parser.add_argument(
        "output_midi", type=str, help="Path to save the generated MIDI file."
    )

    args = parser.parse_args()

    try:
        score_notes, perf_preds, metadata = load_data_from_json(args.json_file)
        print(f"-> Successfully loaded {len(score_notes)} notes.")

        try:
            chunk_size = metadata["generation_params"]["sequence_length"]
            print(f"Using chunk size defined in JSON metadata: {chunk_size}")
        except KeyError:
            print(
                "\nCRITICAL ERROR: Could not find 'sequence_length' in the JSON file's metadata."
            )
            print(
                "  - Path checked: data['metadata']['generation_params']['sequence_length']"
            )
            sys.exit(1)

        print("\nStarting detokenization process...")
        detokenize_to_midi(
            score_note_tokens=score_notes,
            performance_predictions=perf_preds,
            output_midi_path=args.output_midi,
            chunk_size=chunk_size,
        )
        print("\nDetokenization process completed.")
        print(f"Final MIDI file has been saved to: {Path(args.output_midi).resolve()}")

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input file not found at '{args.json_file}'")
    except (KeyError, TypeError) as e:
        print(
            f"CRITICAL ERROR: The JSON file seems to be malformed or missing a required key."
        )
        print(f"  - Details: {e}")
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
