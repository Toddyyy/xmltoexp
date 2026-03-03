#!/usr/bin/env python3
"""
One-step script to generate MIDI directly from XML files.
This script combines the functionality of:
- test_score_tokenizer.py (XML -> score tokens)
- generate_overlap.py (score tokens -> performance predictions)
- run_detokenizer.py (performance -> MIDI)
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict

# Add MIREX to Python path to handle imports
current_dir = Path(__file__).parent
mirex_path = current_dir / "MIREX_Tokenizer"
if mirex_path.exists():
    sys.path.insert(0, str(mirex_path))
    print(f"Added MIREX path: {mirex_path}")

# Import from your existing modules (local to your repo)
from MIREX_Tokenizer.score_tokenizer import extract_score_tokens_from_xml, sort_score_tokens_by_position
from generate_overlap import ScorePerformanceGenerator

# Import MIREX modules with error handling
DETOKENIZER_AVAILABLE = False
try:
    from MIREX_Tokenizer.tokenizer.base_score_tokenizer import ScoreNoteToken
    from MIREX_Tokenizer.detokenizer.detokenizer import (
        RenderingPerformanceNoteToken,
        detokenize_to_midi,
    )
    DETOKENIZER_AVAILABLE = True
    print("✅ MIREX modules loaded successfully")
except ImportError as e:
    print(f"❌ Warning: MIREX modules not available: {e}")
    print("   Please ensure MIREX/ directory is present in your repo")

# Your existing tokenizer functions (fallback if MIREX not available)
try:
    from score_tokenizer import extract_score_tokens_from_xml as local_extract_tokens
    from score_tokenizer import sort_score_tokens_by_position as local_sort_tokens
    LOCAL_TOKENIZER_AVAILABLE = True
except ImportError:
    LOCAL_TOKENIZER_AVAILABLE = False


def convert_performance_json_to_tokens(performance_data: Dict) -> tuple[
    List[ScoreNoteToken], List[RenderingPerformanceNoteToken], Dict]:
    """
    Convert performance JSON data to token objects needed by detokenizer.
    This replicates the functionality from run_detokenizer.py
    """
    score_note_tokens = []
    performance_predictions = []

    if "note_by_note_results" not in performance_data:
        raise KeyError("The JSON data is missing the required 'note_by_note_results' field.")

    note_results = performance_data["note_by_note_results"]

    for i, item in enumerate(note_results):
        score_data = item.get("score", {})
        perf_data = item.get("performance", {})
        seg_info = item.get("segment_info", {})

        # Create ScoreNoteToken
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

        # Create RenderingPerformanceNoteToken
        performance_predictions.append(
            RenderingPerformanceNoteToken(
                velocity=perf_data.get("velocity"),
                onset_deviation_in_sec=perf_data.get("onset_deviation_in_seconds"),
                duration_deviation_in_sec=perf_data.get("duration_deviation_in_seconds"),
                local_tempo=perf_data.get("local_tempo"),
                sustain_level=perf_data.get("sustain_level"),
                segment_avg_tempo=seg_info.get("segment_avg_tempo"),
            )
        )

    metadata = performance_data.get("metadata", {})
    return score_note_tokens, performance_predictions, metadata


def generate_midi_from_xml(
        xml_path: str,
        output_midi_path: str,
        model_path: str,
        config_path: str = None,
        composer_id: int = 0,
        device: str = 'auto',
        temperature: float = 1.0,
        top_p: float = 0.9,
        sequence_length: int = 512,
        overlap_length: int = 0,
        use_predicted_tempo: bool = True,
        sort_by_position: bool = True,
        save_intermediate_json: bool = False,
        intermediate_json_path: str = None,
        save_performance_json: bool = False,
        performance_json_path: str = None
):
    """
    Generate MIDI directly from XML file.

    Args:
        xml_path: Path to input XML file
        output_midi_path: Path to save generated MIDI file
        model_path: Path to trained model checkpoint
        config_path: Path to model config file
        composer_id: Composer ID for generation
        device: Device to use for inference
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        sequence_length: Length of each sequence segment
        overlap_length: Length of overlap between segments
        use_predicted_tempo: Whether to use predicted tempo
        sort_by_position: Whether to sort tokens by position
        save_intermediate_json: Whether to save intermediate score tokens JSON
        intermediate_json_path: Path to save intermediate JSON (if None, auto-generate)
        save_performance_json: Whether to save performance predictions JSON
        performance_json_path: Path to save performance JSON (if None, auto-generate)

    Returns:
        str: Path to generated MIDI file
    """

    if not DETOKENIZER_AVAILABLE:
        raise ImportError("Detokenizer modules are not available. Cannot generate MIDI.")

    print(f"Generating MIDI directly from XML: {xml_path}")
    print(f"MIDI output will be saved to: {output_midi_path}")

    # Step 1: Extract score tokens from XML
    print("\nStep 1: Extracting score tokens from XML...")
    try:
        score_tokens = extract_score_tokens_from_xml(xml_path)
        print(f"✅ Extracted {len(score_tokens)} score tokens")
    except Exception as e:
        print(f"❌ Error extracting score tokens: {str(e)}")
        raise

    # Step 2: Sort tokens by position if requested
    if sort_by_position:
        print("\nStep 2: Sorting tokens by position...")
        score_tokens = sort_score_tokens_by_position(score_tokens)
        print("✅ Tokens sorted by position")
    else:
        print("\nStep 2: Skipping token sorting")

    # Step 3: Save intermediate JSON if requested
    temp_json_path = None
    if save_intermediate_json or intermediate_json_path:
        if intermediate_json_path is None:
            # Auto-generate intermediate path
            xml_file = Path(xml_path)
            intermediate_json_path = xml_file.parent / f"{xml_file.stem}_score_tokens.json"

        print(f"\nStep 3: Saving intermediate score tokens to: {intermediate_json_path}")

        intermediate_data = {
            'metadata': {
                'source_xml': str(xml_path),
                'total_notes': len(score_tokens),
                'extraction_method': 'MusicXMLTokenizer',
                'sorted_by_position': sort_by_position
            },
            'full_tokens': score_tokens
        }

        output_json_path = Path(intermediate_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        # os.makedirs(os.path.dirname(intermediate_json_path), exist_ok=True)
        with open(intermediate_json_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

        print(f"Intermediate JSON saved")
        temp_json_path = intermediate_json_path
    else:
        print("\nStep 3: Skipping intermediate JSON save")

    # Step 4: Create temporary JSON for generation if not saved permanently
    if temp_json_path is None:
        print("\nStep 4: Creating temporary JSON for generation...")
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_json_path = temp_file.name

        temp_data = {
            'metadata': {
                'source_xml': str(xml_path),
                'total_notes': len(score_tokens),
                'extraction_method': 'MusicXMLTokenizer',
                'sorted_by_position': sort_by_position
            },
            'full_tokens': score_tokens
        }

        json.dump(temp_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        print(f"Temporary JSON created: {temp_json_path}")
    else:
        print("\nStep 4: Using saved intermediate JSON for generation")

    # Step 5: Initialize generator
    print("\n🤖 Step 5: Initializing performance generator...")
    try:
        generator = ScorePerformanceGenerator(
            model_path=model_path,
            config_path=config_path,
            device=device,
            sequence_length=sequence_length,
            overlap_length=overlap_length,
            temperature=temperature,
            top_p=top_p,
            use_predicted_tempo=use_predicted_tempo
        )
        print("✅ Generator initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing generator: {str(e)}")
        # Clean up temporary file if created
        if not save_intermediate_json and not intermediate_json_path and temp_json_path:
            try:
                os.unlink(temp_json_path)
            except:
                pass
        raise

    # Step 6: Generate performance
    print("\nStep 6: Generating performance predictions...")

    # Determine performance JSON path
    perf_json_path = None
    if save_performance_json or performance_json_path:
        if performance_json_path is None:
            # Auto-generate performance path
            xml_file = Path(xml_path)
            performance_json_path = xml_file.parent / f"{xml_file.stem}_performance.json"
        perf_json_path = performance_json_path
    else:
        # Create temporary performance JSON
        temp_perf_file = tempfile.NamedTemporaryFile(mode='w', suffix='_performance.json', delete=False)
        perf_json_path = temp_perf_file.name
        temp_perf_file.close()

    try:
        generator.generate_from_json(
            input_json_path=temp_json_path,
            output_json_path=perf_json_path,
            composer_id=composer_id
        )
        print("✅ Performance generation completed successfully!")
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        raise

    # Step 7: Convert performance to MIDI
    print("\nStep 7: Converting performance to MIDI...")
    try:
        # Load performance data
        with open(perf_json_path, 'r', encoding='utf-8') as f:
            performance_data = json.load(f)

        # Convert to token objects
        score_notes, perf_preds, metadata = convert_performance_json_to_tokens(performance_data)
        print(f"✅ Loaded {len(score_notes)} notes for MIDI conversion")

        # Get chunk size from metadata
        try:
            chunk_size = metadata["generation_params"]["sequence_length"]
            print(f"Using chunk size from metadata: {chunk_size}")
        except KeyError:
            chunk_size = sequence_length
            print(f"Using fallback chunk size: {chunk_size}")

        # Create output directory if needed
        output_path = Path(output_midi_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)

        # Generate MIDI
        print("Starting MIDI generation...")
        detokenize_to_midi(
            score_note_tokens=score_notes,
            performance_predictions=perf_preds,
            output_midi_path=output_midi_path,
            chunk_size=chunk_size,
        )
        print("✅ MIDI generation completed successfully!")

    except Exception as e:
        print(f"❌ Error during MIDI conversion: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        cleanup_files = []

        # Clean up temporary score tokens JSON
        if not save_intermediate_json and temp_json_path and temp_json_path.startswith(tempfile.gettempdir()):
            cleanup_files.append(temp_json_path)

        # Clean up temporary performance JSON
        if not save_performance_json and perf_json_path and perf_json_path.startswith(tempfile.gettempdir()):
            cleanup_files.append(perf_json_path)

        for file_path in cleanup_files:
            try:
                os.unlink(file_path)
            except:
                pass

        if cleanup_files:
            print("🧹 Temporary files cleaned up")

    print(f"\n🎉 All done! Generated MIDI saved to: {output_midi_path}")
    return output_midi_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate MIDI directly from MusicXML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_midi_from_xml.py --xml_path song.xml --model_path model.pt --output_midi result.mid

  # With overlap and specific composer
  python generate_midi_from_xml.py --xml_path song.xml --model_path model.pt --output_midi result.mid \\
    --sequence_length 512 --overlap_length 256 --composer_id 3

  # Save all intermediate files
  python generate_midi_from_xml.py --xml_path song.xml --model_path model.pt --output_midi result.mid \\
    --save_intermediate --save_performance
        """
    )

    # Required arguments
    parser.add_argument('--xml_path', type=str, required=True,
                        help='Path to input MusicXML file (.xml, .mxl, .musicxml)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_midi', type=str, required=True,
                        help='Path to save generated MIDI file')

    # Optional model arguments
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model config file')
    parser.add_argument('--composer_id', type=int, default=0,
                        help='Composer ID for generation (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for inference (default: auto)')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling parameter (default: 0.9)')
    parser.add_argument('--sequence_length', type=int, default=512,
                        help='Length of each sequence segment (default: 512)')
    parser.add_argument('--overlap_length', type=int, default=0,
                        help='Length of overlap between segments (default: 0)')
    parser.add_argument('--no_tempo_prediction', action='store_true',
                        help='Disable tempo prediction')
    parser.add_argument('--no_sort', action='store_true',
                        help='Do not sort tokens by position')

    # Intermediate file options
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate score tokens JSON file')
    parser.add_argument('--intermediate_json_path', type=str, default=None,
                        help='Path to save intermediate score tokens JSON (implies --save_intermediate)')
    parser.add_argument('--save_performance', action='store_true',
                        help='Save performance predictions JSON file')
    parser.add_argument('--performance_json_path', type=str, default=None,
                        help='Path to save performance JSON (implies --save_performance)')

    args = parser.parse_args()

    # Validate input file
    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        print(f"❌ Error: XML file does not exist: {xml_path}")
        return 1

    if xml_path.suffix.lower() not in ['.xml', '.mxl', '.musicxml']:
        print(f"Warning: File extension '{xml_path.suffix}' may not be a valid MusicXML file")

    # Validate model file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model file does not exist: {model_path}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output_midi)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set save flags if paths are provided
    save_intermediate = args.save_intermediate or (args.intermediate_json_path is not None)
    save_performance = args.save_performance or (args.performance_json_path is not None)

    try:
        # Generate MIDI
        result_path = generate_midi_from_xml(
            xml_path=str(xml_path),
            output_midi_path=str(output_path),
            model_path=str(model_path),
            config_path=args.config_path,
            composer_id=args.composer_id,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
            sequence_length=args.sequence_length,
            overlap_length=args.overlap_length,
            use_predicted_tempo=not args.no_tempo_prediction,
            sort_by_position=not args.no_sort,
            save_intermediate_json=save_intermediate,
            intermediate_json_path=args.intermediate_json_path,
            save_performance_json=save_performance,
            performance_json_path=args.performance_json_path
        )

        print(f"\n Success! You can now:")
        print(f"   • Play the generated MIDI: {result_path}")
        if save_intermediate and args.intermediate_json_path:
            print(f"   • View the extracted score tokens: {args.intermediate_json_path}")
        if save_performance and args.performance_json_path:
            print(f"   • View the performance predictions: {args.performance_json_path}")
        print(f"   • Use any MIDI player or DAW to listen to the performance")

        return 0

    except KeyboardInterrupt:
        print("\n Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())


# python generate_from_xml.py --model_path ./check/sustain_classify_20250815_073720_best.pt --xml_path xml_data/12_Romances_Op.21__Sergei_Rachmaninoff_Zdes_khorosho_-_Arrangement_for_solo_piano.mxl --output_midi generate_results/12_Romances_Op.21__Sergei_Rachmaninoff_Zdes_khorosho_-_Arrangement_for_solo_piano.mid --composer_id 10 --config_path config.yaml --sequence_length 512 --overlap_length 256

# python generate_from_xml.py --model_path ./check/sustain_classify_20250815_073720_best.pt --xml_path xml_data/Prokofiev_Concerto_No.2_Op.16_Mvt._1.xml --output_midi Prokofiev_Concerto_No.2_Op.16_Mvt._1.mid --composer_id 9 --config_path config.yaml --sequence_length 512 --overlap_length 256 --save_performance --performance_json_path test.json

# python generate_from_xml.py --model_path ./check/sustain_classify_20250815_073720_best.pt --xml_path xml_data/With_Dog-teams.mxl --output_midi generate_results/With_Dog-teams.mid --composer_id 2 --config_path config.yaml --sequence_length 512 --overlap_length 256