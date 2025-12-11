from tokenizer.score_tokenizer import MusicXMLTokenizer
from rich import print

# # Create tokenizer.score_tokenizer instance
# tokenizer = MusicXMLTokenizer(
#     "competition_data/With_Dog-teams.mxl"
# )
#
# # # # Extract and print metadata
# # metadata = tokenizer.parse_metadata()
# # print("=== MUSIC SCORE METADATA ===")
# # print(f"Composer: {metadata.composer}")
# # print(f"Performer: {metadata.performer}")
# # print(f"Genre: {metadata.genre}")
# # print(f"Time Signature: {metadata.major_time_sig}")
# # print(f"Title: {metadata.title or 'Untitled'}")
# # print(f"Year: {metadata.year or 'Unknown'}")
# #
# # # Extract and print first 5 notes
# notes = tokenizer.tokenize_notes()
# for note_token in notes[:200]:
#     if note_token.tie:
#         print(f"Tie Note: {note_token}")

# Extract and print expressions
# expressions = tokenizer.parse_expressions()
# for expr in expressions[:200]:
#     if expr.type == "tie":
#         print(f"Tie Expression: {expr}")
import os
import json
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Tuple
from tokenizer.score_tokenizer import MusicXMLTokenizer


def convert_score_note_token_to_dict(note_token):
    """Convert ScoreNoteToken to dictionary format expected by generate.py"""
    return {
        'pitch': note_token.pitch,
        'duration': note_token.duration,
        'position': note_token.position,
        'part_id': note_token.part_id,
        'is_staccato': note_token.is_staccato,
        'is_accent': note_token.is_accent
    }


def extract_score_tokens_from_xml(xml_path: str) -> List[Dict]:
    """
    Extract score tokens from MusicXML file using your existing tokenizer.

    Args:
        xml_path: Path to MusicXML file

    Returns:
        List of score token dictionaries in the format expected by generate.py
    """
    print(f"Processing: {xml_path}")

    try:
        # Create tokenizer instance
        tokenizer = MusicXMLTokenizer(xml_path)

        # Extract notes using your tokenizer
        notes = tokenizer.tokenize_notes()
        print(f"Extracted {len(notes)} notes from {xml_path}")

        # Convert to the format expected by generate.py
        full_tokens = []

        for note_token in notes:
            # Create the structure expected by generate.py
            token_dict = {
                'score_note_token': convert_score_note_token_to_dict(note_token)
            }
            full_tokens.append(token_dict)

        return full_tokens

    except Exception as e:
        print(f"Error processing {xml_path}: {str(e)}")
        raise


def sort_score_tokens_by_position(score_tokens: List[Dict]) -> List[Dict]:
    """
    Sort score tokens by position (from smallest to largest).

    Args:
        score_tokens: List of score token dictionaries

    Returns:
        Sorted list of score tokens
    """
    print(f"Sorting {len(score_tokens)} score tokens by position...")

    # Sort by position field in score_note_token
    sorted_tokens = sorted(score_tokens, key=lambda x: x['score_note_token']['position'])

    print(f"✅ Sorted tokens by position")
    return sorted_tokens


def save_score_tokens_to_json(score_tokens: List[Dict], output_path: str, xml_path: str, is_sorted: bool = False):
    """Save score tokens to JSON file with metadata."""

    # Create output data structure matching your existing format
    output_data = {
        'metadata': {
            'source_xml': xml_path,
            'total_notes': len(score_tokens),
            'extraction_method': 'MusicXMLTokenizer',
            'sorted_by_position': is_sorted
        },
        'full_tokens': score_tokens
    }

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    sorted_msg = " (sorted by position)" if is_sorted else ""
    print(f"Saved {len(score_tokens)} score tokens{sorted_msg} to {output_path}")


def process_single_file(xml_path: str, output_path: str = None, sort_by_position: bool = True):
    """Process a single XML file and save as JSON."""

    if output_path is None:
        # Auto-generate output path
        xml_file = Path(xml_path)
        suffix = "_sorted_score_tokens" if sort_by_position else "_score_tokens"
        output_path = xml_file.parent / f"{xml_file.stem}{suffix}.json"

    # Extract score tokens
    score_tokens = extract_score_tokens_from_xml(xml_path)

    # Sort by position if requested
    if sort_by_position:
        score_tokens = sort_score_tokens_by_position(score_tokens)

    # Save to JSON
    save_score_tokens_to_json(score_tokens, output_path, xml_path, is_sorted=sort_by_position)

    return output_path


def process_directory(input_dir: str, output_dir: str = None, sort_by_position: bool = True):
    """Process all XML files in a directory."""

    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = input_path / "score_tokens_output"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all XML files
    xml_extensions = ['.xml', '.mxl', '.musicxml']
    xml_files = []

    for ext in xml_extensions:
        xml_files.extend(input_path.glob(f"**/*{ext}"))

    print(f"Found {len(xml_files)} XML files in {input_dir}")

    processed_files = []

    for xml_file in xml_files:
        try:
            # Generate output filename
            suffix = "_sorted_score_tokens" if sort_by_position else "_score_tokens"
            output_filename = f"{xml_file.stem}{suffix}.json"
            output_file = output_path / output_filename

            print(f"\nProcessing: {xml_file.name}")

            # Process the file
            score_tokens = extract_score_tokens_from_xml(str(xml_file))

            # Sort by position if requested
            if sort_by_position:
                score_tokens = sort_score_tokens_by_position(score_tokens)

            save_score_tokens_to_json(score_tokens, str(output_file), str(xml_file), is_sorted=sort_by_position)

            processed_files.append(str(output_file))

        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
            continue

    print(f"\n✅ Successfully processed {len(processed_files)} files")
    print(f"📁 Output directory: {output_path}")

    return processed_files


def main():
    parser = argparse.ArgumentParser(description='Convert MusicXML files to score tokens JSON for generation')
    parser.add_argument('--input_path', default='option_list',
                        help='Path to XML file or directory containing XML files')
    parser.add_argument('--output', '-o', default='option_list',
                        help='Output path (file for single XML, directory for batch)')
    parser.add_argument('--batch', action='store_true', help='Process directory of XML files')
    parser.add_argument('--no_sort', action='store_true',
                        help='Do not sort tokens by position (default: sort by position)')

    args = parser.parse_args()

    input_path = Path(args.input_path)
    sort_by_position = not args.no_sort

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    try:
        if args.batch or input_path.is_dir():
            # Process directory
            print(f"Processing directory: {input_path}")
            if sort_by_position:
                print("📊 Sorting tokens by position (use --no_sort to disable)")

            processed_files = process_directory(str(input_path), args.output, sort_by_position)

            print(f"\n🎉 Batch processing complete!")
            print(f"📊 Processed {len(processed_files)} files")

        else:
            # Process single file
            print(f"Processing single file: {input_path}")
            if sort_by_position:
                print("📊 Sorting tokens by position (use --no_sort to disable)")

            output_file = process_single_file(str(input_path), args.output, sort_by_position)

            print(f"\n🎉 Processing complete!")
            print(f"📁 Output file: {output_file}")

        print(f"\n💡 Usage tip:")
        print(f"You can now use the generated JSON file(s) with generate.py:")
        if args.batch or input_path.is_dir():
            print(f"python generate.py --input_json path/to/generated/file_score_tokens.json --output_json output.json")
        else:
            suffix = "_sorted_score_tokens" if sort_by_position else "_score_tokens"
            output_file = args.output or f"{input_path.stem}{suffix}.json"
            print(f"python generate.py --input_json {output_file} --output_json output.json")

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
#
#
# # Example usage functions for testing
# def test_single_file():
#     """Test function for processing a single file."""
#     xml_path = "competition_data/With_Dog-teams.mxl"
#     output_path = "test_score_tokens.json"
#
#     try:
#         process_single_file(xml_path, output_path)
#         print("✅ Test completed successfully!")
#
#         # Verify the output format
#         with open(output_path, 'r') as f:
#             data = json.load(f)
#
#         print(f"📊 Verification:")
#         print(f"  Total tokens: {len(data['full_tokens'])}")
#         print(f"  First token structure: {list(data['full_tokens'][0].keys())}")
#         print(f"  Score note token fields: {list(data['full_tokens'][0]['score_note_token'].keys())}")
#
#     except Exception as e:
#         print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    # Uncomment to run test
    # test_single_file()
    main()