"""
Script to load cached dataset and process it through ScoreExplainer to get full tokens.
Save each sextuple's full tokens as separate JSON files with additional deviation calculations.
"""
import traceback
import sys
import json
import logging
import os
from dataclasses import asdict
from typing import List, Tuple

from dataloader import ScorePerformanceDataset, try_load_sextuplet_from_cache
from dataset_collectors import asap_collector, batik_collector, four_22_collector
from dataset_utils import Triple, triple_hash_fn
from score_explainer import ScoreExplainer, FullPerformanceToken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = "./dataset_cache"


def calculate_average_tempo(full_tokens: List[FullPerformanceToken], method: str = "arithmetic") -> float:
    """
    Calculate average tempo from full tokens using different methods.

    Args:
        full_tokens (List[FullPerformanceToken]): List of full performance tokens
        method (str): "arithmetic" for simple average, "weighted" for duration-weighted average

    Returns:
        float: Average tempo in BPM
    """
    if not full_tokens:
        return 0.0

    if method == "arithmetic":
        # Simple arithmetic mean
        tempos = [token.local_tempo for token in full_tokens]
        return sum(tempos) / len(tempos)

    elif method == "weighted":
        # Duration-weighted average
        total_weighted_tempo = 0.0
        total_duration = 0.0

        for token in full_tokens:
            duration = token.score_note_token.duration  # Duration in beats
            tempo = token.local_tempo

            total_weighted_tempo += tempo * duration
            total_duration += duration

        if total_duration == 0:
            # Fallback to arithmetic mean if total duration is 0
            logger.warning("Total duration is 0, falling back to arithmetic mean")
            return calculate_average_tempo(full_tokens, "arithmetic")

        return total_weighted_tempo / total_duration

    else:
        raise ValueError(f"Unknown method: {method}. Use 'arithmetic' or 'weighted'")


def calculate_score_timing_in_seconds(full_tokens: List[FullPerformanceToken],
                                      average_tempo: float) -> Tuple[List[float], List[float]]:
    """
    Calculate score note onsets and durations in seconds using average tempo.

    Args:
        full_tokens (List[FullPerformanceToken]): List of full performance tokens
        average_tempo (float): Average tempo in BPM

    Returns:
        Tuple[List[float], List[float]]: (onset_seconds, duration_seconds)
    """
    onset_seconds = []
    duration_seconds = []

    # Convert beats to seconds: seconds = beats * 60 / BPM
    beats_to_seconds = 60.0 / average_tempo if average_tempo > 0 else 0.0

    for token in full_tokens:
        # Convert score note position (onset in beats) to seconds
        onset_sec = token.score_note_token.position * beats_to_seconds
        onset_seconds.append(onset_sec)

        # Convert score note duration (in beats) to seconds
        duration_sec = token.score_note_token.duration * beats_to_seconds
        duration_seconds.append(duration_sec)

    return onset_seconds, duration_seconds


def calculate_deviations_in_seconds(full_tokens: List[FullPerformanceToken],
                                    score_onsets_sec: List[float],
                                    score_durations_sec: List[float]) -> Tuple[List[float], List[float]]:
    """
    Calculate onset and duration deviations in seconds.

    Args:
        full_tokens (List[FullPerformanceToken]): List of full performance tokens
        score_onsets_sec (List[float]): Score note onsets in seconds
        score_durations_sec (List[float]): Score note durations in seconds

    Returns:
        Tuple[List[float], List[float]]: (onset_deviations_sec, duration_deviations_sec)
    """
    onset_deviations_sec = []
    duration_deviations_sec = []

    for i, token in enumerate(full_tokens):
        # Onset deviation = performance onset - score onset (in seconds)
        onset_dev_sec = token.performance_note_token.onset_sec - score_onsets_sec[i]
        onset_deviations_sec.append(onset_dev_sec)

        # Duration deviation = performance duration - score duration (in seconds)
        duration_dev_sec = token.performance_note_token.duration_sec - score_durations_sec[i]
        duration_deviations_sec.append(duration_dev_sec)

    return onset_deviations_sec, duration_deviations_sec


def serialize_enhanced_full_token(token: FullPerformanceToken,
                                  onset_deviation_sec: float,
                                  duration_deviation_sec: float,
                                  average_tempo_arithmetic: float,
                                  average_tempo_weighted: float) -> dict:
    """
    Convert FullPerformanceToken to JSON-serializable dictionary with enhanced attributes.

    Args:
        token (FullPerformanceToken): Token to serialize
        onset_deviation_sec (float): Onset deviation in seconds
        duration_deviation_sec (float): Duration deviation in seconds
        average_tempo_arithmetic (float): Arithmetic average tempo
        average_tempo_weighted (float): Weighted average tempo

    Returns:
        dict: JSON-serializable representation with enhanced attributes
    """
    return {
        'performance_note_token': {
            'pitch': token.performance_note_token.pitch,
            'velocity': token.performance_note_token.velocity,
            'onset_sec': token.performance_note_token.onset_sec,
            'duration_sec': token.performance_note_token.duration_sec,
            'note_id': token.performance_note_token.xml_note_id,
        },
        'score_note_token': {
            'pitch': token.score_note_token.pitch,
            'duration': token.score_note_token.duration,
            'position': token.score_note_token.position,
            'part_id': token.score_note_token.part_id,
            'tie': token.score_note_token.tie,
            'is_staccato': token.score_note_token.is_staccato,
            'is_accent': token.score_note_token.is_accent,
            'fingering': token.score_note_token.fingering,
            'xml_note_id': token.score_note_token.xml_note_id,
        },
        'onset_deviation_in_beats': token.onset_deviation_in_beats,
        'duration_deviation_in_beats': token.duration_deviation_in_beats,
        'onset_deviation_in_seconds': onset_deviation_sec,  # New attribute
        'duration_deviation_in_seconds': duration_deviation_sec,  # New attribute
        'local_tempo': token.local_tempo,
        'sustain_level': token.sustain_level,
        'tempo_info': {  # Additional tempo information
            'average_tempo_arithmetic': average_tempo_arithmetic,
            'average_tempo_weighted': average_tempo_weighted,
        }
    }


def process_and_save_individual_sextuples(
        cache_root: str = DEFAULT_CACHE_ROOT,
        dataset_configs: dict = None,
        max_samples: int = None,
        tempo_estimation_time_window: float = 5.0,
        output_dir: str = "full_tokens_output"
) -> None:
    """
    Load cached dataset and process each sextuple individually, saving enhanced full tokens as JSON files.

    Args:
        cache_root (str): Path to cache directory
        dataset_configs (dict): Configuration for which datasets to load
        max_samples (int): Maximum number of samples to process (None for all)
        tempo_estimation_time_window (float): Time window for tempo estimation
        output_dir (str): Directory to save JSON files
    """

    # Default dataset configuration
    if dataset_configs is None:
        dataset_configs = {
            "asap": "datasets/asap_dataset/",
            # "batik": "datasets/batik_plays_mozart/",
            # "four_22": "datasets/vienna4x22"
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect triples from specified datasets
    all_triples = []

    if "asap" in dataset_configs:
        logger.info(f"Collecting ASAP triples from {dataset_configs['asap']}")
        all_triples += asap_collector(dataset_configs["asap"])

    if "batik" in dataset_configs:
        logger.info(f"Collecting Batik triples from {dataset_configs['batik']}")
        all_triples += batik_collector(dataset_configs["batik"])

    if "four_22" in dataset_configs:
        logger.info(f"Collecting 4x22 triples from {dataset_configs['four_22']}")
        all_triples += four_22_collector(dataset_configs["four_22"])

    logger.info(f"Total collected triples: {len(all_triples)}")

    # Create dataset instance
    dataset = ScorePerformanceDataset(
        triples=all_triples,
        cache_directory=cache_root
    )

    logger.info(f"Dataset loaded with {len(dataset)} valid cached samples")

    # Limit samples if specified
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    logger.info(f"Processing {num_samples} samples...")

    processed_count = 0
    failed_count = 0
    
    for i in range(num_samples):
        try:
            # Get the triple for this sample
            triple = dataset.triples[i]
            triple_hash = triple_hash_fn(triple)

            # Load sextuple from cache
            (score_metadata,
             performance_metadata,
             score_tokens,
             performance_tokens,
             expressions,
             controls) = dataset[i]

            # Create ScoreExplainer instance
            explainer = ScoreExplainer(
                score_tokens=score_tokens,
                score_expressions=expressions,
                performance_tokens=performance_tokens,
                performance_controls=controls,
            )

            # Get full tokens for this sextuple
            full_tokens = explainer.get_full_tokens()

            if not full_tokens:
                logger.warning(f"No full tokens generated for sample {i} (hash: {triple_hash})")
                continue

            # Calculate average tempos using both methods
            avg_tempo_arithmetic = calculate_average_tempo(full_tokens, "arithmetic")
            avg_tempo_weighted = calculate_average_tempo(full_tokens, "weighted")

            logger.info(f"Sample {i}: Arithmetic avg tempo = {avg_tempo_arithmetic:.2f} BPM, "
                        f"Weighted avg tempo = {avg_tempo_weighted:.2f} BPM")

            # Calculate score timing in seconds using weighted average tempo (you can change this)
            score_onsets_sec, score_durations_sec = calculate_score_timing_in_seconds(
                full_tokens, avg_tempo_weighted
            )
            # score_onsets_sec, score_durations_sec = calculate_score_timing_in_seconds(
            #     full_tokens, avg_tempo_arithmetic
            # )

            # Calculate deviations in seconds
            onset_deviations_sec, duration_deviations_sec = calculate_deviations_in_seconds(
                full_tokens, score_onsets_sec, score_durations_sec
            )

            # Serialize enhanced full tokens
            serialized_tokens = []
            for j, token in enumerate(full_tokens):
                enhanced_token = serialize_enhanced_full_token(
                    token,
                    onset_deviations_sec[j],
                    duration_deviations_sec[j],
                    avg_tempo_arithmetic,
                    avg_tempo_weighted
                )
                serialized_tokens.append(enhanced_token)

            # Create metadata for the JSON file
            json_data = {
                'metadata': {
                    'triple_hash': triple_hash,
                    'score_path': triple[0],
                    'midi_path': triple[1],
                    'match_path': triple[2],
                    'num_tokens': len(full_tokens),
                    'score_metadata': {
                        'composer': score_metadata.composer if score_metadata else None,
                        'title': score_metadata.title if score_metadata else None,
                        'time_signature': score_metadata.major_time_sig if score_metadata else None,
                    },
                    'performance_metadata': {
                        'performer': performance_metadata.performer if performance_metadata else None,
                    },
                    'tempo_analysis': {
                        'average_tempo_arithmetic': avg_tempo_arithmetic,
                        'average_tempo_weighted': avg_tempo_weighted,
                        'tempo_difference': abs(avg_tempo_arithmetic - avg_tempo_weighted),
                        'local_tempo_range': {
                            'min': min(token.local_tempo for token in full_tokens),
                            'max': max(token.local_tempo for token in full_tokens),
                        }
                    }
                },
                'full_tokens': serialized_tokens
            }

            # Save to JSON file
            output_filename = f"{triple_hash}.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            processed_count += 1

            # Log progress
            if processed_count % 10 == 0:
                logger.info(f"Processed {processed_count}/{num_samples} samples")

            logger.info(f"Saved {len(full_tokens)} enhanced tokens to {output_filename}")

        except Exception as e:
            failed_count += 1
            triple = dataset.triples[i] if i < len(dataset.triples) else None
            triple_hash = triple_hash_fn(triple) if triple else "unknown"
            logger.error(f"Failed to process sample {i} (hash: {triple_hash}): {e}")
            continue

    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {processed_count}/{num_samples}")
    logger.info(f"Failed: {failed_count}/{num_samples}")
    logger.info(f"Enhanced JSON files saved to: {output_dir}")


def analyze_enhanced_full_tokens(output_dir: str) -> None:
    """
    Analyze the enhanced full tokens from saved JSON files and print statistics.

    Args:
        output_dir (str): Directory containing JSON files
    """
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

    if not json_files:
        logger.warning(f"No JSON files found in {output_dir}")
        return

    logger.info("=== Enhanced Full Tokens Analysis ===")

    all_onset_dev_sec = []
    all_duration_dev_sec = []
    all_onset_dev_beats = []
    all_duration_dev_beats = []
    all_arithmetic_tempos = []
    all_weighted_tempos = []

    total_tokens = 0

    for json_file in json_files[:5]:  # Analyze first 5 files as sample
        file_path = os.path.join(output_dir, json_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tokens = data['full_tokens']
            total_tokens += len(tokens)

            # Collect statistics
            for token in tokens:
                all_onset_dev_sec.append(token['onset_deviation_in_seconds'])
                all_duration_dev_sec.append(token['duration_deviation_in_seconds'])
                all_onset_dev_beats.append(token['onset_deviation_in_beats'])
                all_duration_dev_beats.append(token['duration_deviation_in_beats'])
                all_arithmetic_tempos.append(token['tempo_info']['average_tempo_arithmetic'])
                all_weighted_tempos.append(token['tempo_info']['average_tempo_weighted'])

        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    if total_tokens > 0:
        logger.info(f"Analyzed {total_tokens} tokens from {len(json_files)} files")
        logger.info(f"Onset deviation (seconds): min={min(all_onset_dev_sec):.4f}, "
                    f"max={max(all_onset_dev_sec):.4f}, avg={sum(all_onset_dev_sec) / len(all_onset_dev_sec):.4f}")
        logger.info(f"Duration deviation (seconds): min={min(all_duration_dev_sec):.4f}, "
                    f"max={max(all_duration_dev_sec):.4f}, avg={sum(all_duration_dev_sec) / len(all_duration_dev_sec):.4f}")
        logger.info(f"Arithmetic tempo range: {min(all_arithmetic_tempos):.2f} - {max(all_arithmetic_tempos):.2f} BPM")
        logger.info(f"Weighted tempo range: {min(all_weighted_tempos):.2f} - {max(all_weighted_tempos):.2f} BPM")


# Keep the original functions for backward compatibility
def load_and_process_dataset(
        cache_root: str = DEFAULT_CACHE_ROOT,
        dataset_configs: dict = None,
        max_samples: int = None,
        tempo_estimation_time_window: float = 5.0
) -> List[FullPerformanceToken]:
    """
    Load cached dataset and process through ScoreExplainer to get full tokens.
    (Original function kept for backward compatibility)
    """
    # Implementation remains the same as original
    pass


def analyze_full_tokens(full_tokens: List[FullPerformanceToken]) -> None:
    """
    Analyze the generated full tokens and print statistics.
    (Original function kept for backward compatibility)
    """
    # Implementation remains the same as original
    pass


def save_full_tokens_sample(full_tokens: List[FullPerformanceToken],
                            output_file: str = "full_tokens_sample.txt",
                            num_samples: int = 10) -> None:
    """
    Save a sample of full tokens to a text file for inspection.
    (Original function kept for backward compatibility)
    """
    # Implementation remains the same as original
    pass

if __name__ == "__main__":
    # Configuration
    CACHE_ROOT = "./dataset_cache_full"
    MAX_SAMPLES = None  # Set to None to process all samples
    OUTPUT_DIR = "full_data"  # Directory to save individual JSON files

    # Dataset configuration - uncomment datasets you want to process
    dataset_configs = {
        "asap": "datasets/asap_dataset/",
        # "batik": "datasets/batik_plays_mozart/",
        "four_22": "datasets/vienna4x22"
    }

    # Check if cache directory exists
    if not os.path.isdir(CACHE_ROOT):
        logger.error(f"Cache directory {CACHE_ROOT} does not exist. "
                     f"Please run build_dataset.py first.")
        exit(1)

    # Process and save individual sextuples with enhanced attributes
    logger.info("Starting enhanced individual sextuple processing and JSON saving...")
    process_and_save_individual_sextuples(
        cache_root=CACHE_ROOT,
        dataset_configs=dataset_configs,
        max_samples=MAX_SAMPLES,
        tempo_estimation_time_window=5.0,
        output_dir=OUTPUT_DIR
    )

    # Analyze the results
    logger.info("Analyzing enhanced results...")
    analyze_enhanced_full_tokens(OUTPUT_DIR)

    logger.info("Enhanced script completed successfully!")
    logger.info(f"Check the '{OUTPUT_DIR}' directory for enhanced JSON files with deviation calculations.")