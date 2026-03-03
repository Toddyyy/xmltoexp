type Triple = tuple[str, str, str]  # (score_path, midi_path, match_path)

import logging
import os

logger = logging.getLogger("DatasetCollectors")


def asap_collector(
    dataset_directory: str,
) -> list[Triple]:
    """Collects all triples from the ASAP dataset"""
    triples: list[Triple] = []
    for root, _, files in os.walk(dataset_directory):
        for file in files:
            if not file.endswith(".match"):
                continue

            base_name = file.rstrip(".match")
            midi_file = os.path.join(root, f"{base_name}.mid")
            xml_file = os.path.join(root, "xml_score.musicxml")
            if not all(
                os.path.exists(f)
                for f in [xml_file, midi_file, os.path.join(root, file)]
            ):
                logger.warning(
                    f"Some of the following three files are missing for {base_name}: "
                    f"XML: {xml_file}, MIDI: {midi_file}, Match: {os.path.join(root, file)}"
                )
                continue

            triples.append(
                (
                    os.path.join(root, "xml_score.musicxml").replace("\\", "/"),
                    os.path.join(root, f"{base_name}.mid").replace("\\", "/"),
                    os.path.join(root, file).replace("\\", "/"),
                )
            )

    logger.info(f"Collected {len(triples)} triples from ASAP dataset.")
    return triples


def batik_collector(
    dataset_directory: str,
) -> list[Triple]:
    """
    Collect (score, midi, match) triples from the Batik Plays Mozart dataset.
    """
    xml_dir = os.path.join(dataset_directory, "scores")
    midi_dir = os.path.join(dataset_directory, "midi")
    match_dir = os.path.join(dataset_directory, "match")

    # Build a set of available base names from the MusicXML folder
    base_names = {
        os.path.splitext(f)[0]
        for f in os.listdir(xml_dir)
        if f.endswith(".musicxml") or f.endswith(".xml")
    }

    triples: list[Triple] = []
    for base in base_names:
        score_path = os.path.join(xml_dir, base + ".musicxml").replace("\\", "/")
        midi_path = os.path.join(midi_dir, base + ".mid").replace("\\", "/")
        match_path = os.path.join(match_dir, base + ".match").replace("\\", "/")

        if os.path.exists(midi_path) and os.path.exists(match_path):
            triples.append((score_path, midi_path, match_path))
        else:
            logger.warning(f"Missing MIDI or match file for base: {base}")

    logger.info(f"Collected {len(triples)} triples from Batik dataset.")
    return triples


def four_22_collector(
    dataset_directory: str,
) -> list[Triple]:
    """
    Collect (score, midi, match) triples from the Vienna4x22 dataset.
    """
    xml_dir = os.path.join(dataset_directory, "musicxml")
    midi_dir = os.path.join(dataset_directory, "midi")
    match_dir = os.path.join(dataset_directory, "match")

    # Load all MusicXML files and create a mapping from prefix to full path
    xml_files = {
        os.path.splitext(f)[0]: os.path.join(xml_dir, f).replace("\\", "/")
        for f in os.listdir(xml_dir)
        if f.endswith(".musicxml") or f.endswith(".xml")
    }

    triples: list[Triple] = []
    for filename in os.listdir(match_dir):
        if not filename.endswith(".match"):
            continue

        base_name = filename[:-6]  # Remove ".match"
        match_path = os.path.join(match_dir, filename).replace("\\", "/")
        midi_path = os.path.join(midi_dir, base_name + ".mid").replace("\\", "/")

        # Extract XML file prefix (e.g., "Chopin_op10_no3" from "Chopin_op10_no3_p01")
        xml_prefix = base_name.split("_p")[0]

        if xml_prefix in xml_files and os.path.exists(midi_path):
            score_path = xml_files[xml_prefix]
            triples.append((score_path, midi_path, match_path))
        else:
            logger.warning(f"Missing MIDI or MusicXML for base: {base_name}")

    logger.info(f"Collected {len(triples)} triples from 4x22 dataset.")
    return triples
