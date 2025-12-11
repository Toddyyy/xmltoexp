from tokenizer.paired_tokenizer import PairedTokenizer


def test_alignment_loading():
    """Test the loading of alignment data."""

    paired_tokenizer = PairedTokenizer(
        xml_score_path="./datasets/asap_dataset/Bach/Fugue/bwv_846/xml_score.musicxml",
        midi_path="./datasets/asap_dataset/Bach/Fugue/bwv_846/Shi05M.mid",
        match_path="./datasets/asap_dataset/Bach/Fugue/bwv_846/Shi05M.match",
    )

    (
        _score_metadata,
        _midi_metadata,
        score_tokens,
        midi_tokens,
        _expressions,
        _controls,
    ) = paired_tokenizer.tokenize()

    print(
        f"Score Metadata, Score tokens: {len(score_tokens)}, MIDI tokens: {len(midi_tokens)}"
    )

    assert len(score_tokens) == len(
        midi_tokens
    ), "Score and MIDI tokens should match in length"

    assert (
        len(score_tokens) > 0
    ), "There should be at least one score token after alignment"
