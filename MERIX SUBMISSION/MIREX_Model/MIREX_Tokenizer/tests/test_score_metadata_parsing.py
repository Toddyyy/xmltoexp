def test_xml_score_title_parsing():
    from tokenizer.musicxml_utils import parse_title_from_xml

    # xml = "./datasets/asap_dataset/Bach/Fugue/bwv_846/xml_score.musicxml"  # ASAP
    xml = "./datasets/batik_plays_mozart/scores/kv284_2.musicxml"  # Batik
    title = parse_title_from_xml(xml)
    print(f"Parsed title: {title}")
    assert title is not None, "Title should not be None"


# def test_music21_score_metadata_parsing():
#     from music21 import converter, metadata
#
#     # Load the score
#     score = converter.parse(
#         "./datasets/asap_dataset/Bach/Fugue/bwv_846/xml_score.musicxml"
#     )
#
#     # Check if metadata is present
#     assert score.metadata is not None, "Metadata should not be None"
#     assert isinstance(
#         score.metadata, metadata.Metadata
#     ), "Metadata should be of type music21.metadata.Metadata"
#     print(score.metadata.all())
#
#     # Check specific metadata fields
#     assert isinstance(score.metadata.title, str), "Title should be a string"
#     assert isinstance(score.metadata.composer, str), "Composer should be a string"
#     assert isinstance(score.metadata.date, str), "Date should be a string"
#
#     # Check if the metadata matches expected values
#     assert (
#         score.metadata.title == "Expected Title"
#     ), "Title does not match expected value"
#     assert (
#         score.metadata.composer == "Expected Composer"
#     ), "Composer does not match expected value"
#     assert score.metadata.date == "Expected Date", "Date does not match expected value"
