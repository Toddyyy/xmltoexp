from tokenizer.matched_musicxml_tokenizer import MatchedMusicXMLScoreTokenizer


def test_score_token_id():
    """
    Test the parsing of score token id from <note> object
    """
    xml_score_path = "./datasets/asap_dataset/Bach/Fugue/bwv_846/xml_score.musicxml"
    tokenizer = MatchedMusicXMLScoreTokenizer(xml_score_path)

    notes = tokenizer.tokenize_notes()
    for note in notes:
        assert note.xml_note_id is not None, "Note ID should not be None"
