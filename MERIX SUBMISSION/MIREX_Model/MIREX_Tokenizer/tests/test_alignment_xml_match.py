import re

from tokenizer.matched_musicxml_tokenizer import MatchedMusicXMLScoreTokenizer


def extract_snote_ids_from_match(match_file_path: str) -> list[str]:
    """Extract all snote IDs from a .match file, e.g., snote(n2-1,[C,n],...) -> n2-1"""
    snote_ids = []
    pattern = re.compile(r"^snote\(([^,]+),")  # Match line starting with snote(n2-1,
    with open(match_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                snote_ids.append(match.group(1))
    return snote_ids


def get_tokenizer_note_ids(xml_path: str) -> list[str]:
    """Use MatchedMusicXMLScoreTokenizer to get xml_note_id"""
    tokenizer = MatchedMusicXMLScoreTokenizer(xml_path)
    tokens = tokenizer.tokenize_notes()
    return [token.xml_note_id for token in tokens if token.xml_note_id]


def compare_ids(tokenizer_ids: list[str], match_ids: list[str]) -> None:
    """Compare two sets of IDs, output the number of matches and differences"""
    # Add -1 suffix to tokenizer ids
    tokenizer_ids_with_suffix = [tid + "-1" for tid in tokenizer_ids]

    set_tokenizer = set(tokenizer_ids_with_suffix)
    set_match = set(match_ids)

    intersection = set_tokenizer & set_match
    only_in_tokenizer = set_tokenizer - set_match
    only_in_match = set_match - set_tokenizer

    print("✅ Number of matches:", len(intersection))
    print("➕ Number only in tokenizer but not in match:", len(only_in_tokenizer))
    print("➖ Number only in match but not in tokenizer:", len(only_in_match))


if __name__ == "__main__":
    match_file_path = "./example_data/nasap/bwv_846/Shi05M.match"
    xml_file_path = "./example_data/nasap/bwv_846/xml_score.musicxml"

    match_ids = extract_snote_ids_from_match(match_file_path)
    tokenizer_ids = get_tokenizer_note_ids(xml_file_path)

    compare_ids(tokenizer_ids, match_ids)
