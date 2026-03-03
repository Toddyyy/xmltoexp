import logging
from typing import override
from xml.etree.ElementTree import Element

from music21 import note
from music21.musicxml.xmlToM21 import MeasureParser

# A global map used by the patch
note_id_map: dict[int, str] = {}

logger = logging.getLogger(__name__)


class NoteIdPreservingParser(MeasureParser):

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use an external map to prevent garbage collection.
        self.note_id_map = note_id_map

    @override
    def xmlToSimpleNote(
        self, mxNote: Element, freeSpanners=True
    ) -> note.Note | note.Unpitched:
        """
        Wraps the parent class's `xmlToSimpleNote` method to capture and store the
        'id' attribute of the original MusicXML <note> element.

        This method first calls the parent parsing method process, then immediately
        extracts the xml note 'id' attribute from `mxNote` XML element and maps it
        to the newly created music21 object in `self.note_id_map`.

        Args:
            mxNote (Element): The original XML element object representing a <note> tag.
            freeSpanners (bool): A parameter controlling spanner object handling,
                                passed directly to the parent method.

        Returns:
            note.Note | note.Unpitched:
                A standard, fully-functional music21 Note or Unpitched object.
        """
        # Call the parent method to get a standard music21 note object.
        m21_note_object = super().xmlToSimpleNote(mxNote, freeSpanners=freeSpanners)

        # Capture the XML 'id'
        note_xml_id = mxNote.get("id")
        if note_xml_id:
            self.note_id_map[id(m21_note_object)] = note_xml_id
        else:
            logger.warning(
                f"A <note> element at offset {m21_note_object.offset} in "
                f"stream {getattr(self.stream, 'id', 'UnknownStream')} "
                "does not have an 'id' attribute."
            )
        return m21_note_object


class PatchedMusicXMLParser:
    """
    A context manager class to temporarily and safely replace music21's
    default MeasureParser with our enhanced NoteIdPreservingParser.
    Exposes note_id_map and chord_ids_map as attributes.
    """

    def __init__(self):
        self.note_id_map: dict[int, str]
        self._original_parser_class: type[MeasureParser]

    def get_note_id_map(self) -> dict[int, str]:
        """Returns a copy of the note_id_map. Otherwise the map would be cleared"""
        return note_id_map.copy()

    def __enter__(self):
        import music21

        logger.info("Patching MeasureParser to capture note IDs...")

        # Save the original class.
        self._original_parser_class = music21.musicxml.xmlToM21.MeasureParser

        # Monkey-patch the `MeasureParser` class
        self.note_id_map = note_id_map
        music21.musicxml.xmlToM21.MeasureParser = NoteIdPreservingParser

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import music21

        logger.info("Restoring original MeasureParser...")
        music21.musicxml.xmlToM21.MeasureParser = self._original_parser_class

        self.note_id_map.clear()

        if exc_type is not None:
            logger.warning(
                "An error occurred while using the patched MeasureParser: %s", exc_val
            )
