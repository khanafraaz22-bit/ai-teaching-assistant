"""
Unit tests — TranscriptCleaner

Tests that the cleaner:
  - Removes noise markers ([Music], [Applause], etc.)
  - Strips HTML entities
  - Removes filler words
  - Deduplicates boundary words
  - Capitalizes the first character
  - Handles empty input gracefully
"""

import pytest
from app.services.youtube.transcript_cleaner import TranscriptCleaner
from app.services.youtube.transcript_extractor import TranscriptSegment, RawTranscript


@pytest.fixture
def cleaner():
    return TranscriptCleaner()


def make_raw(segments_data: list[tuple[str, float, float]]) -> RawTranscript:
    """Helper: build a RawTranscript from (text, start, duration) tuples."""
    segments = [
        TranscriptSegment(text=t, start=s, duration=d)
        for t, s, d in segments_data
    ]
    return RawTranscript(
        video_id="vid_test",
        segments=segments,
        language="en",
        source="youtube_auto",
    )


class TestNoisRemoval:

    def test_removes_music_marker(self, cleaner):
        raw = make_raw([("[Music] Hello world", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert "[Music]" not in result.cleaned_text
        assert "Hello world" in result.cleaned_text

    def test_removes_applause_marker(self, cleaner):
        raw = make_raw([("Great talk [Applause] thank you", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert "[Applause]" not in result.cleaned_text

    def test_removes_inaudible_marker(self, cleaner):
        raw = make_raw([("So the answer is [inaudible] and that is why", 0.0, 4.0)])
        result = cleaner.clean(raw)
        assert "[inaudible]" not in result.cleaned_text

    def test_removes_parenthetical_noise(self, cleaner):
        raw = make_raw([("Welcome (laughter) to the course", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert "(laughter)" not in result.cleaned_text

    def test_removes_html_entities(self, cleaner):
        raw = make_raw([("Tom &amp; Jerry &lt;3", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert "&amp;" not in result.cleaned_text
        assert "&lt;" not in result.cleaned_text

    def test_removes_filler_words(self, cleaner):
        raw = make_raw([("Um so uh we need to er understand this", 0.0, 4.0)])
        result = cleaner.clean(raw)
        text = result.cleaned_text.lower()
        assert " um " not in text
        assert " uh " not in text
        assert " er " not in text

    def test_removes_music_notes(self, cleaner):
        raw = make_raw([("♪ some music ♪ and then speech", 0.0, 4.0)])
        result = cleaner.clean(raw)
        assert "♪" not in result.cleaned_text


class TestTextMerging:

    def test_merges_multiple_segments(self, cleaner, sample_raw_transcript):
        result = cleaner.clean(sample_raw_transcript)
        assert len(result.cleaned_text) > 100
        assert result.word_count > 20

    def test_capitalizes_first_character(self, cleaner):
        raw = make_raw([("welcome to the course", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert result.cleaned_text[0].isupper()

    def test_normalizes_whitespace(self, cleaner):
        raw = make_raw([("too   many    spaces  here", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert "  " not in result.cleaned_text

    def test_word_count_is_accurate(self, cleaner):
        raw = make_raw([("one two three four five", 0.0, 3.0)])
        result = cleaner.clean(raw)
        assert result.word_count == 5


class TestEdgeCases:

    def test_empty_transcript(self, cleaner):
        raw = RawTranscript(
            video_id="empty",
            segments=[],
            language="en",
            source="youtube_auto",
        )
        result = cleaner.clean(raw)
        assert result.cleaned_text == ""
        assert result.word_count == 0

    def test_all_noise_segments(self, cleaner):
        raw = make_raw([
            ("[Music]", 0.0, 2.0),
            ("[Applause]", 2.0, 2.0),
            ("[Music]", 4.0, 2.0),
        ])
        result = cleaner.clean(raw)
        assert result.word_count == 0

    def test_preserves_segments_for_timestamps(self, cleaner, sample_raw_transcript):
        result = cleaner.clean(sample_raw_transcript)
        # Cleaned segments should preserve timing
        assert len(result.segments) > 0
        for seg in result.segments:
            assert seg.start >= 0
            assert seg.duration >= 0

    def test_source_preserved(self, cleaner, sample_raw_transcript):
        result = cleaner.clean(sample_raw_transcript)
        assert result.source == sample_raw_transcript.source
        assert result.language == sample_raw_transcript.language
