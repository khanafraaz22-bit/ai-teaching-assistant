"""
Unit tests — TranscriptCleaner

Tests noise removal, filler word stripping, segment merging,
and edge cases like empty transcripts.
"""

import pytest
from app.services.youtube.transcript_cleaner import TranscriptCleaner
from app.services.youtube.transcript_extractor import TranscriptSegment, RawTranscript


@pytest.fixture
def cleaner():
    return TranscriptCleaner()


class TestTranscriptCleaner:

    def test_removes_music_markers(self, cleaner):
        segments = [TranscriptSegment("[Music]", 0.0, 2.0)]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        assert "[Music]" not in result.cleaned_text
        assert "[music]" not in result.cleaned_text.lower()

    def test_removes_applause_markers(self, cleaner):
        segments = [
            TranscriptSegment("Great question.", 0.0, 1.5),
            TranscriptSegment("[Applause]",      1.5, 2.0),
            TranscriptSegment("Thank you.",       3.5, 1.5),
        ]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        assert "[Applause]" not in result.cleaned_text

    def test_removes_filler_words(self, cleaner):
        segments = [TranscriptSegment("Um so uh we have ah this concept.", 0.0, 4.0)]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        text = result.cleaned_text.lower()
        assert " um " not in f" {text} "
        assert " uh " not in f" {text} "

    def test_merges_segments_into_text(self, cleaner, sample_raw_transcript):
        result = cleaner.clean(sample_raw_transcript)
        assert len(result.cleaned_text) > 0
        assert "neural networks" in result.cleaned_text.lower()
        assert "backpropagation" in result.cleaned_text.lower()

    def test_capitalizes_first_letter(self, cleaner):
        segments = [TranscriptSegment("gradient descent is important.", 0.0, 3.0)]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        assert result.cleaned_text[0].isupper()

    def test_preserves_timestamps_in_segments(self, cleaner, sample_raw_transcript):
        result = cleaner.clean(sample_raw_transcript)
        # Segments with real content should preserve timestamps
        for seg in result.segments:
            assert seg.start >= 0.0
            assert seg.duration >= 0.0

    def test_empty_transcript_returns_empty(self, cleaner):
        raw = RawTranscript("v1", [], "en", "youtube_auto")
        result = cleaner.clean(raw)
        assert result.cleaned_text == ""
        assert result.word_count == 0

    def test_word_count_is_accurate(self, cleaner):
        segments = [TranscriptSegment("one two three four five", 0.0, 5.0)]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        assert result.word_count == 5

    def test_removes_html_entities(self, cleaner):
        segments = [TranscriptSegment("A &amp; B are &lt;important&gt;", 0.0, 3.0)]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        assert "&amp;" not in result.cleaned_text
        assert "&lt;"  not in result.cleaned_text

    def test_all_noise_segments_produce_no_content(self, cleaner):
        segments = [
            TranscriptSegment("[Music]",    0.0, 2.0),
            TranscriptSegment("[Applause]", 2.0, 1.5),
            TranscriptSegment("♪ la la ♪",  3.5, 2.0),
        ]
        raw = RawTranscript("v1", segments, "en", "youtube_auto")
        result = cleaner.clean(raw)
        # Should have no meaningful content
        assert len(result.cleaned_text.strip()) == 0 or result.word_count < 3
