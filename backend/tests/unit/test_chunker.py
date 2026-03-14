"""
Unit tests — TranscriptChunker

Tests that the chunker:
  - Produces chunks from valid text
  - Assigns unique UUIDs per chunk
  - Preserves video and course metadata
  - Assigns sequential positions
  - Validates timestamps
  - Handles empty input gracefully
  - Respects approximate token size limit
  - Creates word overlap between consecutive chunks
"""

import pytest
from app.services.youtube.chunker import TranscriptChunker, TranscriptChunk


@pytest.fixture
def chunker():
    return TranscriptChunker(chunk_size=100, overlap=20)


@pytest.fixture
def long_text():
    sentence = "Gradient descent is an iterative optimization algorithm used in machine learning. "
    return sentence * 30


class TestChunkProduction:

    def test_valid_text_produces_chunks(self, chunker, sample_cleaned_text, sample_segments):
        chunks = chunker.chunk(
            cleaned_text=sample_cleaned_text,
            segments=sample_segments,
            video_id="vid_001",
            video_title="Test Video",
            course_id="course_001",
        )
        assert len(chunks) >= 1
        assert all(isinstance(c, TranscriptChunk) for c in chunks)

    def test_empty_text_produces_no_chunks(self, chunker, sample_segments):
        chunks = chunker.chunk("", sample_segments, "v", "T", "c")
        assert chunks == []

    def test_whitespace_only_produces_no_chunks(self, chunker, sample_segments):
        chunks = chunker.chunk("   \n\t  ", sample_segments, "v", "T", "c")
        assert chunks == []

    def test_long_text_produces_multiple_chunks(self, chunker, long_text, sample_segments):
        chunks = chunker.chunk(long_text, sample_segments, "v", "T", "c")
        assert len(chunks) > 1


class TestChunkMetadata:

    def test_chunk_ids_are_unique(self, chunker, long_text, sample_segments):
        chunks = chunker.chunk(long_text, sample_segments, "v", "T", "c")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_video_metadata_is_preserved(self, chunker, sample_cleaned_text, sample_segments):
        chunks = chunker.chunk(
            sample_cleaned_text, sample_segments,
            video_id="vid_xyz",
            video_title="My Video",
            course_id="course_abc",
        )
        for chunk in chunks:
            assert chunk.video_id    == "vid_xyz"
            assert chunk.video_title == "My Video"
            assert chunk.course_id   == "course_abc"

    def test_positions_are_sequential(self, chunker, long_text, sample_segments):
        chunks = chunker.chunk(long_text, sample_segments, "v", "T", "c")
        assert [c.position for c in chunks] == list(range(len(chunks)))

    def test_timestamps_are_valid(self, chunker, sample_cleaned_text, sample_segments):
        chunks = chunker.chunk(sample_cleaned_text, sample_segments, "v", "T", "c")
        for chunk in chunks:
            assert chunk.start_timestamp >= 0
            assert chunk.end_timestamp >= chunk.start_timestamp

    def test_token_counts_are_positive(self, chunker, sample_cleaned_text, sample_segments):
        chunks = chunker.chunk(sample_cleaned_text, sample_segments, "v", "T", "c")
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_texts_are_non_empty(self, chunker, sample_cleaned_text, sample_segments):
        chunks = chunker.chunk(sample_cleaned_text, sample_segments, "v", "T", "c")
        for chunk in chunks:
            assert chunk.chunk_text.strip() != ""


class TestTokenLimits:

    def test_chunks_respect_size_limit(self, long_text, sample_segments):
        chunk_size = 80
        chunker = TranscriptChunker(chunk_size=chunk_size, overlap=10)
        chunks = chunker.chunk(long_text, sample_segments, "v", "T", "c")
        for chunk in chunks:
            # 20% buffer for sentence boundary rounding
            assert chunk.token_count <= chunk_size * 1.2

    def test_consecutive_chunks_share_words(self, long_text, sample_segments):
        chunker = TranscriptChunker(chunk_size=100, overlap=30)
        chunks = chunker.chunk(long_text, sample_segments, "v", "T", "c")
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks")
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].chunk_text.lower().split())
            words_b = set(chunks[i + 1].chunk_text.lower().split())
            assert len(words_a & words_b) > 0
