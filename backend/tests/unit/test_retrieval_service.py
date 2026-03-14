"""
Unit tests — RetrievalService

Tests the retrieval logic with mocked ChromaDB and embedding service.
Verifies deduplication, filtering, timestamp formatting, and result parsing.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.rag.retrieval_service import RetrievalService, RetrievedChunk, _seconds_to_label


class TestSecondsToLabel:
    """Test the timestamp formatting helper."""

    def test_under_one_minute(self):
        assert _seconds_to_label(45.0) == "0:45"

    def test_exact_minute(self):
        assert _seconds_to_label(60.0) == "1:00"

    def test_minutes_and_seconds(self):
        assert _seconds_to_label(754.0) == "12:34"

    def test_over_one_hour(self):
        assert _seconds_to_label(3661.0) == "1:01:01"

    def test_zero(self):
        assert _seconds_to_label(0.0) == "0:00"


class TestRetrievedChunk:
    """Test the RetrievedChunk data container."""

    @pytest.fixture
    def chunk(self):
        return RetrievedChunk(
            chunk_id="abc",
            chunk_text="Neural networks are universal function approximators.",
            video_id="dQw4w9WgXcQ",
            video_title="Lecture 5",
            start_timestamp=754.0,
            end_timestamp=800.0,
            course_id="course1",
            relevance_score=0.92,
            position=3,
        )

    def test_timestamp_label(self, chunk):
        assert chunk.timestamp_label == "12:34"

    def test_youtube_url_contains_video_id(self, chunk):
        assert "dQw4w9WgXcQ" in chunk.youtube_url

    def test_youtube_url_contains_timestamp(self, chunk):
        assert "t=754s" in chunk.youtube_url

    def test_youtube_url_format(self, chunk):
        assert chunk.youtube_url.startswith("https://www.youtube.com/watch?v=")


class TestRetrievalServiceDeduplication:
    """Test the Jaccard deduplication logic in isolation."""

    def _make_chunk(self, text: str, score: float = 0.9) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=text[:8],
            chunk_text=text,
            video_id="v1",
            video_title="Test",
            start_timestamp=0.0,
            end_timestamp=10.0,
            course_id="c1",
            relevance_score=score,
            position=0,
        )

    def test_identical_chunks_deduplicated(self):
        text = "Backpropagation uses the chain rule to compute gradients efficiently."
        chunks = [
            self._make_chunk(text, 0.95),
            self._make_chunk(text, 0.90),   # Same text — should be removed
        ]
        result = RetrievalService._deduplicate(chunks)
        assert len(result) == 1

    def test_different_chunks_kept(self):
        chunks = [
            self._make_chunk("Gradient descent minimizes the loss function iteratively."),
            self._make_chunk("Neural networks have multiple layers of weighted connections."),
        ]
        result = RetrievalService._deduplicate(chunks)
        assert len(result) == 2

    def test_empty_list(self):
        assert RetrievalService._deduplicate([]) == []

    def test_single_chunk_kept(self):
        chunks = [self._make_chunk("Only one chunk here.")]
        result = RetrievalService._deduplicate(chunks)
        assert len(result) == 1

    def test_high_overlap_removed(self):
        """Two chunks with 90%+ word overlap should be deduplicated."""
        base = "The learning rate controls how fast gradient descent moves down the loss surface."
        similar = "The learning rate controls how fast gradient descent moves down the loss function."
        chunks = [
            self._make_chunk(base,    0.95),
            self._make_chunk(similar, 0.90),
        ]
        result = RetrievalService._deduplicate(chunks)
        assert len(result) == 1
        assert result[0].relevance_score == 0.95   # Higher score kept


class TestRetrievalServiceParseResults:
    """Test ChromaDB result parsing."""

    def test_parses_valid_results(self):
        raw = {
            "ids":       [["id1", "id2"]],
            "documents": [["Backprop text.", "Gradient text."]],
            "metadatas": [[
                {"course_id": "c1", "video_id": "v1", "video_title": "Lec 1",
                 "start_timestamp": 10.0, "end_timestamp": 30.0, "position": 0},
                {"course_id": "c1", "video_id": "v1", "video_title": "Lec 1",
                 "start_timestamp": 30.0, "end_timestamp": 50.0, "position": 1},
            ]],
            "distances":  [[0.1, 0.3]],   # cosine distance (lower = more similar)
        }
        chunks = RetrievalService._parse_results(raw, min_relevance=0.5)
        assert len(chunks) == 2
        assert chunks[0].relevance_score == pytest.approx(0.9)   # 1.0 - 0.1
        assert chunks[1].relevance_score == pytest.approx(0.7)

    def test_filters_by_min_relevance(self):
        raw = {
            "ids":       [["id1", "id2"]],
            "documents": [["High relevance.", "Low relevance."]],
            "metadatas": [[
                {"course_id": "c1", "video_id": "v1", "video_title": "T",
                 "start_timestamp": 0.0, "end_timestamp": 10.0, "position": 0},
                {"course_id": "c1", "video_id": "v1", "video_title": "T",
                 "start_timestamp": 10.0, "end_timestamp": 20.0, "position": 1},
            ]],
            "distances": [[0.1, 0.9]],   # second chunk: score = 0.1 (below 0.5 threshold)
        }
        chunks = RetrievalService._parse_results(raw, min_relevance=0.5)
        assert len(chunks) == 1
        assert chunks[0].chunk_text == "High relevance."

    def test_empty_results(self):
        raw = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        chunks = RetrievalService._parse_results(raw, min_relevance=0.3)
        assert chunks == []

    def test_sorted_by_relevance_descending(self):
        raw = {
            "ids":       [["id1", "id2"]],
            "documents": [["Less relevant.", "More relevant."]],
            "metadatas": [[
                {"course_id": "c1", "video_id": "v1", "video_title": "T",
                 "start_timestamp": 0.0, "end_timestamp": 10.0, "position": 0},
                {"course_id": "c1", "video_id": "v1", "video_title": "T",
                 "start_timestamp": 10.0, "end_timestamp": 20.0, "position": 1},
            ]],
            "distances": [[0.4, 0.1]],   # second has higher score
        }
        chunks = RetrievalService._parse_results(raw, min_relevance=0.3)
        assert chunks[0].chunk_text == "More relevant."
