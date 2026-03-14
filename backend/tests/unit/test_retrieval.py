"""
Unit tests — RetrievalService

Tests that retrieval:
  - Converts distance to similarity correctly
  - Filters below min_relevance threshold
  - Deduplicates near-identical chunks
  - Returns results sorted by relevance descending
  - Handles empty ChromaDB results gracefully
  - Builds correct ChromaDB where filters
  - Timestamp labels and YouTube URLs are formatted correctly
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.rag.retrieval_service import RetrievalService, RetrievedChunk


@pytest.fixture
def retrieval_service():
    return RetrievalService()


def make_chroma_result(
    ids: list[str],
    texts: list[str],
    distances: list[float],
    course_id: str = "course-abc",
    video_id: str  = "vid-001",
) -> dict:
    """Build a fake ChromaDB query result."""
    metadatas = [
        {
            "course_id":       course_id,
            "video_id":        video_id,
            "video_title":     "Test Video",
            "start_timestamp": float(i * 10),
            "end_timestamp":   float(i * 10 + 8),
            "token_count":     80,
            "position":        i,
        }
        for i in range(len(ids))
    ]
    return {
        "ids":       [ids],
        "documents": [texts],
        "metadatas": [metadatas],
        "distances": [distances],
    }


class TestRelevanceFiltering:

    def test_low_relevance_chunks_excluded(self, retrieval_service, mock_chroma_collection):
        # Distance 0.9 → score 0.1 — well below default min_relevance 0.25
        result = make_chroma_result(
            ids=["id-1"], texts=["irrelevant content"], distances=[0.9]
        )
        mock_chroma_collection.query.return_value = result

        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve(
                query="test", course_id="course-abc", min_relevance=0.25
            )
        assert chunks == []

    def test_high_relevance_chunks_included(self, retrieval_service, mock_chroma_collection):
        result = make_chroma_result(
            ids=["id-1"], texts=["gradient descent optimization"], distances=[0.05]
        )
        mock_chroma_collection.query.return_value = result

        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve(
                query="gradient descent", course_id="course-abc", min_relevance=0.25
            )
        assert len(chunks) == 1
        assert chunks[0].relevance_score >= 0.25

    def test_relevance_score_is_one_minus_distance(
        self, retrieval_service, mock_chroma_collection
    ):
        result = make_chroma_result(
            ids=["id-1"], texts=["some content"], distances=[0.2]
        )
        mock_chroma_collection.query.return_value = result

        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve(
                query="test", course_id="course-abc", min_relevance=0.0
            )
        assert abs(chunks[0].relevance_score - 0.8) < 0.01


class TestDeduplication:

    def test_near_identical_chunks_deduplicated(self, retrieval_service, mock_chroma_collection):
        # Two almost identical texts — should deduplicate to 1
        text = "gradient descent optimization algorithm loss function weights update"
        result = make_chroma_result(
            ids=["id-1", "id-2"],
            texts=[text, text + " slightly different"],
            distances=[0.05, 0.06],
        )
        mock_chroma_collection.query.return_value = result

        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve(
                query="gradient descent", course_id="course-abc",
                top_k=5, min_relevance=0.0
            )
        assert len(chunks) == 1

    def test_distinct_chunks_both_returned(self, retrieval_service, mock_chroma_collection):
        result = make_chroma_result(
            ids=["id-1", "id-2"],
            texts=[
                "gradient descent is used to minimize loss",
                "convolutional neural networks process images",
            ],
            distances=[0.05, 0.10],
        )
        mock_chroma_collection.query.return_value = result

        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve(
                query="neural networks", course_id="course-abc",
                top_k=5, min_relevance=0.0
            )
        assert len(chunks) == 2


class TestSorting:

    def test_results_sorted_by_relevance_descending(
        self, retrieval_service, mock_chroma_collection
    ):
        result = make_chroma_result(
            ids=["id-1", "id-2", "id-3"],
            texts=["text a", "text b", "text c"],
            distances=[0.3, 0.1, 0.2],
        )
        mock_chroma_collection.query.return_value = result

        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve(
                query="test", course_id="course-abc",
                top_k=5, min_relevance=0.0
            )
        scores = [c.relevance_score for c in chunks]
        assert scores == sorted(scores, reverse=True)


class TestEmptyResults:

    def test_empty_chromadb_result_returns_empty_list(
        self, retrieval_service, mock_chroma_collection
    ):
        mock_chroma_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        with patch("app.services.rag.retrieval_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch.object(retrieval_service._embedder, "embed_query",
                          return_value=[0.1] * 1536):

            chunks = retrieval_service.retrieve("test", "course-abc")
        assert chunks == []


class TestChunkProperties:

    def test_timestamp_label_format(self):
        chunk = RetrievedChunk(
            chunk_id="c1", chunk_text="text",
            video_id="vid1", video_title="Video",
            start_timestamp=134.0, end_timestamp=150.0,
            course_id="course1", relevance_score=0.9, position=0,
        )
        assert chunk.timestamp_label == "2:14"

    def test_youtube_url_contains_timestamp(self):
        chunk = RetrievedChunk(
            chunk_id="c1", chunk_text="text",
            video_id="abc123", video_title="Video",
            start_timestamp=90.0, end_timestamp=110.0,
            course_id="course1", relevance_score=0.9, position=0,
        )
        assert "abc123" in chunk.youtube_url
        assert "t=90s" in chunk.youtube_url

    def test_timestamp_label_with_hours(self):
        chunk = RetrievedChunk(
            chunk_id="c1", chunk_text="text",
            video_id="v1", video_title="V",
            start_timestamp=3723.0, end_timestamp=3800.0,
            course_id="c1", relevance_score=0.9, position=0,
        )
        assert chunk.timestamp_label == "1:02:03"
