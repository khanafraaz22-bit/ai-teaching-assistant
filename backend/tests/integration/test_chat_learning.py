"""
Integration tests — Chat and Learning endpoints

Tests the full request/response cycle.
LLM and vector DB calls are mocked throughout.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestChatAsk:

    def test_ask_requires_auth(self, client):
        response = client.post("/api/v1/chat/ask", json={
            "question": "What is backpropagation?",
            "course_id": "course123",
        })
        assert response.status_code == 401

    def test_ask_question_too_short_returns_422(self, client, auth_headers, mock_user):
        with patch("app.database.mongodb.get_users_collection") as mock_users:
            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            response = client.post(
                "/api/v1/chat/ask",
                json={"question": "Hi", "course_id": "course123"},
                headers=auth_headers,
            )
        assert response.status_code == 422

    def test_ask_returns_answer_and_sources(
        self, client, auth_headers, mock_user, sample_retrieved_chunks
    ):
        from app.services.rag.llm_chain import ChatResponse, SourceReference

        mock_response = ChatResponse(
            answer="Backpropagation uses the chain rule to compute gradients.",
            sources=[
                SourceReference(
                    video_id="video_abc",
                    video_title="Lecture 3",
                    timestamp_label="2:00",
                    youtube_url="https://youtube.com/watch?v=video_abc&t=120s",
                )
            ],
            is_grounded=True,
            chunks_used=1,
        )

        with patch("app.database.mongodb.get_users_collection") as mock_users, \
             patch("app.database.mongodb.get_progress_collection") as mock_progress, \
             patch("app.services.rag.retrieval_service.RetrievalService.retrieve",
                   return_value=sample_retrieved_chunks), \
             patch("app.services.rag.llm_chain.LLMChain.ask",
                   return_value=mock_response):

            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_progress.return_value.update_one = AsyncMock()

            response = client.post(
                "/api/v1/chat/ask",
                json={
                    "question": "What is backpropagation?",
                    "course_id": "course123",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "is_grounded" in data
        assert data["is_grounded"] is True

    def test_ask_returns_session_id(
        self, client, auth_headers, mock_user, sample_retrieved_chunks
    ):
        from app.services.rag.llm_chain import ChatResponse
        mock_response = ChatResponse(
            answer="Test answer.", sources=[], is_grounded=True, chunks_used=1
        )

        with patch("app.database.mongodb.get_users_collection") as mock_users, \
             patch("app.database.mongodb.get_progress_collection") as mock_progress, \
             patch("app.services.rag.retrieval_service.RetrievalService.retrieve",
                   return_value=sample_retrieved_chunks), \
             patch("app.services.rag.llm_chain.LLMChain.ask", return_value=mock_response):

            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_progress.return_value.update_one = AsyncMock()

            response = client.post(
                "/api/v1/chat/ask",
                json={"question": "What is gradient descent?", "course_id": "c1"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        assert response.json()["session_id"]  # Not empty


class TestLearningEndpoints:

    def test_summarize_requires_auth(self, client):
        response = client.post("/api/v1/learning/summarize", json={"course_id": "c1"})
        assert response.status_code == 401

    def test_topic_search_requires_auth(self, client):
        response = client.post("/api/v1/learning/topic-search", json={
            "topic": "neural networks", "course_id": "c1"
        })
        assert response.status_code == 401

    def test_mindmap_requires_auth(self, client):
        response = client.post("/api/v1/learning/mindmap", json={"course_id": "c1"})
        assert response.status_code == 401

    def test_exam_requires_auth(self, client):
        response = client.post("/api/v1/learning/exam", json={"course_id": "c1"})
        assert response.status_code == 401

    def test_exam_invalid_difficulty_returns_422(self, client, auth_headers, mock_user):
        with patch("app.database.mongodb.get_users_collection") as mock_users:
            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            response = client.post(
                "/api/v1/learning/exam",
                json={"course_id": "c1", "difficulty": "extreme"},
                headers=auth_headers,
            )
        assert response.status_code == 422

    def test_summarize_returns_correct_shape(self, client, auth_headers, mock_user):
        from app.services.learning.summarizer import SummaryResult

        mock_result = SummaryResult(
            title="Test Video",
            overview="This video covers neural networks.",
            key_points=["Point 1", "Point 2"],
            concepts_covered=["Neural networks", "Backpropagation"],
            takeaway="Backpropagation is the key algorithm.",
            video_id="video_abc",
            course_id="course_xyz",
        )

        async def mock_chunks(*args, **kwargs):
            return [{"video_title": "Test Video", "chunk_text": "...", "start_timestamp": 0.0}]

        with patch("app.database.mongodb.get_users_collection") as mock_users, \
             patch("app.services.learning.summarizer.SummarizerService._fetch_chunks",
                   side_effect=mock_chunks), \
             patch("app.services.learning.summarizer.SummarizerService._direct_summarize",
                   return_value={
                       "overview": "This video covers neural networks.",
                       "key_points": ["Point 1", "Point 2"],
                       "concepts_covered": ["Neural networks"],
                       "takeaway": "Backpropagation is the key algorithm.",
                   }):

            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)

            response = client.post(
                "/api/v1/learning/summarize",
                json={"course_id": "course_xyz", "video_id": "video_abc"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert "overview" in data
        assert "key_points" in data
        assert "concepts_covered" in data
        assert "takeaway" in data


class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_returns_version(self, client):
        response = client.get("/health")
        assert "version" in response.json()
