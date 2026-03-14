"""
Integration tests — Chat endpoint

Tests the /chat/ask endpoint with mocked RAG pipeline.
Verifies request validation, session handling, and response shape.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestChatAskEndpoint:

    @pytest.fixture
    def mock_chatbot_result(self):
        from app.services.rag.chatbot import AskResult
        return AskResult(
            answer="Gradient descent is an optimization algorithm that minimizes a loss function.",
            sources=[{
                "video_id":    "abc123",
                "video_title": "Lecture 4 — Optimization",
                "timestamp":   "12:34",
                "url":         "https://youtube.com/watch?v=abc123&t=754s",
            }],
            is_grounded=True,
            chunks_used=6,
            session_id="test-session-id",
        )

    def test_ask_without_auth_returns_403(self, client):
        response = client.post("/api/v1/chat/ask", json={
            "question": "What is gradient descent?",
            "course_id": "course123",
        })
        assert response.status_code in (401, 403)

    def test_ask_short_question_returns_422(self, client, auth_headers, mock_user):
        with patch("app.core.dependencies.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            response = client.post("/api/v1/chat/ask",
                headers=auth_headers,
                json={"question": "Hi", "course_id": "course123"},
            )
        assert response.status_code == 422

    def test_ask_missing_course_id_returns_422(self, client, auth_headers, mock_user):
        with patch("app.core.dependencies.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            response = client.post("/api/v1/chat/ask",
                headers=auth_headers,
                json={"question": "What is backpropagation?"},
            )
        assert response.status_code == 422

    def test_ask_returns_correct_response_shape(self, client, auth_headers, mock_user, mock_chatbot_result):
        with patch("app.core.dependencies.get_users_collection") as mock_col, \
             patch("app.api.v1.endpoints.chat._chatbot") as mock_bot, \
             patch("app.services.learning.progress_service.ProgressService.record_topic_studied",
                   new_callable=AsyncMock):

            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_bot.ask.return_value = mock_chatbot_result

            response = client.post("/api/v1/chat/ask",
                headers=auth_headers,
                json={
                    "question": "What is gradient descent?",
                    "course_id": "course123",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "is_grounded" in data
        assert "chunks_used" in data
        assert "session_id" in data

    def test_ask_generates_session_id_if_not_provided(self, client, auth_headers, mock_user, mock_chatbot_result):
        with patch("app.core.dependencies.get_users_collection") as mock_col, \
             patch("app.api.v1.endpoints.chat._chatbot") as mock_bot, \
             patch("app.services.learning.progress_service.ProgressService.record_topic_studied",
                   new_callable=AsyncMock):

            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_bot.ask.return_value = mock_chatbot_result

            response = client.post("/api/v1/chat/ask",
                headers=auth_headers,
                json={
                    "question": "What is gradient descent?",
                    "course_id": "course123",
                },
            )

        data = response.json()
        assert data["session_id"]   # Must not be empty

    def test_ask_accepts_provided_session_id(self, client, auth_headers, mock_user, mock_chatbot_result):
        session_id = "my-custom-session-id"
        mock_chatbot_result.session_id = session_id

        with patch("app.core.dependencies.get_users_collection") as mock_col, \
             patch("app.api.v1.endpoints.chat._chatbot") as mock_bot, \
             patch("app.services.learning.progress_service.ProgressService.record_topic_studied",
                   new_callable=AsyncMock):

            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_bot.ask.return_value = mock_chatbot_result

            response = client.post("/api/v1/chat/ask",
                headers=auth_headers,
                json={
                    "question": "What is gradient descent?",
                    "course_id": "course123",
                    "session_id": session_id,
                },
            )

        assert response.json()["session_id"] == session_id

    def test_ungrounded_answer_is_flagged(self, client, auth_headers, mock_user):
        from app.services.rag.chatbot import AskResult
        ungrounded_result = AskResult(
            answer="This topic doesn't appear to be covered in this course.",
            sources=[],
            is_grounded=False,
            chunks_used=0,
            session_id="session-x",
        )
        with patch("app.core.dependencies.get_users_collection") as mock_col, \
             patch("app.api.v1.endpoints.chat._chatbot") as mock_bot, \
             patch("app.services.learning.progress_service.ProgressService.record_topic_studied",
                   new_callable=AsyncMock):

            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_bot.ask.return_value = ungrounded_result

            response = client.post("/api/v1/chat/ask",
                headers=auth_headers,
                json={
                    "question": "What is quantum computing?",
                    "course_id": "course123",
                },
            )

        data = response.json()
        assert data["is_grounded"] is False
        assert data["sources"] == []
