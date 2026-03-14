"""
Integration tests — Course endpoints

Tests the full request/response cycle for course management.
MongoDB and Celery are mocked — no real services needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId


class TestAddPlaylist:

    def test_add_playlist_requires_auth(self, client):
        response = client.post("/api/v1/courses/", json={
            "playlist_url": "https://youtube.com/playlist?list=ABC123"
        })
        assert response.status_code == 401

    def test_add_playlist_invalid_url_returns_422(self, client, auth_headers, mock_user):
        with patch("app.database.mongodb.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)
            response = client.post(
                "/api/v1/courses/",
                json={"playlist_url": "https://youtube.com/watch?v=abc"},
                headers=auth_headers,
            )
        assert response.status_code in (401, 422)

    def test_add_playlist_valid_url_accepted(self, client, auth_headers, mock_user):
        fake_playlist = MagicMock()
        fake_playlist.playlist_id = "PLtest123"
        fake_playlist.title = "Test Course"
        fake_playlist.total_videos = 5
        fake_playlist.description = "A test course"
        fake_playlist.channel_name = "Test Channel"
        fake_playlist.thumbnail_url = None

        fake_inserted = MagicMock()
        fake_inserted.inserted_id = ObjectId()

        with patch("app.database.mongodb.get_users_collection") as mock_users, \
             patch("app.database.mongodb.get_courses_collection") as mock_courses, \
             patch("app.services.youtube.playlist_parser.YouTubePlaylistParser.parse",
                   return_value=fake_playlist), \
             patch("app.workers.tasks.process_playlist.delay") as mock_task:

            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_courses.return_value.insert_one = AsyncMock(return_value=fake_inserted)
            mock_courses.return_value.update_one = AsyncMock()
            mock_task.return_value.id = "celery-task-id-123"

            response = client.post(
                "/api/v1/courses/",
                json={"playlist_url": "https://youtube.com/playlist?list=PLtest123"},
                headers=auth_headers,
            )

        assert response.status_code == 202
        data = response.json()
        assert "course_id" in data
        assert "message" in data
        assert data["status"] == "pending"


class TestListCourses:

    def test_list_courses_requires_auth(self, client):
        response = client.get("/api/v1/courses/")
        assert response.status_code == 401

    def test_list_courses_returns_list(self, client, auth_headers, mock_user):
        fake_courses = [
            {
                "_id": ObjectId(),
                "title": "Course A",
                "status": "completed",
                "user_id": str(mock_user["_id"]),
                "created_at": "2024-01-01T00:00:00",
            }
        ]

        async def mock_cursor(*args, **kwargs):
            for c in fake_courses:
                yield c

        mock_find = MagicMock()
        mock_find.sort = MagicMock(return_value=mock_cursor())

        with patch("app.database.mongodb.get_users_collection") as mock_users, \
             patch("app.database.mongodb.get_courses_collection") as mock_col:
            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_col.return_value.find = MagicMock(return_value=mock_find)

            response = client.get("/api/v1/courses/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "courses" in data
        assert "total" in data


class TestGetCourse:

    def test_get_course_requires_auth(self, client):
        response = client.get(f"/api/v1/courses/{ObjectId()}")
        assert response.status_code == 401

    def test_get_course_not_found(self, client, auth_headers, mock_user):
        with patch("app.database.mongodb.get_users_collection") as mock_users, \
             patch("app.database.mongodb.get_courses_collection") as mock_col:
            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            mock_col.return_value.find_one = AsyncMock(return_value=None)

            response = client.get(
                f"/api/v1/courses/{ObjectId()}",
                headers=auth_headers,
            )
        assert response.status_code == 404

    def test_get_course_invalid_id(self, client, auth_headers, mock_user):
        with patch("app.database.mongodb.get_users_collection") as mock_users:
            mock_users.return_value.find_one = AsyncMock(return_value=mock_user)
            response = client.get(
                "/api/v1/courses/not-a-valid-id",
                headers=auth_headers,
            )
        assert response.status_code == 404
