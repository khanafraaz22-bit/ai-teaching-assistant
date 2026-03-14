"""
Integration tests — Auth API Endpoints

Uses FastAPI's TestClient to make real HTTP requests to the app.
MongoDB and email are mocked — no real DB or SMTP needed.

Run with: pytest tests/integration/test_auth_api.py -m integration
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    """
    TestClient with MongoDB and email mocked out.
    No real database — all collection calls are mocked.
    """
    with patch("app.database.mongodb.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.mongodb.close_mongo_connection", new_callable=AsyncMock), \
         patch("app.database.indexes.ensure_indexes", new_callable=AsyncMock), \
         patch("app.database.vectordb.init_vector_db"), \
         patch("app.core.startup_validator.validate_config"):

        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _mock_users_collection(existing_user=None, inserted_id="507f1f77bcf86cd799439011"):
    """Build a mock Motor collection for users."""
    mock_col = AsyncMock()
    mock_col.find_one   = AsyncMock(return_value=existing_user)
    mock_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))
    mock_col.update_one = AsyncMock(return_value=None)
    return mock_col


class TestSignup:

    def test_signup_new_user_returns_201(self, client):
        mock_col = _mock_users_collection(existing_user=None)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col), \
             patch("app.services.auth.email_service.EmailService.send_verification_email"):

            response = client.post("/api/v1/auth/signup", json={
                "email":     "newuser@example.com",
                "password":  "strongpassword123",
                "full_name": "New User",
            })

        assert response.status_code == 201
        data = response.json()
        assert "user_id" in data
        assert data["email"] == "newuser@example.com"

    def test_duplicate_email_returns_400(self, client):
        existing = {"email": "existing@example.com", "hashed_password": "hash"}
        mock_col = _mock_users_collection(existing_user=existing)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col):
            response = client.post("/api/v1/auth/signup", json={
                "email":     "existing@example.com",
                "password":  "strongpassword123",
                "full_name": "Existing User",
            })

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_invalid_email_returns_422(self, client):
        response = client.post("/api/v1/auth/signup", json={
            "email":     "not-an-email",
            "password":  "strongpassword123",
            "full_name": "User",
        })
        assert response.status_code == 422

    def test_short_password_returns_422(self, client):
        response = client.post("/api/v1/auth/signup", json={
            "email":     "user@example.com",
            "password":  "short",
            "full_name": "User",
        })
        assert response.status_code == 422


class TestLogin:

    def _make_verified_user(self):
        from app.core.security import hash_password
        return {
            "_id":             "507f1f77bcf86cd799439011",
            "email":           "user@example.com",
            "hashed_password": hash_password("correctpassword"),
            "full_name":       "Test User",
            "is_active":       True,
            "is_verified":     True,
        }

    def test_correct_credentials_returns_tokens(self, client):
        user = self._make_verified_user()
        mock_col = _mock_users_collection(existing_user=user)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col):
            response = client.post("/api/v1/auth/login", json={
                "email":    "user@example.com",
                "password": "correctpassword",
            })

        assert response.status_code == 200
        data = response.json()
        assert "access_token"  in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_wrong_password_returns_401(self, client):
        user = self._make_verified_user()
        mock_col = _mock_users_collection(existing_user=user)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col):
            response = client.post("/api/v1/auth/login", json={
                "email":    "user@example.com",
                "password": "wrongpassword",
            })

        assert response.status_code == 401

    def test_unknown_email_returns_401(self, client):
        mock_col = _mock_users_collection(existing_user=None)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col):
            response = client.post("/api/v1/auth/login", json={
                "email":    "ghost@example.com",
                "password": "anypassword",
            })

        assert response.status_code == 401

    def test_unverified_user_returns_401(self, client):
        user = self._make_verified_user()
        user["is_verified"] = False
        mock_col = _mock_users_collection(existing_user=user)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col):
            response = client.post("/api/v1/auth/login", json={
                "email":    "user@example.com",
                "password": "correctpassword",
            })

        assert response.status_code == 401


class TestProtectedEndpoints:

    def test_missing_token_returns_403(self, client):
        response = client.get("/api/v1/courses/")
        # 403 from HTTPBearer when no token provided
        assert response.status_code in (401, 403)

    def test_invalid_token_returns_401(self, client):
        response = client.get(
            "/api/v1/courses/",
            headers={"Authorization": "Bearer invalid.token.here"}
        )
        assert response.status_code == 401

    def test_valid_token_reaches_endpoint(self, client):
        from app.core.security import create_access_token, hash_password
        from bson import ObjectId

        user_id = "507f1f77bcf86cd799439011"
        token = create_access_token({"sub": user_id, "email": "u@e.com"})

        verified_user = {
            "_id":             ObjectId(user_id),
            "email":           "u@e.com",
            "hashed_password": hash_password("pw"),
            "full_name":       "Test",
            "is_active":       True,
            "is_verified":     True,
        }
        mock_users = _mock_users_collection(existing_user=verified_user)
        mock_courses = AsyncMock()
        mock_courses.find.return_value.__aiter__ = AsyncMock(return_value=iter([]))

        with patch("app.database.mongodb.get_users_collection", return_value=mock_users), \
             patch("app.database.mongodb.get_courses_collection", return_value=mock_courses):

            response = client.get(
                "/api/v1/courses/",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200


class TestForgotPassword:

    def test_always_returns_200_regardless_of_email(self, client):
        mock_col = _mock_users_collection(existing_user=None)

        with patch("app.database.mongodb.get_users_collection", return_value=mock_col):
            response = client.post("/api/v1/auth/forgot-password", json={
                "email": "ghost@example.com"
            })

        # Must always return 200 — prevents email enumeration
        assert response.status_code == 200
