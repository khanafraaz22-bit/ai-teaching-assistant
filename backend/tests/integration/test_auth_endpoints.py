"""
Integration tests — Auth endpoints

Tests the full request/response cycle for auth endpoints.
MongoDB calls are mocked — no real database needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


class TestSignupEndpoint:

    def test_signup_missing_fields_returns_422(self, client):
        response = client.post("/api/v1/auth/signup", json={})
        assert response.status_code == 422
        data = response.json()
        assert "errors" in data or "detail" in data

    def test_signup_short_password_returns_422(self, client):
        response = client.post("/api/v1/auth/signup", json={
            "email": "test@example.com",
            "password": "short",
            "full_name": "Test User",
        })
        assert response.status_code == 422

    def test_signup_invalid_email_returns_422(self, client):
        response = client.post("/api/v1/auth/signup", json={
            "email": "not-an-email",
            "password": "securepassword123",
            "full_name": "Test User",
        })
        assert response.status_code == 422

    def test_signup_success(self, client):
        with patch("app.services.auth.auth_service.get_users_collection") as mock_col:
            # User doesn't exist yet
            mock_col.return_value.find_one = AsyncMock(return_value=None)
            mock_col.return_value.insert_one = AsyncMock(
                return_value=MagicMock(inserted_id="64f1a2b3c4d5e6f7a8b9c0d1")
            )

            response = client.post("/api/v1/auth/signup", json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "full_name": "New User",
            })

        assert response.status_code == 201
        data = response.json()
        assert "user_id" in data
        assert "message" in data

    def test_signup_duplicate_email_returns_400(self, client):
        existing_user = {"email": "existing@example.com", "_id": "someid"}
        with patch("app.services.auth.auth_service.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=existing_user)

            response = client.post("/api/v1/auth/signup", json={
                "email": "existing@example.com",
                "password": "securepassword123",
                "full_name": "Existing User",
            })

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()


class TestLoginEndpoint:

    def test_login_wrong_credentials_returns_401(self, client):
        with patch("app.services.auth.auth_service.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=None)

            response = client.post("/api/v1/auth/login", json={
                "email": "nobody@example.com",
                "password": "wrongpassword",
            })

        assert response.status_code == 401

    def test_login_unverified_user_returns_401(self, client):
        from app.core.security import hash_password
        unverified_user = {
            "_id": "someid",
            "email": "unverified@example.com",
            "hashed_password": hash_password("correctpassword"),
            "is_active": True,
            "is_verified": False,
        }
        with patch("app.services.auth.auth_service.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=unverified_user)

            response = client.post("/api/v1/auth/login", json={
                "email": "unverified@example.com",
                "password": "correctpassword",
            })

        assert response.status_code == 401
        assert "verify" in response.json()["detail"].lower()

    def test_login_success_returns_tokens(self, client):
        from app.core.security import hash_password
        from bson import ObjectId
        verified_user = {
            "_id": ObjectId(),
            "email": "verified@example.com",
            "hashed_password": hash_password("correctpassword"),
            "is_active": True,
            "is_verified": True,
        }
        with patch("app.services.auth.auth_service.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=verified_user)

            response = client.post("/api/v1/auth/login", json={
                "email": "verified@example.com",
                "password": "correctpassword",
            })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"


class TestProtectedEndpoints:

    def test_get_me_without_token_returns_403(self, client):
        response = client.get("/api/v1/auth/me")
        assert response.status_code in (401, 403)

    def test_get_me_with_invalid_token_returns_401(self, client):
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalidtoken"},
        )
        assert response.status_code == 401

    def test_get_me_with_valid_token(self, client, auth_headers, mock_user):
        with patch("app.core.dependencies.get_users_collection") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=mock_user)

            response = client.get("/api/v1/auth/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == mock_user["email"]
