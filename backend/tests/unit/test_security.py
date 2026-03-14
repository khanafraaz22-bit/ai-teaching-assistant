"""
Unit tests — Security (JWT + Password hashing)
"""

import time
import pytest
from datetime import timedelta
from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    create_verification_token, create_reset_token,
    decode_token,
)


class TestPasswordHashing:

    def test_hash_differs_from_plaintext(self):
        assert hash_password("secret") != "secret"

    def test_correct_password_verifies(self):
        hashed = hash_password("correctpassword")
        assert verify_password("correctpassword", hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("correctpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_empty_password_fails(self):
        hashed = hash_password("realpassword")
        assert verify_password("", hashed) is False

    def test_same_password_gives_different_hashes(self):
        # bcrypt uses random salt
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_both_salted_hashes_verify(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert verify_password("same", h1) is True
        assert verify_password("same", h2) is True


class TestAccessTokens:

    def test_encodes_and_decodes_payload(self):
        token = create_access_token({"sub": "user123", "email": "a@b.com"})
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["email"] == "a@b.com"

    def test_has_access_type(self):
        token = create_access_token({"sub": "u1"})
        assert decode_token(token)["type"] == "access"

    def test_has_future_expiry(self):
        token = create_access_token({"sub": "u1"})
        assert decode_token(token)["exp"] > time.time()

    def test_expired_token_returns_none(self):
        token = create_access_token({"sub": "u1"}, expires_delta=timedelta(seconds=-1))
        assert decode_token(token) is None

    def test_tampered_token_returns_none(self):
        token = create_access_token({"sub": "u1"})
        assert decode_token(token[:-5] + "XXXXX") is None

    def test_garbage_returns_none(self):
        assert decode_token("not.a.real.token") is None


class TestRefreshTokens:

    def test_has_refresh_type(self):
        token = create_refresh_token({"sub": "u1"})
        assert decode_token(token)["type"] == "refresh"

    def test_refresh_expires_later_than_access(self):
        access_exp  = decode_token(create_access_token({"sub": "u1"}))["exp"]
        refresh_exp = decode_token(create_refresh_token({"sub": "u1"}))["exp"]
        assert refresh_exp > access_exp


class TestSpecialTokens:

    def test_verification_token(self):
        token = create_verification_token("user@example.com")
        payload = decode_token(token)
        assert payload["type"] == "verify"
        assert payload["sub"]  == "user@example.com"

    def test_reset_token(self):
        token = create_reset_token("user@example.com")
        payload = decode_token(token)
        assert payload["type"] == "reset"
        assert payload["sub"]  == "user@example.com"

    def test_reset_expires_before_refresh(self):
        reset_exp   = decode_token(create_reset_token("u@e.com"))["exp"]
        refresh_exp = decode_token(create_refresh_token({"sub": "u1"}))["exp"]
        assert reset_exp < refresh_exp
