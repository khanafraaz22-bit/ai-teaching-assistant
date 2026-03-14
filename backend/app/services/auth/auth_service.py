"""
Auth Service
------------
All user account operations:
  - signup
  - login
  - email verification
  - forgot password / reset password
  - token refresh

No HTTP logic here — that lives in the endpoint.
This service only deals with data and business rules.
"""

import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    create_verification_token,
    create_reset_token,
    decode_token,
)
from app.database.mongodb import get_users_collection
from app.models.schemas import UserCreate, UserInDB
from app.services.auth.email_service import EmailService

logger = logging.getLogger(__name__)


class AuthService:

    def __init__(self):
        self._email = EmailService()

    # ── Signup ────────────────────────────────────────────────────────────────

    async def signup(self, data: UserCreate) -> dict:
        """
        Register a new user.
        Sends a verification email.
        Raises ValueError if email already exists.
        """
        users = get_users_collection()

        # Check for duplicate email
        existing = await users.find_one({"email": data.email.lower()})
        if existing:
            raise ValueError("An account with this email already exists.")

        # Hash password and create verification token
        verification_token = create_verification_token(data.email)
        user_doc = UserInDB(
            email=data.email.lower(),
            full_name=data.full_name.strip(),
            hashed_password=hash_password(data.password),
            is_active=True,
            is_verified=False,
            verification_token=verification_token,
        )

        result = await users.insert_one(
            user_doc.model_dump(by_alias=False, exclude={"id"})
        )
        user_id = str(result.inserted_id)

        # Send verification email (non-blocking — failure doesn't break signup)
        try:
            self._email.send_verification_email(data.email, verification_token)
        except Exception as e:
            logger.warning(f"Verification email failed for {data.email}: {e}")

        logger.info(f"New user registered: {data.email} (id={user_id})")
        return {"user_id": user_id, "email": data.email, "message": "Account created. Please verify your email."}

    # ── Email Verification ────────────────────────────────────────────────────

    async def verify_email(self, token: str) -> dict:
        """
        Verify a user's email address using the token from the verification email.
        """
        payload = decode_token(token)
        if not payload or payload.get("type") != "verify":
            raise ValueError("Invalid or expired verification link.")

        email = payload.get("sub")
        users = get_users_collection()
        user = await users.find_one({"email": email})

        if not user:
            raise ValueError("User not found.")

        if user.get("is_verified"):
            return {"message": "Email already verified. You can log in."}

        await users.update_one(
            {"email": email},
            {"$set": {
                "is_verified": True,
                "verification_token": None,
                "updated_at": datetime.utcnow(),
            }},
        )

        logger.info(f"Email verified: {email}")
        return {"message": "Email verified successfully. You can now log in."}

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(self, email: str, password: str) -> dict:
        """
        Authenticate a user and return access + refresh tokens.
        Raises ValueError on bad credentials.
        """
        users = get_users_collection()
        user = await users.find_one({"email": email.lower()})

        # Same error message for both "user not found" and "wrong password"
        # — prevents email enumeration attacks
        if not user or not verify_password(password, user["hashed_password"]):
            raise ValueError("Incorrect email or password.")

        if not user.get("is_active"):
            raise ValueError("This account has been deactivated.")

        if not user.get("is_verified"):
            raise ValueError("Please verify your email before logging in.")

        user_id = str(user["_id"])
        token_data = {"sub": user_id, "email": user["email"]}

        access_token  = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        logger.info(f"User logged in: {email}")
        return {
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "token_type":    "bearer",
            "user": {
                "id":        user_id,
                "email":     user["email"],
                "full_name": user["full_name"],
            },
        }

    # ── Forgot Password ───────────────────────────────────────────────────────

    async def forgot_password(self, email: str) -> dict:
        """
        Generate a reset token and email it to the user.
        Always returns success — never reveals if email exists (security).
        """
        users = get_users_collection()
        user = await users.find_one({"email": email.lower()})

        if user:
            reset_token = create_reset_token(email)
            await users.update_one(
                {"email": email.lower()},
                {"$set": {
                    "reset_token": reset_token,
                    "reset_token_expires": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }},
            )
            try:
                self._email.send_password_reset_email(email, reset_token)
            except Exception as e:
                logger.warning(f"Reset email failed for {email}: {e}")

        # Always return the same message — don't reveal if email exists
        return {"message": "If an account with that email exists, a reset link has been sent."}

    # ── Reset Password ────────────────────────────────────────────────────────

    async def reset_password(self, token: str, new_password: str) -> dict:
        """
        Reset a user's password using the token from the reset email.
        """
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters.")

        payload = decode_token(token)
        if not payload or payload.get("type") != "reset":
            raise ValueError("Invalid or expired reset link.")

        email = payload.get("sub")
        users = get_users_collection()
        user = await users.find_one({"email": email})

        if not user:
            raise ValueError("User not found.")

        # Ensure the token matches what's stored (one-time use)
        if user.get("reset_token") != token:
            raise ValueError("Reset link has already been used.")

        await users.update_one(
            {"email": email},
            {"$set": {
                "hashed_password": hash_password(new_password),
                "reset_token":     None,
                "reset_token_expires": None,
                "updated_at":      datetime.utcnow(),
            }},
        )

        logger.info(f"Password reset successful: {email}")
        return {"message": "Password updated successfully. You can now log in."}

    # ── Refresh Token ─────────────────────────────────────────────────────────

    async def refresh_access_token(self, refresh_token: str) -> dict:
        """
        Issue a new access token using a valid refresh token.
        """
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise ValueError("Invalid or expired refresh token.")

        user_id = payload.get("sub")
        users = get_users_collection()
        user = await users.find_one({"_id": ObjectId(user_id)})

        if not user or not user.get("is_active"):
            raise ValueError("User not found or deactivated.")

        token_data = {"sub": user_id, "email": user["email"]}
        new_access_token = create_access_token(token_data)

        return {"access_token": new_access_token, "token_type": "bearer"}
