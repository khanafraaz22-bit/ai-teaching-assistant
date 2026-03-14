"""
Auth endpoints — Phase 5.

POST /auth/signup              — Register a new account
POST /auth/login               — Login, receive JWT tokens
POST /auth/logout              — Logout (client-side token discard)
GET  /auth/verify-email        — Verify email from link
POST /auth/forgot-password     — Request password reset email
POST /auth/reset-password      — Set new password with reset token
POST /auth/refresh             — Get new access token via refresh token
GET  /auth/me                  — Get current user profile
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

from app.models.schemas import UserCreate
from app.services.auth.auth_service import AuthService
from app.core.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
_auth = AuthService()


# ── Request models ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)

class RefreshRequest(BaseModel):
    refresh_token: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/signup", status_code=status.HTTP_201_CREATED,
             summary="Register a new user account")
async def signup(payload: UserCreate):
    """
    Creates a new account and sends a verification email.
    The account cannot log in until the email is verified.
    """
    try:
        result = await _auth.signup(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/verify-email", summary="Verify email address from link",
            response_class=HTMLResponse)
async def verify_email(token: str):
    """
    Called when the user clicks the verification link in their email.
    Returns a simple HTML page (frontend can replace this later).
    """
    try:
        result = await _auth.verify_email(token)
        return HTMLResponse(f"""
        <html><body style="font-family:sans-serif;text-align:center;padding:60px">
        <h2>✅ {result['message']}</h2>
        <p>You can now close this tab and log in.</p>
        </body></html>
        """)
    except ValueError as e:
        return HTMLResponse(f"""
        <html><body style="font-family:sans-serif;text-align:center;padding:60px">
        <h2>❌ Verification Failed</h2><p>{e}</p>
        </body></html>
        """, status_code=400)


@router.post("/login", summary="Login and receive JWT tokens")
async def login(payload: LoginRequest):
    """
    Returns an access token (short-lived) and a refresh token (long-lived).
    Store the refresh token securely — use it to get new access tokens.
    """
    try:
        result = await _auth.login(payload.email, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return result


@router.post("/logout", summary="Logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    JWT tokens are stateless — logout is handled client-side by
    discarding the tokens. This endpoint confirms the action.
    Phase 6 can add a token blacklist via Redis if needed.
    """
    return {"message": "Logged out successfully."}


@router.post("/forgot-password", summary="Request a password reset email")
async def forgot_password(payload: ForgotPasswordRequest):
    """
    Sends a reset link if the email exists.
    Always returns the same message to prevent email enumeration.
    """
    result = await _auth.forgot_password(payload.email)
    return result


@router.post("/reset-password", summary="Reset password using token from email")
async def reset_password(payload: ResetPasswordRequest):
    try:
        result = await _auth.reset_password(payload.token, payload.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/refresh", summary="Get a new access token using refresh token")
async def refresh(payload: RefreshRequest):
    try:
        result = await _auth.refresh_access_token(payload.refresh_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return result


@router.get("/me", summary="Get current user profile")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the authenticated user's profile. Requires a valid access token."""
    return {
        "id":         str(current_user["_id"]),
        "email":      current_user["email"],
        "full_name":  current_user["full_name"],
        "is_verified": current_user["is_verified"],
        "created_at": current_user["created_at"],
    }
