"""
Auth Dependencies
-----------------
FastAPI dependencies injected into protected endpoints.

Usage in any endpoint:
    @router.get("/something")
    async def my_endpoint(current_user: dict = Depends(get_current_user)):
        user_id = current_user["_id"]
        ...
"""

from fastapi import Depends, HTTPException, status
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security import decode_token
from app.database.mongodb import get_users_collection

# Extracts Bearer token from Authorization header
_bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """
    Validate the JWT token and return the user document from MongoDB.
    Raises 401 if token is missing, invalid, or expired.
    Raises 403 if the account is not verified.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    payload = decode_token(token)

    if not payload or payload.get("type") != "access":
        raise credentials_exception

    user_id: str = payload.get("sub")
    if not user_id:
        raise credentials_exception

    # Fetch the user from MongoDB
    from bson import ObjectId
    users = get_users_collection()
    try:
        user = await users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise credentials_exception

    if not user:
        raise credentials_exception

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email address not verified. Please check your inbox.",
        )

    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[dict]:
    """
    Same as get_current_user but returns None instead of raising
    for endpoints that work both authenticated and anonymous.
    """
    if not credentials:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None