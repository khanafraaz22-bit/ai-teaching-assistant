"""
Request Logging Middleware
--------------------------
Logs every request with timing, status, and user identity.

Useful for:
  - Debugging slow endpoints
  - Auditing who called what
  - Detecting abuse patterns

Sensitive paths (auth endpoints) are logged without bodies.
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths where we never log request bodies
_SENSITIVE_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/signup",
    "/api/v1/auth/reset-password",
    "/api/v1/auth/forgot-password",
}


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip noisy health check logs
        if request.url.path == "/health":
            return await call_next(request)

        start_time = time.perf_counter()
        user_id    = self._extract_user_id(request)

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        status_code = response.status_code

        log_fn = logger.warning if status_code >= 400 else logger.info

        log_fn(
            f"{request.method} {request.url.path} "
            f"→ {status_code} "
            f"({duration_ms}ms) "
            f"user={user_id}"
        )

        # Add timing header for debugging
        response.headers["X-Response-Time"] = f"{duration_ms}ms"
        return response

    @staticmethod
    def _extract_user_id(request: Request) -> str:
        """Extract user ID from JWT without full validation (just for logging)."""
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return "anonymous"
        try:
            from app.core.security import decode_token
            payload = decode_token(auth[7:])
            return payload.get("sub", "unknown") if payload else "invalid_token"
        except Exception:
            return "unknown"
