"""
Rate Limiting Middleware
------------------------
Sliding window rate limiter backed by Redis.

Limits are applied per IP address and per authenticated user.
Different limits for different endpoint categories:

  Auth endpoints:     10 requests / minute   (prevent brute force)
  AI endpoints:       30 requests / minute   (expensive LLM calls)
  General endpoints: 120 requests / minute   (normal API use)

If Redis is unavailable, rate limiting is skipped (fail open)
so an outage doesn't take down the whole API.
"""

import time
import logging
from typing import Optional, Callable

import redis.asyncio as aioredis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import settings

logger = logging.getLogger(__name__)

# ── Rate limit rules — (requests, window_seconds) ────────────────────────────
_LIMITS: dict[str, tuple[int, int]] = {
    "auth":    (10,  60),    # 10 req / minute
    "ai":      (30,  60),    # 30 req / minute
    "default": (120, 60),    # 120 req / minute
}

# Endpoint prefix → category
_ENDPOINT_CATEGORIES: dict[str, str] = {
    "/api/v1/auth/login":           "auth",
    "/api/v1/auth/signup":          "auth",
    "/api/v1/auth/forgot-password": "auth",
    "/api/v1/auth/reset-password":  "auth",
    "/api/v1/chat/ask":             "ai",
    "/api/v1/learning/summarize":   "ai",
    "/api/v1/learning/mindmap":     "ai",
    "/api/v1/learning/exam":        "ai",
    "/api/v1/learning/topic-search":"ai",
    "/api/v1/learning/course-intelligence": "ai",
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter middleware.

    Uses Redis INCR + EXPIRE for atomic, distributed counters.
    Each (identifier, endpoint_category) pair gets its own counter.
    """

    def __init__(self, app, redis_url: str = None):
        super().__init__(app)
        self._redis_url = redis_url or settings.REDIS_URL
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Lazy Redis connection — created on first use."""
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning(f"Rate limiter: Redis connection failed: {e}")
                return None
        return self._redis

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health check and docs
        path = request.url.path
        if path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        redis = await self._get_redis()
        if redis is None:
            # Fail open — don't block requests if Redis is down
            return await call_next(request)

        category = self._get_category(path)
        max_requests, window_seconds = _LIMITS[category]
        identifier = self._get_identifier(request)
        key = f"ratelimit:{category}:{identifier}"

        try:
            current = await redis.incr(key)
            if current == 1:
                # First request in this window — set expiry
                await redis.expire(key, window_seconds)

            # Add rate limit headers to every response
            ttl = await redis.ttl(key)

            if current > max_requests:
                logger.warning(
                    f"Rate limit exceeded: {identifier} on {path} "
                    f"({current}/{max_requests})"
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Too many requests. Try again in {ttl} seconds.",
                        "code":   "RATE_LIMITED",
                    },
                    headers={
                        "Retry-After":          str(ttl),
                        "X-RateLimit-Limit":    str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset":    str(int(time.time()) + ttl),
                    },
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"]     = str(max_requests)
            response.headers["X-RateLimit-Remaining"] = str(max_requests - current)
            response.headers["X-RateLimit-Reset"]     = str(int(time.time()) + ttl)
            return response

        except Exception as e:
            # Fail open on any Redis error
            logger.warning(f"Rate limiter error (skipping): {e}")
            return await call_next(request)

    @staticmethod
    def _get_category(path: str) -> str:
        """Map a request path to a rate limit category."""
        for prefix, category in _ENDPOINT_CATEGORIES.items():
            if path.startswith(prefix):
                return category
        return "default"

    @staticmethod
    def _get_identifier(request: Request) -> str:
        """
        Identify the requester by user ID (from JWT) or IP address.
        User-based limiting is fairer — one user on shared IP doesn't
        affect others.
        """
        # Try to extract user ID from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                from app.core.security import decode_token
                payload = decode_token(token)
                if payload and payload.get("sub"):
                    return f"user:{payload['sub']}"
            except Exception:
                pass

        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"