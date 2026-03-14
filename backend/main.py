"""
Entry point for the YT Course Assistant API.
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.config.settings import settings
from app.database.mongodb import connect_to_mongo, close_mongo_connection
from app.database.vectordb import init_vector_db
from app.database.indexes import ensure_indexes
from app.core.exceptions import register_exception_handlers
from app.core.rate_limiter import RateLimitMiddleware
from app.core.security_headers import SecurityHeadersMiddleware
from app.core.request_logging import RequestLoggingMiddleware
from app.core.startup_validator import validate_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    validate_config()           # Fail fast on bad config
    await connect_to_mongo()
    await ensure_indexes()
    init_vector_db()
    yield
    await close_mongo_connection()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered learning assistant for YouTube courses.",
    lifespan=lifespan,
    # Hide API docs in production
    docs_url="/docs"        if settings.DEBUG else None,
    redoc_url="/redoc"      if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# ── Middleware (last added = first to run) ────────────────────────────────────

# 1. Request logging — outermost, times the full request
app.add_middleware(RequestLoggingMiddleware)

# 2. Security headers — on every response
app.add_middleware(SecurityHeadersMiddleware)

# 3. Rate limiting — blocks abusive clients early
app.add_middleware(RateLimitMiddleware)

# 4. CORS — tightened: explicit methods and headers only
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-Response-Time",
    ],
)

# ── Global exception handlers ─────────────────────────────────────────────────
register_exception_handlers(app)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}
