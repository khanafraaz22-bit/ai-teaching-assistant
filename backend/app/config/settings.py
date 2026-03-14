"""
Centralized settings — all configuration is read from environment variables.
Never hardcode secrets. Use a .env file locally.
"""

from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from typing import List


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────
    APP_NAME: str = "YT Course Assistant"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"    # "development" | "production"

    # ── Security ──────────────────────────────────────────
    SECRET_KEY: str                         # Required — no default
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    MIN_SECRET_KEY_LENGTH: int = 32         # Enforce strong keys

    # ── MongoDB ───────────────────────────────────────────
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "yt_course_assistant"

    # ── Redis ─────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── ChromaDB ──────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_store"
    CHROMA_COLLECTION_NAME: str = "course_chunks"

    # ── OpenAI ────────────────────────────────────────────
    OPENAI_API_KEY: str                     # Required — no default
    OPENAI_LLM_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536

    # ── Chunking ──────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = 700
    CHUNK_OVERLAP_TOKENS: int = 100

    # ── RAG retrieval ─────────────────────────────────────
    RAG_TOP_K: int = 6                      # Chunks returned per query

    # ── CORS ──────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Email (for auth verification — Phase 3) ───────────
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM: str = "no-reply@ytcourseassistant.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Single shared instance — import this everywhere
settings = Settings()
