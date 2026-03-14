"""
MongoDB connection using Motor (async driver).
Connection is opened at app startup and closed on shutdown.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Module-level client — shared across the entire app lifetime
_client: Optional[AsyncIOMotorClient] = None


async def connect_to_mongo() -> None:
    global _client
    logger.info("Connecting to MongoDB...")
    _client = AsyncIOMotorClient(settings.MONGODB_URL)
    # Ping to confirm connection
    await _client.admin.command("ping")
    logger.info("✅ MongoDB connected.")


async def close_mongo_connection() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Return the application database. Used as a FastAPI dependency."""
    if _client is None:
        raise RuntimeError("MongoDB client is not initialized.")
    return _client[settings.MONGODB_DB_NAME]


# ── Collection helpers ────────────────────────────────────────────────────────
# These thin wrappers keep collection names in one place.

def get_users_collection():
    return get_database()["users"]

def get_courses_collection():
    return get_database()["courses"]

def get_videos_collection():
    return get_database()["videos"]

def get_transcripts_collection():
    return get_database()["transcripts"]

def get_chunks_collection():
    return get_database()["chunks"]

def get_progress_collection():
    return get_database()["progress"]

def get_quiz_results_collection():
    return get_database()["quiz_results"]