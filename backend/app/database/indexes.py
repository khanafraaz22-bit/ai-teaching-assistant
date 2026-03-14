"""
MongoDB Index Setup
-------------------
Run this once after first deployment, or on every startup (idempotent).
Indexes are essential for query performance at scale.

Usage:
    python -m app.database.indexes

Or call ensure_indexes() from main.py lifespan (already wired in).
"""

import asyncio
import logging

from app.database.mongodb import (
    connect_to_mongo,
    get_users_collection,
    get_courses_collection,
    get_videos_collection,
    get_transcripts_collection,
    get_chunks_collection,
    get_progress_collection,
    get_quiz_results_collection,
)

logger = logging.getLogger(__name__)


async def ensure_indexes() -> None:
    """Create all indexes. Safe to call multiple times (idempotent)."""
    logger.info("Creating MongoDB indexes...")

    # ── users ─────────────────────────────────────────────────────────────────
    users = get_users_collection()
    await users.create_index("email", unique=True)
    await users.create_index("verification_token", sparse=True)
    await users.create_index("reset_token", sparse=True)

    # ── courses ───────────────────────────────────────────────────────────────
    courses = get_courses_collection()
    await courses.create_index("user_id")
    await courses.create_index("playlist_id")
    await courses.create_index([("user_id", 1), ("created_at", -1)])
    # Prevent duplicate playlists per user
    await courses.create_index(
        [("user_id", 1), ("playlist_id", 1)],
        unique=True,
        sparse=True,
    )

    # ── videos ────────────────────────────────────────────────────────────────
    videos = get_videos_collection()
    await videos.create_index("course_id")
    await videos.create_index("video_id")
    await videos.create_index([("course_id", 1), ("position", 1)])

    # ── transcripts ───────────────────────────────────────────────────────────
    transcripts = get_transcripts_collection()
    await transcripts.create_index("video_id", unique=True)
    await transcripts.create_index("course_id")

    # ── chunks ────────────────────────────────────────────────────────────────
    chunks = get_chunks_collection()
    await chunks.create_index("chunk_id", unique=True)
    await chunks.create_index("course_id")
    await chunks.create_index("video_id")
    await chunks.create_index([("course_id", 1), ("video_id", 1), ("position", 1)])

    # ── progress ──────────────────────────────────────────────────────────────
    progress = get_progress_collection()
    await progress.create_index([("user_id", 1), ("course_id", 1)], unique=True)

    # ── quiz_results ──────────────────────────────────────────────────────────
    quiz = get_quiz_results_collection()
    await quiz.create_index("user_id")
    await quiz.create_index([("user_id", 1), ("course_id", 1)])

    logger.info("✅ All MongoDB indexes created.")


if __name__ == "__main__":
    async def _run():
        await connect_to_mongo()
        await ensure_indexes()

    asyncio.run(_run())
