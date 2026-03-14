"""
Celery Tasks
------------
All background jobs are defined here.

Current tasks:
  - process_playlist   The main ingestion pipeline (Phase 2 logic)

Each task:
  1. Updates course status in MongoDB throughout execution
  2. Reports real-time progress (video N of M)
  3. Retries automatically on failure (up to 3 times)
  4. Stores final result (success/failure) in Redis

Important: Celery tasks are synchronous functions.
Async MongoDB calls are run via asyncio.run() inside each task.
"""

import asyncio
import logging
from datetime import datetime

from celery import Task

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ── Base task class with async support ────────────────────────────────────────

class AsyncTask(Task):
    """
    Base class that lets Celery tasks run async code.
    Reuses a single event loop per worker process.
    """
    _loop = None

    def run_async(self, coro):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)


# ── Task: Process Playlist ────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    base=AsyncTask,
    name="tasks.process_playlist",
    max_retries=3,
    default_retry_delay=60,
)
def process_playlist(
    self: AsyncTask,
    playlist_url: str,
    user_id: str,
    course_id: str,
) -> dict:
    """
    Full ingestion pipeline for a YouTube playlist.

    Args:
        playlist_url:  The YouTube playlist URL.
        user_id:       MongoDB user ID (string).
        course_id:     Pre-created MongoDB course ID (string).

    Returns dict with:
        status:        "completed" | "failed"
        course_id:     The course ID
        total_videos:  Number of videos processed
        total_chunks:  Number of chunks stored
        error:         Error message if failed
    """
    logger.info(
        f"[Task {self.request.id}] Starting ingestion: "
        f"course={course_id}, user={user_id}"
    )

    try:
        result = self.run_async(
            _run_ingestion(
                task=self,
                playlist_url=playlist_url,
                user_id=user_id,
                course_id=course_id,
            )
        )
        return result

    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Ingestion failed: {exc}")

        # Update course status to FAILED in MongoDB
        self.run_async(_mark_course_failed(course_id, str(exc)))

        # Retry if under the retry limit
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error(f"[Task {self.request.id}] Max retries exceeded for course {course_id}")
            return {
                "status": "failed",
                "course_id": course_id,
                "error": str(exc),
            }


# ── Async implementation ──────────────────────────────────────────────────────

async def _run_ingestion(
    task: AsyncTask,
    playlist_url: str,
    user_id: str,
    course_id: str,
) -> dict:
    """
    The actual async ingestion logic.
    Separated from the task so it can be tested independently.
    """
    # Import here to avoid circular imports at module load time
    from app.database.mongodb import connect_to_mongo
    from app.database.vectordb import init_vector_db
    from app.services.youtube.playlist_parser import YouTubePlaylistParser
    from app.services.youtube.transcript_extractor import TranscriptExtractor
    from app.services.youtube.transcript_cleaner import TranscriptCleaner
    from app.services.youtube.chunker import TranscriptChunker
    from app.services.rag.embedding_service import EmbeddingService
    from app.services.youtube.ingestion_service import CourseIngestionService
    from app.models.schemas import ProcessingStatus
    from app.database.mongodb import get_courses_collection

    # Each Celery worker needs its own DB connections
    await connect_to_mongo()
    init_vector_db()

    # Update status to PROCESSING
    await _update_course_status(course_id, ProcessingStatus.PROCESSING)

    # Parse playlist to get video list
    parser = YouTubePlaylistParser()
    playlist_meta = parser.parse(playlist_url)
    total_videos = len(playlist_meta.videos)

    extractor = TranscriptExtractor(use_whisper_fallback=False)
    cleaner   = TranscriptCleaner()
    chunker   = TranscriptChunker()
    embedder  = EmbeddingService()

    from app.database.mongodb import (
        get_videos_collection,
        get_transcripts_collection,
        get_chunks_collection,
    )
    from app.models.schemas import VideoInDB
    from bson import ObjectId

    # Create video documents
    videos_col = get_videos_collection()
    video_docs = [
        VideoInDB(
            course_id=course_id,
            user_id=user_id,
            video_id=v.video_id,
            title=v.title,
            url=v.url,
            position=v.position,
            duration_seconds=v.duration_seconds,
            thumbnail_url=v.thumbnail_url,
        ).model_dump(by_alias=False, exclude={"id"})
        for v in playlist_meta.videos
    ]
    insert_result = await videos_col.insert_many(video_docs)
    video_db_ids  = [str(oid) for oid in insert_result.inserted_ids]

    total_chunks = 0
    videos_done  = 0

    for video_meta, video_db_id in zip(playlist_meta.videos, video_db_ids):
        vid = video_meta.video_id

        # Report progress — stored in Celery result backend (Redis)
        task.update_state(
            state="PROGRESS",
            meta={
                "current":       videos_done,
                "total":         total_videos,
                "current_video": video_meta.title,
                "course_id":     course_id,
            },
        )

        try:
            # Mark video as processing
            await videos_col.update_one(
                {"_id": ObjectId(video_db_id)},
                {"$set": {"status": ProcessingStatus.PROCESSING.value}},
            )

            # Extract transcript
            raw_transcript = extractor.extract(vid)
            if not raw_transcript:
                logger.warning(f"[{vid}] No transcript — skipping.")
                await videos_col.update_one(
                    {"_id": ObjectId(video_db_id)},
                    {"$set": {"status": ProcessingStatus.COMPLETED.value}},
                )
                videos_done += 1
                continue

            # Clean → chunk → embed
            cleaned = cleaner.clean(raw_transcript)

            from app.models.schemas import TranscriptInDB, TranscriptSegment as TSegment
            transcript_doc = TranscriptInDB(
                video_id=vid,
                course_id=course_id,
                raw_segments=[
                    TSegment(text=s.text, start=s.start, duration=s.duration)
                    for s in raw_transcript.segments
                ],
                cleaned_text=cleaned.cleaned_text,
                language=raw_transcript.language,
                source=raw_transcript.source,
            )
            transcripts_col = get_transcripts_collection()
            await transcripts_col.insert_one(
                transcript_doc.model_dump(by_alias=False, exclude={"id"})
            )

            chunks = chunker.chunk(
                cleaned_text=cleaned.cleaned_text,
                segments=cleaned.segments,
                video_id=vid,
                video_title=video_meta.title,
                course_id=course_id,
            )

            if chunks:
                embedder.embed_and_store(chunks)

                from app.models.schemas import ChunkInDB
                chunk_docs = [
                    ChunkInDB(
                        chunk_id=c.chunk_id,
                        course_id=c.course_id,
                        video_id=c.video_id,
                        video_title=c.video_title,
                        chunk_text=c.chunk_text,
                        start_timestamp=c.start_timestamp,
                        end_timestamp=c.end_timestamp,
                        token_count=c.token_count,
                        position=c.position,
                    ).model_dump(by_alias=False, exclude={"id"})
                    for c in chunks
                ]
                chunks_col = get_chunks_collection()
                await chunks_col.insert_many(chunk_docs)
                total_chunks += len(chunks)

            await videos_col.update_one(
                {"_id": ObjectId(video_db_id)},
                {"$set": {
                    "status": ProcessingStatus.COMPLETED.value,
                    "transcript_available": True,
                }},
            )
            logger.info(f"[{vid}] ✅ Done. {len(chunks) if chunks else 0} chunks.")

        except Exception as e:
            logger.error(f"[{vid}] Failed: {e}")
            await videos_col.update_one(
                {"_id": ObjectId(video_db_id)},
                {"$set": {"status": ProcessingStatus.FAILED.value}},
            )

        videos_done += 1

    # Mark course complete
    await _update_course_status(course_id, ProcessingStatus.COMPLETED)

    logger.info(
        f"✅ Ingestion complete — course={course_id}, "
        f"videos={total_videos}, chunks={total_chunks}"
    )
    return {
        "status":       "completed",
        "course_id":    course_id,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
    }


# ── Helper coroutines ─────────────────────────────────────────────────────────

async def _update_course_status(course_id: str, status) -> None:
    from bson import ObjectId
    from app.database.mongodb import get_courses_collection
    courses = get_courses_collection()
    await courses.update_one(
        {"_id": ObjectId(course_id)},
        {"$set": {"status": status.value, "updated_at": datetime.utcnow()}},
    )


async def _mark_course_failed(course_id: str, error_message: str) -> None:
    from bson import ObjectId
    from app.database.mongodb import get_courses_collection
    from app.models.schemas import ProcessingStatus
    courses = get_courses_collection()
    await courses.update_one(
        {"_id": ObjectId(course_id)},
        {"$set": {
            "status":        ProcessingStatus.FAILED.value,
            "error_message": error_message,
            "updated_at":    datetime.utcnow(),
        }},
    )
