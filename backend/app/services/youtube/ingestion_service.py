"""
Course Ingestion Service
------------------------
Orchestrates the full processing pipeline:

  Playlist URL
    → Parse playlist metadata
    → Store Course + Video documents in MongoDB
    → For each video:
        → Extract transcript
        → Clean transcript
        → Chunk transcript
        → Generate & store embeddings (ChromaDB)
        → Store chunks + transcript in MongoDB
    → Update course status

This is the central coordinator. Individual steps are handled by
focused services (parser, extractor, cleaner, chunker, embedder).

Background job note:
  Currently runs synchronously. Phase 5 will wrap this in a Celery task.
  The interface is already designed so that swap is trivial.
"""

import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId

from app.database.mongodb import (
    get_courses_collection,
    get_videos_collection,
    get_transcripts_collection,
    get_chunks_collection,
)
from app.models.schemas import (
    CourseInDB,
    VideoInDB,
    TranscriptInDB,
    TranscriptSegment as TranscriptSegmentSchema,
    ChunkInDB,
    ProcessingStatus,
)
from app.services.youtube.playlist_parser import YouTubePlaylistParser
from app.services.youtube.transcript_extractor import TranscriptExtractor
from app.services.youtube.transcript_cleaner import TranscriptCleaner
from app.services.youtube.chunker import TranscriptChunker
from app.services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class CourseIngestionService:
    """
    End-to-end ingestion pipeline for a YouTube playlist.

    Usage:
        service = CourseIngestionService()
        course_id = await service.ingest(playlist_url, user_id)
    """

    def __init__(self, use_whisper_fallback: bool = False):
        self._parser = YouTubePlaylistParser()
        self._extractor = TranscriptExtractor(use_whisper_fallback=use_whisper_fallback)
        self._cleaner = TranscriptCleaner()
        self._chunker = TranscriptChunker()
        self._embedder = EmbeddingService()

    # ── Main entry point ──────────────────────────────────────────────────────

    async def ingest(self, playlist_url: str, user_id: str) -> str:
        """
        Ingest a YouTube playlist — creates the Course document itself.
        Returns the MongoDB course_id (str).
        """
        logger.info(f"Starting ingestion: {playlist_url} for user {user_id}")
        playlist_meta = self._parser.parse(playlist_url)
        course_id = await self._create_course(playlist_meta, user_id)
        await self._process_all_videos(playlist_meta, course_id, user_id)
        return course_id

    async def ingest_from_existing_course(
        self, playlist_url: str, user_id: str, course_id: str
    ) -> None:
        """
        Ingest using a Course document that already exists in MongoDB.
        Used by the background task (course doc created in the endpoint for
        immediate ID response).
        """
        logger.info(f"Ingesting into existing course: {course_id}")
        playlist_meta = self._parser.parse(playlist_url)
        await self._process_all_videos(playlist_meta, course_id, user_id)

    async def _process_all_videos(
        self, playlist_meta, course_id: str, user_id: str
    ) -> None:
        """Shared pipeline: create videos → process each → finalize course."""
        try:
            video_db_ids = await self._create_videos(playlist_meta, course_id, user_id)
            total_chunks = 0

            for video_meta, video_db_id in zip(playlist_meta.videos, video_db_ids):
                try:
                    chunks_count = await self._process_video(
                        video_meta=video_meta,
                        video_db_id=video_db_id,
                        course_id=course_id,
                    )
                    total_chunks += chunks_count
                except Exception as e:
                    logger.error(f"[{video_meta.video_id}] Failed: {e}")
                    await self._update_video_status(video_db_id, ProcessingStatus.FAILED)

            await self._update_course_status(course_id, ProcessingStatus.COMPLETED)
            logger.info(
                f"✅ Ingestion complete. Course: {course_id} | "
                f"Videos: {len(playlist_meta.videos)} | Chunks: {total_chunks}"
            )

        except Exception as e:
            logger.error(f"Ingestion failed for course {course_id}: {e}")
            await self._update_course_status(
                course_id=course_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
            )
            raise

    # ── Per-video processing ──────────────────────────────────────────────────

    async def _process_video(
        self,
        video_meta,
        video_db_id: str,
        course_id: str,
    ) -> int:
        """
        Full pipeline for a single video.
        Returns number of chunks stored.
        """
        vid = video_meta.video_id
        logger.info(f"[{vid}] Processing: {video_meta.title}")

        await self._update_video_status(video_db_id, ProcessingStatus.PROCESSING)

        # Extract
        raw_transcript = self._extractor.extract(vid)
        if not raw_transcript:
            logger.warning(f"[{vid}] No transcript — skipping.")
            await self._update_video_status(video_db_id, ProcessingStatus.COMPLETED)
            return 0

        # Clean
        cleaned = self._cleaner.clean(raw_transcript)

        # Store transcript in MongoDB
        await self._store_transcript(
            raw_transcript=raw_transcript,
            cleaned_text=cleaned.cleaned_text,
            course_id=course_id,
        )

        # Chunk
        chunks = self._chunker.chunk(
            cleaned_text=cleaned.cleaned_text,
            segments=cleaned.segments,
            video_id=vid,
            video_title=video_meta.title,
            course_id=course_id,
        )

        if not chunks:
            logger.warning(f"[{vid}] No chunks produced — skipping embedding.")
            await self._update_video_status(video_db_id, ProcessingStatus.COMPLETED)
            return 0

        # Embed + store in ChromaDB
        self._embedder.embed_and_store(chunks)

        # Store chunk metadata in MongoDB
        await self._store_chunks(chunks)

        # Mark video complete
        await self._update_video_status(
            video_db_id,
            ProcessingStatus.COMPLETED,
            transcript_available=True,
        )

        logger.info(f"[{vid}] ✅ Done. {len(chunks)} chunks stored.")
        return len(chunks)

    # ── MongoDB write helpers ─────────────────────────────────────────────────

    async def _create_course(self, playlist_meta, user_id: str) -> str:
        courses = get_courses_collection()
        doc = CourseInDB(
            user_id=user_id,
            playlist_id=playlist_meta.playlist_id,
            playlist_url=playlist_meta.playlist_url,
            title=playlist_meta.title,
            description=playlist_meta.description,
            channel_name=playlist_meta.channel_name,
            thumbnail_url=playlist_meta.thumbnail_url,
            total_videos=playlist_meta.total_videos,
            status=ProcessingStatus.PROCESSING,
        )
        result = await courses.insert_one(
            doc.model_dump(by_alias=False, exclude={"id"})
        )
        return str(result.inserted_id)

    async def _create_videos(self, playlist_meta, course_id: str, user_id: str) -> list[str]:
        videos = get_videos_collection()
        docs = [
            VideoInDB(
                course_id=course_id,
                user_id=user_id,
                video_id=v.video_id,
                title=v.title,
                url=v.url,
                position=v.position,
                duration_seconds=v.duration_seconds,
                thumbnail_url=v.thumbnail_url,
                status=ProcessingStatus.PENDING,
            ).model_dump(by_alias=False, exclude={"id"})
            for v in playlist_meta.videos
        ]
        result = await videos.insert_many(docs)
        return [str(oid) for oid in result.inserted_ids]

    async def _store_transcript(
        self,
        raw_transcript,
        cleaned_text: str,
        course_id: str,
    ) -> None:
        transcripts = get_transcripts_collection()
        segments_data = [
            TranscriptSegmentSchema(
                text=seg.text,
                start=seg.start,
                duration=seg.duration,
            )
            for seg in raw_transcript.segments
        ]
        doc = TranscriptInDB(
            video_id=raw_transcript.video_id,
            course_id=course_id,
            raw_segments=segments_data,
            cleaned_text=cleaned_text,
            language=raw_transcript.language,
            source=raw_transcript.source,
        )
        await transcripts.insert_one(
            doc.model_dump(by_alias=False, exclude={"id"})
        )

    async def _store_chunks(self, chunks) -> None:
        chunks_col = get_chunks_collection()
        docs = [
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
        if docs:
            await chunks_col.insert_many(docs)

    async def _update_course_status(
        self,
        course_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None,
    ) -> None:
        courses = get_courses_collection()
        update: dict = {
            "$set": {
                "status": status.value,
                "updated_at": datetime.utcnow(),
            }
        }
        if error_message:
            update["$set"]["error_message"] = error_message
        await courses.update_one({"_id": ObjectId(course_id)}, update)

    async def _update_video_status(
        self,
        video_db_id: str,
        status: ProcessingStatus,
        transcript_available: bool = False,
    ) -> None:
        videos = get_videos_collection()
        await videos.update_one(
            {"_id": ObjectId(video_db_id)},
            {"$set": {
                "status": status.value,
                "transcript_available": transcript_available,
            }},
        )
