"""
Embedding Service
-----------------
Converts text chunks into vector embeddings using OpenAI's
text-embedding-3-small model and stores them in ChromaDB.

Design:
  - Batched API calls (OpenAI supports up to 2048 inputs per request)
  - Idempotent upserts (re-running won't duplicate chunks)
  - Metadata stored alongside embeddings for filtered retrieval
"""

import logging
from typing import Optional

from openai import OpenAI

from app.config.settings import settings
from app.database.vectordb import get_collection
from app.services.youtube.chunker import TranscriptChunk

logger = logging.getLogger(__name__)

# OpenAI embedding batch limit
_BATCH_SIZE = 100


class EmbeddingService:
    """
    Generates embeddings for transcript chunks and stores them in ChromaDB.

    Usage:
        service = EmbeddingService()
        service.embed_and_store(chunks)
    """

    def __init__(self):
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_EMBEDDING_MODEL

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_and_store(self, chunks: list[TranscriptChunk]) -> int:
        """
        Embed all chunks and upsert into ChromaDB.
        Returns the number of chunks successfully stored.
        """
        if not chunks:
            return 0

        collection = get_collection()
        total_stored = 0

        # Process in batches to stay within API limits
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start: batch_start + _BATCH_SIZE]

            try:
                embeddings = self._generate_embeddings([c.chunk_text for c in batch])
            except Exception as e:
                logger.error(f"Embedding API error on batch {batch_start}: {e}")
                continue

            ids = [c.chunk_id for c in batch]
            metadatas = [self._build_metadata(c) for c in batch]
            documents = [c.chunk_text for c in batch]

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                total_stored += len(batch)
                logger.info(
                    f"Stored batch {batch_start}–{batch_start + len(batch)} "
                    f"({len(batch)} chunks)"
                )
            except Exception as e:
                logger.error(f"ChromaDB upsert error on batch {batch_start}: {e}")

        logger.info(f"✅ Embedded and stored {total_stored}/{len(chunks)} chunks.")
        return total_stored

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query string.
        Used at retrieval time.
        """
        result = self._client.embeddings.create(
            model=self._model,
            input=[query],
        )
        return result.data[0].embedding

    def delete_course_chunks(self, course_id: str) -> None:
        """Remove all chunks for a course (used when re-processing)."""
        collection = get_collection()
        collection.delete(where={"course_id": course_id})
        logger.info(f"Deleted all ChromaDB chunks for course: {course_id}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API for a batch of texts."""
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        # Sort by index to guarantee order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    @staticmethod
    def _build_metadata(chunk: TranscriptChunk) -> dict:
        """
        Build the ChromaDB metadata dict for a chunk.
        Only scalar types (str, int, float, bool) are allowed.
        """
        return {
            "course_id":       chunk.course_id,
            "video_id":        chunk.video_id,
            "video_title":     chunk.video_title,
            "start_timestamp": chunk.start_timestamp,
            "end_timestamp":   chunk.end_timestamp,
            "token_count":     chunk.token_count,
            "position":        chunk.position,
        }