"""
Retrieval Service
-----------------
Converts a user query into an embedding, runs cosine similarity search
against ChromaDB, and returns the most relevant transcript chunks.

Supports:
  - Course-scoped search (filter by course_id)
  - Video-scoped search (filter by video_id)
  - Cross-video retrieval (default — searches the whole course)
  - Deduplication of near-identical chunks
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.config.settings import settings
from app.database.vectordb import get_collection
from app.services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_text: str
    video_id: str
    video_title: str
    start_timestamp: float
    end_timestamp: float
    course_id: str
    relevance_score: float      # 0.0 (irrelevant) → 1.0 (identical)
    position: int

    @property
    def timestamp_label(self) -> str:
        """Human-readable timestamp, e.g. '12:34'"""
        return _seconds_to_label(self.start_timestamp)

    @property
    def youtube_url(self) -> str:
        """Deep link to the exact moment in the video."""
        t = int(self.start_timestamp)
        return f"https://www.youtube.com/watch?v={self.video_id}&t={t}s"


def _seconds_to_label(seconds: float) -> str:
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ── Retrieval service ─────────────────────────────────────────────────────────

class RetrievalService:
    """
    Semantic search over stored transcript chunks.

    Usage:
        retriever = RetrievalService()
        chunks = retriever.retrieve(
            query="What is gradient descent?",
            course_id="abc123",
            top_k=6,
        )
    """

    def __init__(self):
        self._embedder = EmbeddingService()

    def retrieve(
        self,
        query: str,
        course_id: str,
        top_k: int = None,
        video_id: Optional[str] = None,
        min_relevance: float = 0.25,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:          The user's question or search string.
            course_id:      Scope results to this course.
            top_k:          Number of results to return (default from settings).
            video_id:       Optional — restrict search to a single video.
            min_relevance:  Discard chunks below this cosine similarity score.
        """
        top_k = top_k or settings.RAG_TOP_K

        # 1. Embed the query
        query_embedding = self._embedder.embed_query(query)

        # 2. Build ChromaDB filter
        where_filter = self._build_filter(course_id, video_id)

        # 3. Query ChromaDB
        collection = get_collection()
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 20),   # Over-fetch, then filter + dedupe
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return []

        # 4. Parse results
        chunks = self._parse_results(results, min_relevance)

        # 5. Deduplicate and trim
        chunks = self._deduplicate(chunks)
        chunks = chunks[:top_k]

        logger.info(
            f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...' "
            f"(course={course_id})"
        )
        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(course_id: str, video_id: Optional[str]) -> dict:
        """Build a ChromaDB `where` filter dict."""
        if video_id:
            return {
                "$and": [
                    {"course_id": {"$eq": course_id}},
                    {"video_id":  {"$eq": video_id}},
                ]
            }
        return {"course_id": {"$eq": course_id}}

    @staticmethod
    def _parse_results(results: dict, min_relevance: float) -> list[RetrievedChunk]:
        """Convert raw ChromaDB results into RetrievedChunk objects."""
        chunks = []

        ids       = (results.get("ids")       or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance → similarity score (1 = identical)
            score = 1.0 - distance

            if score < min_relevance:
                continue

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                chunk_text=text,
                video_id=meta.get("video_id", ""),
                video_title=meta.get("video_title", "Unknown Video"),
                start_timestamp=float(meta.get("start_timestamp", 0)),
                end_timestamp=float(meta.get("end_timestamp", 0)),
                course_id=meta.get("course_id", ""),
                relevance_score=round(score, 4),
                position=int(meta.get("position", 0)),
            ))

        # Sort by relevance descending
        chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        return chunks

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Remove chunks that are too similar to a higher-ranked chunk.
        Uses simple text overlap (Jaccard similarity on word sets).
        """
        seen: list[RetrievedChunk] = []

        for chunk in chunks:
            words_a = set(chunk.chunk_text.lower().split())
            is_duplicate = False

            for existing in seen:
                words_b = set(existing.chunk_text.lower().split())
                union = words_a | words_b
                if not union:
                    continue
                jaccard = len(words_a & words_b) / len(union)
                if jaccard > 0.80:      # 80%+ word overlap → duplicate
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen.append(chunk)

        return seen
