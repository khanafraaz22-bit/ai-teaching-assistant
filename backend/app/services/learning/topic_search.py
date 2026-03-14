"""
Topic Search Service
--------------------
Answers: "Where is [topic] explained in this course?"

Uses the RAG retrieval pipeline (same as the chatbot) but formats the
result as a list of located references rather than a conversational answer.

Also identifies the best (most complete) explanation among results.
"""

import logging
from dataclasses import dataclass

from app.config.settings import settings
from app.services.rag.retrieval_service import RetrievalService, RetrievedChunk
from app.services.learning.base import LearningToolsBase

logger = logging.getLogger(__name__)


# ── Output containers ─────────────────────────────────────────────────────────

@dataclass
class TopicOccurrence:
    video_id: str
    video_title: str
    timestamp: str          # "12:34"
    youtube_url: str
    relevance_score: float
    excerpt: str            # Short preview of what's said at this moment


@dataclass
class TopicSearchResult:
    topic: str
    found: bool
    occurrences: list[TopicOccurrence]
    best_explanation: str   # LLM-generated synthesis of the best occurrence
    total_found: int


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a helpful teaching assistant.
Answer only from the provided transcript excerpts. Be concise and clear."""

_BEST_EXPLANATION_PROMPT = """Based on these transcript excerpts about "{topic}",
write a clear 2-3 sentence explanation of the topic as it is taught in this course.
Use only the information in the excerpts below.

{context}

If the excerpts don't clearly explain the topic, say:
"This topic is mentioned briefly but not explained in depth in this course."
"""


# ── Service ───────────────────────────────────────────────────────────────────

class TopicSearchService(LearningToolsBase):
    """
    Locates where a topic is covered across all videos in a course.

    Usage:
        service = TopicSearchService()
        result = service.search(topic="backpropagation", course_id="abc123")
    """

    def __init__(self):
        super().__init__()
        self._retriever = RetrievalService()

    def search(
        self,
        topic: str,
        course_id: str,
        top_k: int = 8,
    ) -> TopicSearchResult:
        """
        Find all occurrences of a topic in the course.

        Args:
            topic:      The concept or term to search for.
            course_id:  Which course to search.
            top_k:      Max number of occurrences to return.
        """
        logger.info(f"Topic search: '{topic}' in course {course_id}")

        # Retrieve semantically relevant chunks
        chunks = self._retriever.retrieve(
            query=topic,
            course_id=course_id,
            top_k=top_k,
            min_relevance=0.30,
        )

        if not chunks:
            return TopicSearchResult(
                topic=topic,
                found=False,
                occurrences=[],
                best_explanation=f"'{topic}' does not appear to be covered in this course.",
                total_found=0,
            )

        # Build occurrence list
        occurrences = [
            TopicOccurrence(
                video_id=c.video_id,
                video_title=c.video_title,
                timestamp=c.timestamp_label,
                youtube_url=c.youtube_url,
                relevance_score=c.relevance_score,
                excerpt=self._make_excerpt(c.chunk_text),
            )
            for c in chunks
        ]

        # Generate best explanation from top 3 chunks
        best_explanation = self._generate_best_explanation(topic, chunks[:3])

        logger.info(f"Topic '{topic}' found in {len(occurrences)} locations.")
        return TopicSearchResult(
            topic=topic,
            found=True,
            occurrences=occurrences,
            best_explanation=best_explanation,
            total_found=len(occurrences),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_best_explanation(
        self,
        topic: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Ask the LLM to synthesize the best explanation from top chunks."""
        context = "\n\n".join(
            f"[{c.video_title} — {c.timestamp_label}]\n{c.chunk_text}"
            for c in chunks
        )
        prompt = _BEST_EXPLANATION_PROMPT.format(topic=topic, context=context)
        return self._call_llm(_SYSTEM_PROMPT, prompt, temperature=0.3, max_tokens=300)

    @staticmethod
    def _make_excerpt(text: str, max_chars: int = 200) -> str:
        """Return a short preview of a chunk."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
