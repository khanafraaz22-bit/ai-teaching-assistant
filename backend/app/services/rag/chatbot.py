"""
RAG Chatbot Service
-------------------
The single entry point for all Q&A interactions.
Wires together RetrievalService and LLMChain.

Also owns conversation history management — stored in-memory per session
for now. Phase 5 will move this to MongoDB for persistence.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import settings
from app.services.rag.retrieval_service import RetrievalService
from app.services.rag.llm_chain import LLMChain, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


# ── In-memory session store ───────────────────────────────────────────────────
# Key: session_id (str) → list of ChatMessage
# Replaced by MongoDB-backed sessions in Phase 5
_sessions: dict[str, list[ChatMessage]] = {}


@dataclass
class AskResult:
    """Full result returned to the API layer."""
    answer: str
    sources: list[dict]         # Serializable source dicts for JSON response
    is_grounded: bool
    chunks_used: int
    session_id: str


class RAGChatbot:
    """
    Course-aware AI chatbot.

    Usage:
        bot = RAGChatbot()
        result = bot.ask(
            question="Explain backpropagation",
            course_id="abc123",
            session_id="user-xyz-session-1",
        )
    """

    def __init__(self):
        self._retriever = RetrievalService()
        self._chain = LLMChain()

    def ask(
        self,
        question: str,
        course_id: str,
        session_id: str,
        video_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> AskResult:
        """
        Answer a question about the course using RAG.

        Args:
            question:   The user's question.
            course_id:  Which course to search.
            session_id: Conversation session (for multi-turn context).
            video_id:   Optional — restrict search to one video.
            top_k:      Override default number of chunks to retrieve.
        """
        logger.info(
            f"[session={session_id}] Question: '{question[:80]}' "
            f"(course={course_id})"
        )

        # 1. Load conversation history for this session
        history = _sessions.get(session_id, [])

        # 2. Retrieve relevant chunks
        chunks = self._retriever.retrieve(
            query=question,
            course_id=course_id,
            video_id=video_id,
            top_k=top_k or settings.RAG_TOP_K,
        )

        # 3. Generate answer
        response: ChatResponse = self._chain.ask(
            question=question,
            chunks=chunks,
            history=history,
        )

        # 4. Update conversation history
        self._update_history(session_id, question, response.answer)

        # 5. Serialize sources for JSON response
        sources = [
            {
                "video_id":        src.video_id,
                "video_title":     src.video_title,
                "timestamp":       src.timestamp_label,
                "url":             src.youtube_url,
            }
            for src in response.sources
        ]

        return AskResult(
            answer=response.answer,
            sources=sources,
            is_grounded=response.is_grounded,
            chunks_used=response.chunks_used,
            session_id=session_id,
        )

    def clear_session(self, session_id: str) -> None:
        """Reset conversation history for a session."""
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _update_history(session_id: str, question: str, answer: str) -> None:
        """Append the latest Q&A turn to the session history."""
        if session_id not in _sessions:
            _sessions[session_id] = []

        _sessions[session_id].extend([
            ChatMessage(role="user",      content=question),
            ChatMessage(role="assistant", content=answer),
        ])

        # Cap history to last 10 turns (20 messages) to avoid unbounded growth
        if len(_sessions[session_id]) > 20:
            _sessions[session_id] = _sessions[session_id][-20:]
