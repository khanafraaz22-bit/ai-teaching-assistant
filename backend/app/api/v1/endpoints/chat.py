"""
Chat endpoints — RAG-powered Q&A over course transcripts.

POST /chat/ask           — Ask a question about a course
DELETE /chat/session     — Clear conversation history for a session
"""

from __future__ import annotations


import uuid
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Optional

from app.services.rag.chatbot import RAGChatbot
from app.core.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Single shared chatbot instance (stateless except for in-memory sessions)
_chatbot = RAGChatbot()


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    course_id: str
    session_id: Optional[str] = None    # Auto-generated if not provided
    video_id: Optional[str] = None      # Restrict search to one video
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceRef(BaseModel):
    video_id: str
    video_title: str
    timestamp: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    is_grounded: bool
    chunks_used: int
    session_id: str


class ClearSessionRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse, summary="Ask a question about a course")
async def ask_question(
    payload: AskRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    RAG-powered Q&A over a course's transcript material.

    - Pass the same `session_id` across requests to enable multi-turn conversation.
    - If `session_id` is omitted, a new session is created and returned.
    - Optionally pass `video_id` to restrict the search to a single video.

    Example request:
    ```json
    {
      "question": "What is gradient descent?",
      "course_id": "64f1a2b3c4d5e6f7a8b9c0d1",
      "session_id": "user-abc-session-1"
    }
    ```
    """
    # Auto-generate a session ID for new conversations
    session_id = payload.session_id or str(uuid.uuid4())

    try:
        result = _chatbot.ask(
            question=payload.question,
            course_id=payload.course_id,
            session_id=session_id,
            video_id=payload.video_id,
            top_k=payload.top_k,
        )
    except RuntimeError as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable. Please try again.",
        )
    except Exception as e:
        logger.error(f"Unexpected chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )

    # Auto-record the topic studied for progress tracking
    try:
        from app.services.learning.progress_service import ProgressService
        await ProgressService().record_topic_studied(
            user_id=str(current_user["_id"]),
            course_id=payload.course_id,
            topic=payload.question,
        )
    except Exception as e:
        logger.warning(f"Progress tracking failed (non-fatal): {e}")

    return AskResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        is_grounded=result.is_grounded,
        chunks_used=result.chunks_used,
        session_id=result.session_id,
    )


@router.delete("/session", summary="Clear conversation history", status_code=status.HTTP_200_OK)
async def clear_session(
    payload: ClearSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Reset the conversation history so the next question starts fresh."""
    _chatbot.clear_session(payload.session_id)
    return {"message": f"Session '{payload.session_id}' cleared."}