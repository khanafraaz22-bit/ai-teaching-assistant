"""
LLM Chain
---------
Builds a grounded prompt from retrieved chunks and calls GPT-4o.

Design principles:
  1. Grounded — LLM is instructed to answer ONLY from provided context
  2. Source-aware — model is told which video + timestamp each chunk came from
  3. Honest — model explicitly says if the topic isn't covered
  4. Conversational — supports multi-turn chat via message history
  5. Structured — response includes answer + source references separately
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from app.config.settings import settings
from app.services.rag.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str       # "user" | "assistant"
    content: str


@dataclass
class SourceReference:
    video_id: str
    video_title: str
    timestamp_label: str    # "12:34"
    youtube_url: str        # Deep link with &t=...


@dataclass
class ChatResponse:
    answer: str
    sources: list[SourceReference]
    is_grounded: bool           # False if topic wasn't found in the course
    chunks_used: int


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert AI teaching assistant for an online course.
You help students understand the course material by answering their questions
clearly and accurately.

STRICT RULES:
1. Answer ONLY using the course transcript excerpts provided below.
2. If the topic is not covered in the provided excerpts, say:
   "This topic doesn't appear to be covered in this course."
   Do NOT speculate or use outside knowledge.
3. When referencing content, mention the video title and timestamp naturally
   in your answer (e.g., "As explained in 'Video 3 — Backpropagation' at 12:34...").
4. Be clear, educational, and student-friendly.
5. If the question is a follow-up, use the conversation history for context,
   but still ground your answer in the provided excerpts.
6. Format your answer with clear paragraphs. Use bullet points for lists.
   Do not use markdown headers.
"""

_CONTEXT_TEMPLATE = """
--- COURSE TRANSCRIPT EXCERPTS ---
{context_block}
--- END OF EXCERPTS ---

Student question: {question}
"""

_NO_CONTEXT_RESPONSE = (
    "This topic doesn't appear to be covered in this course based on the "
    "available transcript material."
)


# ── LLM Chain ─────────────────────────────────────────────────────────────────

class LLMChain:
    """
    Wraps the OpenAI chat completion API with a grounded RAG prompt.

    Usage:
        chain = LLMChain()
        response = chain.ask(
            question="What is backpropagation?",
            chunks=[...],
            history=[ChatMessage(role="user", content="..."), ...],
        )
    """

    def __init__(self):
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_LLM_MODEL

    def ask(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: Optional[list[ChatMessage]] = None,
    ) -> ChatResponse:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            question:   The user's current question.
            chunks:     Retrieved transcript chunks (from RetrievalService).
            history:    Prior conversation turns for multi-turn support.
        """
        history = history or []

        if not chunks:
            logger.info("No chunks retrieved — returning not-covered response.")
            return ChatResponse(
                answer=_NO_CONTEXT_RESPONSE,
                sources=[],
                is_grounded=False,
                chunks_used=0,
            )

        # Build context block from chunks
        context_block = self._build_context_block(chunks)

        # Build the message list for the API call
        messages = self._build_messages(question, context_block, history)

        # Call GPT-4o
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,        # Low temp = factual, consistent answers
                max_tokens=1500,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"LLM call failed: {e}")

        # Extract unique source references from the chunks that were used
        sources = self._extract_sources(chunks)

        is_grounded = "doesn't appear to be covered" not in answer.lower()

        logger.info(
            f"LLM response generated. "
            f"Chunks used: {len(chunks)} | Sources: {len(sources)} | "
            f"Grounded: {is_grounded}"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            is_grounded=is_grounded,
            chunks_used=len(chunks),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context_block(chunks: list[RetrievedChunk]) -> str:
        """
        Format chunks into a numbered context block.
        Each chunk includes its source video and timestamp.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Excerpt {i}]\n"
                f"Source: {chunk.video_title} — {chunk.timestamp_label}\n"
                f"URL: {chunk.youtube_url}\n"
                f"{chunk.chunk_text.strip()}\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _build_messages(
        question: str,
        context_block: str,
        history: list[ChatMessage],
    ) -> list[dict]:
        """
        Assemble the full message list:
          [system] → [history turns] → [user question with context]
        """
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Inject prior turns (capped at last 6 to stay within context window)
        for turn in history[-6:]:
            messages.append({"role": turn.role, "content": turn.content})

        # Final user message includes the retrieved context
        user_content = _CONTEXT_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )
        messages.append({"role": "user", "content": user_content})

        return messages

    @staticmethod
    def _extract_sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
        """
        Deduplicate and return one SourceReference per unique video+timestamp.
        Sorted by video position (video_id alphabetically as proxy).
        """
        seen: set[str] = set()
        sources: list[SourceReference] = []

        for chunk in chunks:
            key = f"{chunk.video_id}:{chunk.start_timestamp}"
            if key in seen:
                continue
            seen.add(key)
            sources.append(SourceReference(
                video_id=chunk.video_id,
                video_title=chunk.video_title,
                timestamp_label=chunk.timestamp_label,
                youtube_url=chunk.youtube_url,
            ))

        return sources
