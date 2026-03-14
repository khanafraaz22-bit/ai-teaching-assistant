"""
Transcript Chunker
------------------
Splits a cleaned transcript into overlapping chunks suitable for embedding.

Design goals:
  1. Respect token budget (500–800 tokens per chunk)
  2. Overlap 100 tokens between chunks to avoid cutting mid-concept
  3. Preserve timestamp references so every chunk maps back to a video moment
  4. Split on sentence boundaries where possible (not mid-sentence)

Each output chunk carries full metadata for vector store + MongoDB storage.
"""

import uuid
import logging
import re
from dataclasses import dataclass, field

import tiktoken

from app.config.settings import settings
from .transcript_extractor import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class TranscriptChunk:
    chunk_id: str                    # UUID — shared between MongoDB and ChromaDB
    video_id: str
    video_title: str
    course_id: str
    chunk_text: str
    start_timestamp: float           # seconds
    end_timestamp: float             # seconds
    token_count: int
    position: int                    # Index of this chunk within the video


class TranscriptChunker:
    """
    Splits transcript text into overlapping chunks with timestamp metadata.

    Timestamps are resolved by mapping text positions back to the original
    TranscriptSegment list (which carries start/end times).
    """

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
        self.overlap = overlap or settings.CHUNK_OVERLAP_TOKENS

        # Use the same tokenizer as OpenAI embeddings
        self._encoder = tiktoken.get_encoding("cl100k_base")

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        cleaned_text: str,
        segments: list[TranscriptSegment],
        video_id: str,
        video_title: str,
        course_id: str,
    ) -> list[TranscriptChunk]:
        """
        Split transcript text into chunks.

        Args:
            cleaned_text:  Full cleaned transcript string.
            segments:      Original timed segments for timestamp resolution.
            video_id:      YouTube video ID.
            video_title:   Human-readable video title (stored in chunk metadata).
            course_id:     MongoDB course document ID.
        """
        if not cleaned_text.strip():
            logger.warning(f"[{video_id}] Empty transcript — no chunks produced.")
            return []

        sentences = self._split_sentences(cleaned_text)
        if not sentences:
            return []

        # Build a timeline map: character offset → timestamp
        timeline = self._build_timeline(cleaned_text, segments)

        raw_chunks = self._build_chunks(sentences)

        result: list[TranscriptChunk] = []
        char_offset = 0  # Running character position in the full text

        for position, (chunk_text, token_count) in enumerate(raw_chunks):
            start_ts, end_ts = self._resolve_timestamps(
                chunk_text, char_offset, cleaned_text, timeline
            )
            result.append(TranscriptChunk(
                chunk_id=str(uuid.uuid4()),
                video_id=video_id,
                video_title=video_title,
                course_id=course_id,
                chunk_text=chunk_text,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                token_count=token_count,
                position=position,
            ))
            # Advance offset (approximate — overlap means we step back)
            char_offset += max(1, len(chunk_text) - self._tokens_to_chars(self.overlap))

        logger.info(
            f"[{video_id}] Chunked into {len(result)} chunks "
            f"(size≈{self.chunk_size} tokens, overlap={self.overlap})"
        )
        return result

    # ── Sentence splitting ────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences at natural boundaries.
        Handles abbreviations and decimal numbers gracefully.
        """
        # Simple but effective: split on .!? followed by whitespace + capital
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        # Filter out empty strings
        return [s.strip() for s in sentences if s.strip()]

    # ── Chunk building ────────────────────────────────────────────────────────

    def _build_chunks(self, sentences: list[str]) -> list[tuple[str, int]]:
        """
        Greedily accumulate sentences into chunks, then slide with overlap.
        Returns list of (chunk_text, token_count).
        """
        # Tokenize each sentence once
        tokenized = [(s, len(self._encoder.encode(s))) for s in sentences]

        chunks: list[tuple[str, int]] = []
        start_idx = 0

        while start_idx < len(tokenized):
            # Accumulate sentences until we hit chunk_size
            current_tokens = 0
            end_idx = start_idx

            while end_idx < len(tokenized):
                _, tokens = tokenized[end_idx]
                if current_tokens + tokens > self.chunk_size and end_idx > start_idx:
                    break
                current_tokens += tokens
                end_idx += 1

            chunk_sentences = [s for s, _ in tokenized[start_idx:end_idx]]
            chunk_text = " ".join(chunk_sentences)
            chunks.append((chunk_text, current_tokens))

            if end_idx >= len(tokenized):
                break

            # Slide back by overlap tokens to create the overlap window
            overlap_tokens = 0
            new_start = end_idx - 1
            while new_start > start_idx:
                _, tokens = tokenized[new_start]
                if overlap_tokens + tokens > self.overlap:
                    break
                overlap_tokens += tokens
                new_start -= 1

            start_idx = max(start_idx + 1, new_start)

        return chunks

    # ── Timestamp resolution ──────────────────────────────────────────────────

    @staticmethod
    def _build_timeline(full_text: str, segments: list[TranscriptSegment]) -> list[tuple[int, float, float]]:
        """
        Build a list of (char_offset, start_sec, end_sec) tuples by
        finding each segment's text inside the full merged string.

        This lets us map any character position to a video timestamp.
        """
        timeline = []
        search_start = 0

        for seg in segments:
            seg_text = seg.text.strip()
            if not seg_text:
                continue
            idx = full_text.find(seg_text, search_start)
            if idx == -1:
                continue
            timeline.append((idx, seg.start, seg.end))
            search_start = idx + len(seg_text)

        return timeline

    @staticmethod
    def _resolve_timestamps(
        chunk_text: str,
        char_offset: int,
        full_text: str,
        timeline: list[tuple[int, float, float]],
    ) -> tuple[float, float]:
        """
        Given a chunk's approximate character offset in the full text,
        find the matching start and end timestamps.
        """
        if not timeline:
            return 0.0, 0.0

        chunk_start_char = full_text.find(chunk_text[:50]) if chunk_text else char_offset
        chunk_end_char = chunk_start_char + len(chunk_text)

        # Find closest segment for start
        start_ts = timeline[0][1]
        end_ts = timeline[-1][2]

        for (offset, seg_start, seg_end) in timeline:
            if offset <= chunk_start_char:
                start_ts = seg_start
            if offset <= chunk_end_char:
                end_ts = seg_end

        return start_ts, end_ts

    def _tokens_to_chars(self, token_count: int) -> int:
        """Rough conversion: avg English token ≈ 4 characters."""
        return token_count * 4

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))
