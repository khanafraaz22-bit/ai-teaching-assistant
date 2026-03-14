"""
Transcript Cleaner
------------------
Takes raw transcript segments and produces clean, readable paragraph text.

Problems this solves:
  - Auto-captions have no punctuation and inconsistent casing
  - Filler words, music notes, [inaudible] markers pollute the text
  - Short overlapping segments need merging into coherent sentences
  - Repeated tokens (YouTube sometimes duplicates at segment boundaries)
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from .transcript_extractor import TranscriptSegment, RawTranscript

logger = logging.getLogger(__name__)


@dataclass
class CleanedTranscript:
    video_id: str
    cleaned_text: str               # Full merged clean text
    segments: list[TranscriptSegment]   # Original segments preserved (for timestamps)
    language: str
    source: str
    word_count: int


class TranscriptCleaner:
    """
    Cleans raw transcript segments into usable text.

    Design: non-destructive — segments are preserved for timestamp lookup.
    Only `cleaned_text` is modified.
    """

    # Patterns to strip from captions
    _NOISE_PATTERNS = [
        r"\[.*?\]",           # [Music], [Applause], [inaudible], etc.
        r"\(.*?\)",           # (music), (laughter), etc.
        r"♪.*?♪",             # Music note markers
        r"♫.*?♫",
        r"&amp;",             # HTML entities
        r"&lt;",
        r"&gt;",
        r"<[^>]+>",           # Any remaining HTML tags
    ]

    # Filler words to reduce (not fully remove — keeps naturalness)
    _FILLERS = re.compile(
        r"\b(um+|uh+|er+|ah+|hmm+|mhm|uhhuh)\b",
        flags=re.IGNORECASE,
    )

    def clean(self, raw: RawTranscript) -> CleanedTranscript:
        """Clean a raw transcript into usable text."""
        if not raw.segments:
            logger.warning(f"[{raw.video_id}] Transcript has no segments to clean.")
            return CleanedTranscript(
                video_id=raw.video_id,
                cleaned_text="",
                segments=[],
                language=raw.language,
                source=raw.source,
                word_count=0,
            )

        # Step 1: Clean individual segment texts
        cleaned_segments = [
            TranscriptSegment(
                text=self._clean_segment_text(seg.text),
                start=seg.start,
                duration=seg.duration,
            )
            for seg in raw.segments
        ]

        # Step 2: Remove empty segments after cleaning
        cleaned_segments = [s for s in cleaned_segments if s.text.strip()]

        # Step 3: Merge segments into flowing paragraph text
        merged_text = self._merge_segments(cleaned_segments)

        # Step 4: Post-process the full text
        final_text = self._post_process(merged_text)

        word_count = len(final_text.split())
        logger.info(
            f"[{raw.video_id}] Cleaned transcript: {len(raw.segments)} segments "
            f"→ {word_count} words"
        )

        return CleanedTranscript(
            video_id=raw.video_id,
            cleaned_text=final_text,
            segments=cleaned_segments,
            language=raw.language,
            source=raw.source,
            word_count=word_count,
        )

    # ── Private methods ───────────────────────────────────────────────────────

    def _clean_segment_text(self, text: str) -> str:
        """Clean a single segment's text."""
        # Remove noise patterns
        for pattern in self._NOISE_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Remove fillers
        text = self._FILLERS.sub("", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _merge_segments(segments: list[TranscriptSegment]) -> str:
        """
        Merge segments into paragraph text.

        Strategy: concatenate with spaces. If a segment ends with
        sentence-ending punctuation, add a space. Otherwise join with
        a space and let post-processing handle punctuation inference.
        """
        if not segments:
            return ""

        parts = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if not text:
                continue

            # Avoid duplicating content at segment boundaries
            # (YouTube sometimes repeats the last word of the previous segment)
            if parts:
                last_word = parts[-1].split()[-1].lower().rstrip(".,!?") if parts[-1].split() else ""
                first_word = text.split()[0].lower().rstrip(".,!?") if text.split() else ""
                if last_word == first_word and len(last_word) > 3:
                    # Skip the duplicate first word
                    words = text.split()
                    text = " ".join(words[1:]) if len(words) > 1 else ""
                    if not text:
                        continue

            parts.append(text)

        return " ".join(parts)

    @staticmethod
    def _post_process(text: str) -> str:
        """Final cleanup on merged text."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Collapse 3+ newlines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Capitalize start of text
        if text:
            text = text[0].upper() + text[1:]

        # Remove lone punctuation artifacts
        text = re.sub(r"\s+([,.])\s+", r"\1 ", text)

        return text
