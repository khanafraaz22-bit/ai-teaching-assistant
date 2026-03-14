"""
Video Summarizer
----------------
Summarizes a single video lecture or an entire course.

Strategy:
  - Fetch all chunks for the target video (or course) from MongoDB
  - Sort by position (chronological order)
  - Feed to LLM with a structured summarization prompt
  - Return a summary with key points, concepts, and takeaways

For long videos (many chunks), we use a map-reduce approach:
  1. Summarize chunks in groups (map)
  2. Combine partial summaries into a final summary (reduce)
  This avoids hitting the context window limit.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.database.mongodb import get_chunks_collection
from app.services.learning.base import LearningToolsBase

logger = logging.getLogger(__name__)

# Max chunks to send in a single LLM call before switching to map-reduce
_DIRECT_CHUNK_LIMIT = 15


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class SummaryResult:
    title: str
    overview: str
    key_points: list[str]
    concepts_covered: list[str]
    takeaway: str
    video_id: Optional[str]     # None if this is a course-level summary
    course_id: str


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert educational content summarizer.
Your summaries are clear, structured, and student-friendly.
Always respond in the exact JSON format requested. No extra text outside the JSON.
"""

_SUMMARY_PROMPT = """Summarize the following lecture transcript excerpts.

{context}

Respond ONLY with a JSON object in this exact format:
{{
  "overview": "2-3 sentence overview of what this lecture covers",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "concepts_covered": ["concept1", "concept2", "concept3"],
  "takeaway": "The single most important thing to remember from this lecture"
}}"""

_REDUCE_PROMPT = """You have these partial summaries of different sections of a lecture.
Combine them into one coherent final summary.

{partial_summaries}

Respond ONLY with a JSON object in this exact format:
{{
  "overview": "2-3 sentence overview of the full lecture",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "concepts_covered": ["concept1", "concept2", "concept3"],
  "takeaway": "The single most important thing to remember"
}}"""


# ── Service ───────────────────────────────────────────────────────────────────

class SummarizerService(LearningToolsBase):
    """
    Summarizes a video or full course using transcript chunks.

    Usage:
        service = SummarizerService()
        result = await service.summarize_video(video_id="dQw4w9WgXcQ", course_id="abc123")
        result = await service.summarize_course(course_id="abc123")
    """

    async def summarize_video(self, video_id: str, course_id: str) -> SummaryResult:
        """Summarize a single video lecture."""
        chunks = await self._fetch_chunks(course_id=course_id, video_id=video_id)
        if not chunks:
            raise ValueError(f"No transcript chunks found for video: {video_id}")

        video_title = chunks[0].get("video_title", "Unknown Video")
        text = self._build_context(chunks)
        summary_data = self._summarize_text(text, len(chunks))

        logger.info(f"Summarized video: {video_id}")
        return SummaryResult(
            title=video_title,
            course_id=course_id,
            video_id=video_id,
            **summary_data,
        )

    async def summarize_course(self, course_id: str) -> SummaryResult:
        """Summarize an entire course at a high level."""
        chunks = await self._fetch_chunks(course_id=course_id)
        if not chunks:
            raise ValueError(f"No transcript chunks found for course: {course_id}")

        text = self._build_context(chunks)
        summary_data = self._summarize_text(text, len(chunks))

        logger.info(f"Summarized course: {course_id}")
        return SummaryResult(
            title="Full Course Summary",
            course_id=course_id,
            video_id=None,
            **summary_data,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _summarize_text(self, context: str, chunk_count: int) -> dict:
        """Run direct or map-reduce summarization depending on size."""
        if chunk_count <= _DIRECT_CHUNK_LIMIT:
            return self._direct_summarize(context)
        else:
            return self._mapreduce_summarize(context)

    def _direct_summarize(self, context: str) -> dict:
        """Single LLM call for short transcripts."""
        prompt = _SUMMARY_PROMPT.format(context=context)
        raw = self._call_llm(_SYSTEM_PROMPT, prompt, temperature=0.3)
        return self._parse_json(raw)

    def _mapreduce_summarize(self, context: str) -> dict:
        """
        Map-reduce for long transcripts.
        Split context into segments → summarize each → combine.
        """
        # Split context into segments of ~3000 chars each
        segment_size = 3000
        segments = [
            context[i: i + segment_size]
            for i in range(0, len(context), segment_size)
        ]

        # Map: summarize each segment
        partial_summaries = []
        for i, segment in enumerate(segments):
            prompt = _SUMMARY_PROMPT.format(context=segment)
            raw = self._call_llm(_SYSTEM_PROMPT, prompt, temperature=0.3)
            partial_summaries.append(f"Section {i+1}:\n{raw}")
            logger.info(f"Map step {i+1}/{len(segments)} complete")

        # Reduce: combine partial summaries
        combined = "\n\n".join(partial_summaries)
        prompt = _REDUCE_PROMPT.format(partial_summaries=combined)
        raw = self._call_llm(_SYSTEM_PROMPT, prompt, temperature=0.3)
        return self._parse_json(raw)

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        """Build a readable context string from sorted chunks."""
        return "\n\n".join(
            f"[{c.get('video_title', '')} — {_fmt_ts(c.get('start_timestamp', 0))}]\n"
            f"{c.get('chunk_text', '')}"
            for c in chunks
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse JSON from LLM response, with fallback."""
        import json
        try:
            # Strip markdown code fences if present
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return {
                "overview":          data.get("overview", ""),
                "key_points":        data.get("key_points", []),
                "concepts_covered":  data.get("concepts_covered", []),
                "takeaway":          data.get("takeaway", ""),
            }
        except Exception as e:
            logger.error(f"JSON parse error in summarizer: {e}\nRaw: {raw[:200]}")
            return {
                "overview":         raw[:500],
                "key_points":       [],
                "concepts_covered": [],
                "takeaway":         "",
            }

    @staticmethod
    async def _fetch_chunks(
        course_id: str,
        video_id: Optional[str] = None,
    ) -> list[dict]:
        """Fetch chunks from MongoDB, sorted by video position."""
        col = get_chunks_collection()
        query = {"course_id": course_id}
        if video_id:
            query["video_id"] = video_id
        cursor = col.find(query).sort([("video_id", 1), ("position", 1)])
        return [doc async for doc in cursor]


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"
