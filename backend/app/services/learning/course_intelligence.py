"""
Course Intelligence Service
---------------------------
Two features:

1. Gap Detection
   Analyzes the full course mind map and identifies important topics
   that are missing or only briefly mentioned — not deeply covered.

2. Resource Recommendations
   For each gap found, suggests external learning resources:
   - YouTube videos / playlists
   - Documentation / official guides
   - Articles and tutorials

Both features use the LLM with the course's actual content as context,
so gaps and recommendations are specific to THIS course, not generic.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.database.mongodb import get_chunks_collection, get_courses_collection
from app.services.learning.base import LearningToolsBase
from bson import ObjectId

logger = logging.getLogger(__name__)


# ── Output containers ─────────────────────────────────────────────────────────

@dataclass
class MissingTopic:
    topic: str
    why_important: str          # Why this belongs in the course
    how_related: str            # How it connects to what IS covered
    difficulty: str             # "foundational" | "intermediate" | "advanced"


@dataclass
class ResourceRecommendation:
    topic: str
    resource_type: str          # "youtube" | "documentation" | "article" | "course"
    title: str
    description: str
    search_query: str           # Ready-to-use search query for this resource


@dataclass
class CourseIntelligenceResult:
    course_id: str
    course_title: str
    topics_covered: list[str]
    missing_topics: list[MissingTopic]
    recommendations: list[ResourceRecommendation]
    coverage_summary: str       # One-paragraph assessment of the course's coverage


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert curriculum designer and educator.
You analyze course content and identify knowledge gaps with precision.
Always respond in valid JSON only. No markdown, no extra text."""

_GAP_DETECTION_PROMPT = """You are reviewing a course on the following subject.

Course title: {course_title}

Topics actually covered in this course (extracted from transcripts):
{topics_covered}

Sample transcript content:
{sample_content}

Task:
1. Identify 4-6 important topics that are MISSING from this course or only
   very briefly mentioned without proper explanation.
2. These should be topics that a student would reasonably expect to find
   in a course on this subject.
3. Assess the overall coverage of the course.

Respond ONLY with this JSON:
{{
  "coverage_summary": "One paragraph assessing what the course covers well and what it lacks",
  "missing_topics": [
    {{
      "topic": "Topic name",
      "why_important": "Why a student of this subject needs to know this",
      "how_related": "How this connects to what is already covered in the course",
      "difficulty": "foundational | intermediate | advanced"
    }}
  ]
}}

Rules:
- Only suggest topics genuinely relevant to the course subject
- Be specific — not just 'more examples' but actual missing concepts
- Order by importance (most critical gaps first)
"""

_RECOMMENDATIONS_PROMPT = """For each missing topic below, suggest the best
external learning resource a student could use to fill this gap.

Course subject: {course_title}
Missing topics:
{missing_topics}

Respond ONLY with this JSON:
{{
  "recommendations": [
    {{
      "topic": "Exact topic name from the list above",
      "resource_type": "youtube | documentation | article | course",
      "title": "Specific resource title or description",
      "description": "1-2 sentences on what the student will learn from this",
      "search_query": "Exact search query to find this resource on Google or YouTube"
    }}
  ]
}}

Rules:
- One recommendation per missing topic
- Prefer official documentation and well-known educational channels
- Search queries should be specific enough to find good results
- For programming topics: prefer official docs or well-known channels
- For math/science: prefer Khan Academy, MIT OpenCourseWare, 3Blue1Brown style
"""


# ── Service ───────────────────────────────────────────────────────────────────

class CourseIntelligenceService(LearningToolsBase):
    """
    Analyzes a course for coverage gaps and recommends resources.

    Usage:
        service = CourseIntelligenceService()
        result = await service.analyze(course_id="abc123")
    """

    async def analyze(self, course_id: str) -> CourseIntelligenceResult:
        """
        Full intelligence analysis for a course.
        Runs gap detection then resource recommendation in sequence.
        """
        # Fetch course title
        course_title = await self._get_course_title(course_id)

        # Fetch a sample of chunks for context
        chunks = await self._fetch_chunks(course_id, sample_size=25)
        if not chunks:
            raise ValueError(f"No transcript data found for course: {course_id}")

        # Extract topics already covered (from chunk texts via LLM)
        topics_covered = await self._extract_covered_topics(chunks, course_title)

        # Step 1: Detect gaps
        sample_content = self._build_sample_content(chunks)
        gaps_raw = self._call_llm(
            _SYSTEM_PROMPT,
            _GAP_DETECTION_PROMPT.format(
                course_title=course_title,
                topics_covered="\n".join(f"- {t}" for t in topics_covered),
                sample_content=sample_content,
            ),
            temperature=0.4,
            max_tokens=2000,
        )
        gaps_data = self._parse_json(gaps_raw)
        missing_topics = self._parse_missing_topics(gaps_data)
        coverage_summary = gaps_data.get("coverage_summary", "")

        if not missing_topics:
            logger.info(f"No gaps detected for course: {course_id}")
            return CourseIntelligenceResult(
                course_id=course_id,
                course_title=course_title,
                topics_covered=topics_covered,
                missing_topics=[],
                recommendations=[],
                coverage_summary=coverage_summary,
            )

        # Step 2: Recommend resources for each gap
        missing_list = "\n".join(
            f"- {t.topic}: {t.why_important}" for t in missing_topics
        )
        recs_raw = self._call_llm(
            _SYSTEM_PROMPT,
            _RECOMMENDATIONS_PROMPT.format(
                course_title=course_title,
                missing_topics=missing_list,
            ),
            temperature=0.4,
            max_tokens=2000,
        )
        recs_data = self._parse_json(recs_raw)
        recommendations = self._parse_recommendations(recs_data)

        logger.info(
            f"Course intelligence complete: course={course_id} | "
            f"gaps={len(missing_topics)} | recommendations={len(recommendations)}"
        )

        return CourseIntelligenceResult(
            course_id=course_id,
            course_title=course_title,
            topics_covered=topics_covered,
            missing_topics=missing_topics,
            recommendations=recommendations,
            coverage_summary=coverage_summary,
        )

    # ── Topic extraction ──────────────────────────────────────────────────────

    async def _extract_covered_topics(
        self,
        chunks: list[dict],
        course_title: str,
    ) -> list[str]:
        """Extract the list of topics the course actually covers."""
        sample = self._build_sample_content(chunks[:15])
        prompt = f"""List all distinct topics and concepts covered in this course.
Course: {course_title}

Transcript sample:
{sample}

Respond ONLY with a JSON array of topic strings:
["topic1", "topic2", "topic3", ...]
"""
        raw = self._call_llm(_SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=800)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            topics = json.loads(clean)
            return topics if isinstance(topics, list) else []
        except Exception:
            return []

    # ── Parsers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_missing_topics(data: dict) -> list[MissingTopic]:
        topics = []
        for item in data.get("missing_topics", []):
            topics.append(MissingTopic(
                topic=item.get("topic", ""),
                why_important=item.get("why_important", ""),
                how_related=item.get("how_related", ""),
                difficulty=item.get("difficulty", "intermediate"),
            ))
        return topics

    @staticmethod
    def _parse_recommendations(data: dict) -> list[ResourceRecommendation]:
        recs = []
        for item in data.get("recommendations", []):
            recs.append(ResourceRecommendation(
                topic=item.get("topic", ""),
                resource_type=item.get("resource_type", "article"),
                title=item.get("title", ""),
                description=item.get("description", ""),
                search_query=item.get("search_query", ""),
            ))
        return recs

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception as e:
            logger.error(f"JSON parse error in intelligence service: {e}")
            return {}

    # ── Data helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_sample_content(chunks: list[dict]) -> str:
        return "\n\n".join(
            f"[{c.get('video_title', '')}]\n{c.get('chunk_text', '')[:400]}"
            for c in chunks
            if c.get("chunk_text")
        )

    @staticmethod
    async def _fetch_chunks(course_id: str, sample_size: int = 25) -> list[dict]:
        col = get_chunks_collection()
        total = await col.count_documents({"course_id": course_id})
        if total == 0:
            return []

        # Sample evenly across the course
        step = max(1, total // sample_size)
        cursor = col.find({"course_id": course_id}).sort("position", 1)
        all_chunks = [doc async for doc in cursor]
        return all_chunks[::step][:sample_size]

    @staticmethod
    async def _get_course_title(course_id: str) -> str:
        col = get_courses_collection()
        try:
            doc = await col.find_one({"_id": ObjectId(course_id)})
            return doc.get("title", "Unknown Course") if doc else "Unknown Course"
        except Exception:
            return "Unknown Course"
