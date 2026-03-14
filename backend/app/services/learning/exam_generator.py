"""
Exam Generator
--------------
Generates a quiz from course or video transcript material.

Output:
  - 5 Multiple Choice Questions (4 options each, 1 correct)
  - 2 Short Answer Questions
  - 1 Long Answer / Essay Question

All questions are grounded in the actual transcript content.
Difficulty is configurable: easy / medium / hard.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.database.mongodb import get_chunks_collection
from app.models.schemas import QuestionType
from app.services.learning.base import LearningToolsBase

logger = logging.getLogger(__name__)


# ── Output containers ─────────────────────────────────────────────────────────

@dataclass
class ExamQuestion:
    question_type: QuestionType
    question: str
    options: Optional[list[str]]    # MCQ only — 4 options
    correct_answer: str
    explanation: str                # Why this is the correct answer


@dataclass
class ExamResult:
    title: str
    difficulty: str
    mcq_questions: list[ExamQuestion]
    short_answer_questions: list[ExamQuestion]
    long_answer_questions: list[ExamQuestion]
    video_id: Optional[str]
    course_id: str

    @property
    def all_questions(self) -> list[ExamQuestion]:
        return self.mcq_questions + self.short_answer_questions + self.long_answer_questions

    @property
    def total_questions(self) -> int:
        return len(self.all_questions)


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert educator creating exam questions.
Questions must be based strictly on the provided transcript content.
Always respond in valid JSON only. No markdown fences, no extra text."""

_EXAM_PROMPT = """Create an exam based on these lecture transcript excerpts.
Difficulty level: {difficulty}

{context}

Generate exactly this structure in JSON:
{{
  "mcq": [
    {{
      "question": "Question text?",
      "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct_answer": "A) option1",
      "explanation": "Why this is correct"
    }}
  ],
  "short_answer": [
    {{
      "question": "Question text?",
      "correct_answer": "Expected answer",
      "explanation": "Key points the answer should cover"
    }}
  ],
  "long_answer": [
    {{
      "question": "Essay question text?",
      "correct_answer": "Comprehensive expected answer",
      "explanation": "Key themes and concepts the answer should address"
    }}
  ]
}}

Rules:
- Generate exactly 5 MCQ questions
- Generate exactly 2 short answer questions
- Generate exactly 1 long answer question
- All questions must be answerable from the transcript content only
- {difficulty_instructions}
"""

_DIFFICULTY_INSTRUCTIONS = {
    "easy":   "Focus on definitions, basic facts, and recall questions.",
    "medium": "Mix of recall and application. Some questions require connecting concepts.",
    "hard":   "Focus on analysis, comparison, and deeper understanding. Avoid simple recall.",
}


# ── Service ───────────────────────────────────────────────────────────────────

class ExamGeneratorService(LearningToolsBase):
    """
    Generates exams from transcript content.

    Usage:
        service = ExamGeneratorService()
        result = await service.generate(
            course_id="abc123",
            video_id="xyz",       # Optional — omit for full-course exam
            difficulty="medium",
        )
    """

    async def generate(
        self,
        course_id: str,
        video_id: Optional[str] = None,
        difficulty: str = "medium",
    ) -> ExamResult:
        """
        Generate an exam for a course or single video.

        Args:
            course_id:   The course to examine.
            video_id:    Optional — restrict to a single video.
            difficulty:  "easy" | "medium" | "hard"
        """
        if difficulty not in _DIFFICULTY_INSTRUCTIONS:
            difficulty = "medium"

        chunks = await self._fetch_chunks(course_id, video_id)
        if not chunks:
            raise ValueError("No transcript data found to generate an exam.")

        title = chunks[0].get("video_title", "Course Exam") if video_id else "Full Course Exam"

        # Sample chunks for context (exam doesn't need the full transcript)
        sampled = self._sample_chunks(chunks, max_chunks=12)
        context = self._build_context(sampled)

        prompt = _EXAM_PROMPT.format(
            difficulty=difficulty.capitalize(),
            context=context,
            difficulty_instructions=_DIFFICULTY_INSTRUCTIONS[difficulty],
        )

        raw = self._call_llm(_SYSTEM_PROMPT, prompt, temperature=0.5, max_tokens=3000)
        mcq, short_answer, long_answer = self._parse_exam(raw)

        logger.info(
            f"Exam generated: '{title}' | difficulty={difficulty} | "
            f"{len(mcq)} MCQ, {len(short_answer)} short, {len(long_answer)} long"
        )

        return ExamResult(
            title=title,
            difficulty=difficulty,
            mcq_questions=mcq,
            short_answer_questions=short_answer,
            long_answer_questions=long_answer,
            video_id=video_id,
            course_id=course_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_exam(raw: str) -> tuple[list[ExamQuestion], list[ExamQuestion], list[ExamQuestion]]:
        """Parse LLM JSON into typed ExamQuestion lists."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
        except Exception as e:
            logger.error(f"Exam JSON parse error: {e}\nRaw: {raw[:300]}")
            return [], [], []

        def parse_questions(items: list[dict], qtype: QuestionType) -> list[ExamQuestion]:
            questions = []
            for item in items:
                questions.append(ExamQuestion(
                    question_type=qtype,
                    question=item.get("question", ""),
                    options=item.get("options") if qtype == QuestionType.MCQ else None,
                    correct_answer=item.get("correct_answer", ""),
                    explanation=item.get("explanation", ""),
                ))
            return questions

        mcq          = parse_questions(data.get("mcq", []),          QuestionType.MCQ)
        short_answer = parse_questions(data.get("short_answer", []), QuestionType.SHORT_ANSWER)
        long_answer  = parse_questions(data.get("long_answer", []),  QuestionType.LONG_ANSWER)

        return mcq, short_answer, long_answer

    @staticmethod
    def _sample_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
        if len(chunks) <= max_chunks:
            return chunks
        step = len(chunks) // max_chunks
        return chunks[::step][:max_chunks]

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        return "\n\n".join(
            f"[{c.get('video_title', '')}]\n{c.get('chunk_text', '')}"
            for c in chunks if c.get("chunk_text")
        )

    @staticmethod
    async def _fetch_chunks(
        course_id: str,
        video_id: Optional[str] = None,
    ) -> list[dict]:
        col = get_chunks_collection()
        query = {"course_id": course_id}
        if video_id:
            query["video_id"] = video_id
        cursor = col.find(query).sort([("video_id", 1), ("position", 1)])
        return [doc async for doc in cursor]
