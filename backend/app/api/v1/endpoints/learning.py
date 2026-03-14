"""
Learning Tools endpoints — Phase 4.

POST /learning/summarize      — Summarize a video or full course
POST /learning/topic-search   — Find where a topic appears in the course
POST /learning/mindmap        — Generate a concept mind map
POST /learning/exam           — Generate an exam with MCQ + written questions
"""

from __future__ import annotations


import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Optional

from app.services.learning.summarizer import SummarizerService
from app.services.learning.topic_search import TopicSearchService
from app.services.learning.mindmap import MindMapService
from app.services.learning.exam_generator import ExamGeneratorService
from app.core.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared request base ───────────────────────────────────────────────────────

class CourseRequest(BaseModel):
    course_id: str
    video_id: Optional[str] = None      # If omitted → operate on full course


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARIZE
# ═══════════════════════════════════════════════════════════════════════════════

class SummarizeResponse(BaseModel):
    title: str
    overview: str
    key_points: list[str]
    concepts_covered: list[str]
    takeaway: str
    video_id: Optional[str]
    course_id: str


@router.post("/summarize", response_model=SummarizeResponse, summary="Summarize a video lecture or full course")
async def summarize(payload: CourseRequest, current_user: dict = Depends(get_current_user)):
    """
    Pass only `course_id` for a full course summary.
    Pass both `course_id` and `video_id` for a single video summary.
    """
    service = SummarizerService()
    try:
        if payload.video_id:
            result = await service.summarize_video(
                video_id=payload.video_id,
                course_id=payload.course_id,
            )
        else:
            result = await service.summarize_course(course_id=payload.course_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SummarizeResponse(
        title=result.title,
        overview=result.overview,
        key_points=result.key_points,
        concepts_covered=result.concepts_covered,
        takeaway=result.takeaway,
        video_id=result.video_id,
        course_id=result.course_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class TopicSearchRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    course_id: str
    top_k: int = Field(default=8, ge=1, le=20)


class TopicOccurrenceResponse(BaseModel):
    video_id: str
    video_title: str
    timestamp: str
    youtube_url: str
    relevance_score: float
    excerpt: str


class TopicSearchResponse(BaseModel):
    topic: str
    found: bool
    total_found: int
    best_explanation: str
    occurrences: list[TopicOccurrenceResponse]


@router.post("/topic-search", response_model=TopicSearchResponse, summary="Find where a topic is explained in the course")
async def topic_search(payload: TopicSearchRequest, current_user: dict = Depends(get_current_user)):
    """
    Example: `"topic": "gradient descent"` returns every video and timestamp
    where gradient descent is discussed, plus a synthesized explanation.
    """
    service = TopicSearchService()
    try:
        result = service.search(
            topic=payload.topic,
            course_id=payload.course_id,
            top_k=payload.top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return TopicSearchResponse(
        topic=result.topic,
        found=result.found,
        total_found=result.total_found,
        best_explanation=result.best_explanation,
        occurrences=[
            TopicOccurrenceResponse(
                video_id=o.video_id,
                video_title=o.video_title,
                timestamp=o.timestamp,
                youtube_url=o.youtube_url,
                relevance_score=o.relevance_score,
                excerpt=o.excerpt,
            )
            for o in result.occurrences
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MIND MAP
# ═══════════════════════════════════════════════════════════════════════════════

class MindMapNodeResponse(BaseModel):
    label: str
    children: list["MindMapNodeResponse"] = []


class MindMapResponse(BaseModel):
    title: str
    root: MindMapNodeResponse
    text_render: str
    video_id: Optional[str]
    course_id: str


def _node_to_response(node) -> MindMapNodeResponse:
    return MindMapNodeResponse(
        label=node.label,
        children=[_node_to_response(c) for c in node.children],
    )


@router.post("/mindmap", response_model=MindMapResponse, summary="Generate a concept mind map for a course or video")
async def generate_mindmap(payload: CourseRequest, current_user: dict = Depends(get_current_user)):
    """
    Returns both a structured JSON tree (for frontend rendering)
    and a plain-text ASCII render (for quick display).
    """
    service = MindMapService()
    try:
        result = await service.generate(
            course_id=payload.course_id,
            video_id=payload.video_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return MindMapResponse(
        title=result.title,
        root=_node_to_response(result.root),
        text_render=result.text_render,
        video_id=result.video_id,
        course_id=result.course_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ExamRequest(BaseModel):
    course_id: str
    video_id: Optional[str] = None
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class ExamQuestionResponse(BaseModel):
    question_type: str
    question: str
    options: Optional[list[str]]
    correct_answer: str
    explanation: str


class ExamResponse(BaseModel):
    title: str
    difficulty: str
    total_questions: int
    mcq_questions: list[ExamQuestionResponse]
    short_answer_questions: list[ExamQuestionResponse]
    long_answer_questions: list[ExamQuestionResponse]
    video_id: Optional[str]
    course_id: str


@router.post("/exam", response_model=ExamResponse, summary="Generate an exam from course material")
async def generate_exam(payload: ExamRequest, current_user: dict = Depends(get_current_user)):
    """
    Generates:
    - 5 Multiple Choice Questions
    - 2 Short Answer Questions
    - 1 Long Answer / Essay Question

    All grounded in the actual transcript content.
    Difficulty: easy | medium | hard
    """
    service = ExamGeneratorService()
    try:
        result = await service.generate(
            course_id=payload.course_id,
            video_id=payload.video_id,
            difficulty=payload.difficulty,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    def q_to_response(q) -> ExamQuestionResponse:
        return ExamQuestionResponse(
            question_type=q.question_type.value,
            question=q.question,
            options=q.options,
            correct_answer=q.correct_answer,
            explanation=q.explanation,
        )

    return ExamResponse(
        title=result.title,
        difficulty=result.difficulty,
        total_questions=result.total_questions,
        mcq_questions=[q_to_response(q) for q in result.mcq_questions],
        short_answer_questions=[q_to_response(q) for q in result.short_answer_questions],
        long_answer_questions=[q_to_response(q) for q in result.long_answer_questions],
        video_id=result.video_id,
        course_id=result.course_id,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# COURSE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

class MissingTopicResponse(BaseModel):
    topic: str
    why_important: str
    how_related: str
    difficulty: str


class ResourceRecommendationResponse(BaseModel):
    topic: str
    resource_type: str
    title: str
    description: str
    search_query: str


class CourseIntelligenceResponse(BaseModel):
    course_id: str
    course_title: str
    topics_covered: list[str]
    coverage_summary: str
    missing_topics: list[MissingTopicResponse]
    recommendations: list[ResourceRecommendationResponse]
    total_gaps_found: int
    total_recommendations: int


@router.post(
    "/course-intelligence",
    response_model=CourseIntelligenceResponse,
    summary="Detect missing topics and get external resource recommendations",
)
async def course_intelligence(
    payload: CourseRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Analyzes the full course content and returns:

    1. **Topics covered** — what the course actually teaches
    2. **Coverage summary** — an overall assessment of course completeness
    3. **Missing topics** — important concepts not covered or barely mentioned,
       with explanations of why they matter and how they relate to the course
    4. **Resource recommendations** — one external resource per gap
       (YouTube search, documentation, article) to help students fill the gap

    This is a relatively expensive call — it samples transcript chunks,
    calls the LLM twice, and may take 10–20 seconds for large courses.
    """
    from app.services.learning.course_intelligence import CourseIntelligenceService

    service = CourseIntelligenceService()
    try:
        result = await service.analyze(course_id=payload.course_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return CourseIntelligenceResponse(
        course_id=result.course_id,
        course_title=result.course_title,
        topics_covered=result.topics_covered,
        coverage_summary=result.coverage_summary,
        missing_topics=[
            MissingTopicResponse(
                topic=t.topic,
                why_important=t.why_important,
                how_related=t.how_related,
                difficulty=t.difficulty,
            )
            for t in result.missing_topics
        ],
        recommendations=[
            ResourceRecommendationResponse(
                topic=r.topic,
                resource_type=r.resource_type,
                title=r.title,
                description=r.description,
                search_query=r.search_query,
            )
            for r in result.recommendations
        ],
        total_gaps_found=len(result.missing_topics),
        total_recommendations=len(result.recommendations),
    )