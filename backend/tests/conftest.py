"""
conftest.py — Shared pytest fixtures for all tests.
Available to every test file automatically — no imports needed.
"""

import asyncio
import pytest
from unittest.mock import MagicMock


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Sample transcript data ────────────────────────────────────────────────────

@pytest.fixture
def sample_segments():
    from app.services.youtube.transcript_extractor import TranscriptSegment
    return [
        TranscriptSegment(text="Welcome to this lecture on neural networks.",      start=0.0,  duration=3.5),
        TranscriptSegment(text="Today we will cover gradient descent.",             start=3.5,  duration=4.0),
        TranscriptSegment(text="Gradient descent is an optimization algorithm.",    start=7.5,  duration=4.2),
        TranscriptSegment(text="It minimizes the loss function iteratively.",       start=11.7, duration=3.8),
        TranscriptSegment(text="We start with random weights.",                     start=15.5, duration=3.0),
        TranscriptSegment(text="Then we compute the gradient of the loss.",        start=18.5, duration=4.5),
        TranscriptSegment(text="The gradient points in the direction of steepest ascent.", start=23.0, duration=4.0),
        TranscriptSegment(text="We move in the opposite direction to minimize.",   start=27.0, duration=4.0),
        TranscriptSegment(text="The learning rate controls the step size.",        start=31.0, duration=3.5),
        TranscriptSegment(text="A high learning rate may overshoot the minimum.", start=34.5, duration=4.0),
        TranscriptSegment(text="A low learning rate converges slowly.",            start=38.5, duration=3.5),
        TranscriptSegment(text="Backpropagation computes gradients efficiently.",  start=42.0, duration=4.5),
        TranscriptSegment(text="It uses the chain rule from calculus.",            start=46.5, duration=3.5),
        TranscriptSegment(text="This is the foundation of deep learning.",         start=50.0, duration=3.0),
    ]


@pytest.fixture
def sample_raw_transcript(sample_segments):
    from app.services.youtube.transcript_extractor import RawTranscript
    return RawTranscript(
        video_id="test_video_001",
        segments=sample_segments,
        language="en",
        source="youtube_manual",
    )


@pytest.fixture
def sample_cleaned_text():
    return (
        "Welcome to this lecture on neural networks. Today we will cover gradient descent. "
        "Gradient descent is an optimization algorithm. It minimizes the loss function iteratively. "
        "We start with random weights. Then we compute the gradient of the loss. "
        "The gradient points in the direction of steepest ascent. "
        "We move in the opposite direction to minimize. "
        "The learning rate controls the step size. "
        "A high learning rate may overshoot the minimum. "
        "A low learning rate converges slowly. "
        "Backpropagation computes gradients efficiently. "
        "It uses the chain rule from calculus. This is the foundation of deep learning."
    )


@pytest.fixture
def sample_chunks(sample_cleaned_text):
    from app.services.youtube.chunker import TranscriptChunk
    return [
        TranscriptChunk(
            chunk_id="chunk-uuid-001",
            video_id="test_video_001",
            video_title="Lecture 1 — Gradient Descent",
            course_id="course-abc-123",
            chunk_text=sample_cleaned_text[:400],
            start_timestamp=0.0,
            end_timestamp=27.0,
            token_count=85,
            position=0,
        ),
        TranscriptChunk(
            chunk_id="chunk-uuid-002",
            video_id="test_video_001",
            video_title="Lecture 1 — Gradient Descent",
            course_id="course-abc-123",
            chunk_text=sample_cleaned_text[300:],
            start_timestamp=25.0,
            end_timestamp=53.0,
            token_count=80,
            position=1,
        ),
    ]


@pytest.fixture
def sample_retrieved_chunks():
    from app.services.rag.retrieval_service import RetrievedChunk
    return [
        RetrievedChunk(
            chunk_id="chunk-uuid-001",
            chunk_text="Gradient descent is an optimization algorithm that minimizes the loss function.",
            video_id="test_video_001",
            video_title="Lecture 1 — Gradient Descent",
            start_timestamp=7.5,
            end_timestamp=20.0,
            course_id="course-abc-123",
            relevance_score=0.92,
            position=0,
        ),
        RetrievedChunk(
            chunk_id="chunk-uuid-002",
            chunk_text="The learning rate controls the step size during gradient descent.",
            video_id="test_video_001",
            video_title="Lecture 1 — Gradient Descent",
            start_timestamp=31.0,
            end_timestamp=42.0,
            course_id="course-abc-123",
            relevance_score=0.85,
            position=1,
        ),
    ]


# ── Mock fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_embedding():
    vector = [0.01] * 1536
    mock = MagicMock()
    mock.data = [MagicMock(embedding=vector, index=0)]
    return mock


@pytest.fixture
def mock_chroma_collection():
    mock = MagicMock()
    mock.upsert = MagicMock(return_value=None)
    mock.query = MagicMock(return_value={
        "ids":       [["chunk-uuid-001", "chunk-uuid-002"]],
        "documents": [["Gradient descent text.", "Learning rate text."]],
        "metadatas": [[
            {"course_id": "course-abc-123", "video_id": "test_video_001",
             "video_title": "Lecture 1", "start_timestamp": 7.5,
             "end_timestamp": 20.0, "token_count": 85, "position": 0},
            {"course_id": "course-abc-123", "video_id": "test_video_001",
             "video_title": "Lecture 1", "start_timestamp": 31.0,
             "end_timestamp": 42.0, "token_count": 80, "position": 1},
        ]],
        "distances": [[0.08, 0.15]],
    })
    mock.delete = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_llm_response():
    mock = MagicMock()
    mock.choices = [MagicMock(
        message=MagicMock(content=(
            "Gradient descent is an optimization algorithm used to minimize "
            "the loss function in neural networks, as explained in "
            "'Lecture 1 — Gradient Descent' at 0:07."
        ))
    )]
    return mock
