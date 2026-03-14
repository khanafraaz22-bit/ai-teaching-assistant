"""
Database document schemas using Pydantic.
These serve as both MongoDB document definitions and API response shapes.

Naming convention:
  - Base      → shared fields
  - InDB      → stored in MongoDB (includes _id, timestamps)
  - Create    → incoming API payload
  - Response  → outgoing API payload (no secrets)
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ── Shared helpers ────────────────────────────────────────────────────────────

class PyObjectId(str):
    """Coerce MongoDB ObjectId to plain string for Pydantic v2."""
    pass


def utcnow() -> datetime:
    return datetime.utcnow()


# ── Enums ─────────────────────────────────────────────────────────────────────

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(str, Enum):
    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"


# ═══════════════════════════════════════════════════════════════════════════════
# USER
# ═══════════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=2)


class UserInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    email: EmailStr
    full_name: str
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    verification_token: Optional[str] = None
    reset_token: Optional[str] = None
    reset_token_expires: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# COURSE
# ═══════════════════════════════════════════════════════════════════════════════

class CourseCreate(BaseModel):
    playlist_url: str
    user_id: str


class CourseInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    playlist_id: str                   # YouTube playlist ID
    playlist_url: str
    title: str
    description: Optional[str] = None
    channel_name: Optional[str] = None
    thumbnail_url: Optional[str] = None
    total_videos: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


class CourseResponse(BaseModel):
    id: str
    playlist_id: str
    playlist_url: str
    title: str
    description: Optional[str]
    channel_name: Optional[str]
    thumbnail_url: Optional[str]
    total_videos: int
    status: ProcessingStatus
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO
# ═══════════════════════════════════════════════════════════════════════════════

class VideoInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    course_id: str
    user_id: str
    video_id: str                      # YouTube video ID (e.g. "dQw4w9WgXcQ")
    title: str
    url: str
    position: int                      # Order within the playlist
    duration_seconds: Optional[int] = None
    thumbnail_url: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    transcript_available: bool = False
    created_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


class VideoResponse(BaseModel):
    id: str
    video_id: str
    title: str
    url: str
    position: int
    duration_seconds: Optional[int]
    thumbnail_url: Optional[str]
    status: ProcessingStatus
    transcript_available: bool


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

class TranscriptSegment(BaseModel):
    """A single timed segment from the raw YouTube transcript."""
    text: str
    start: float           # seconds
    duration: float        # seconds


class TranscriptInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    video_id: str                      # YouTube video ID
    course_id: str
    raw_segments: List[TranscriptSegment] = []
    cleaned_text: str = ""             # Full merged + cleaned transcript
    language: str = "en"
    source: str = "youtube"            # "youtube" | "whisper"
    created_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK  (the unit stored in both MongoDB and ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    chunk_id: str                      # UUID — same ID used in ChromaDB
    course_id: str
    video_id: str                      # YouTube video ID
    video_title: str
    chunk_text: str
    start_timestamp: float             # seconds into the video
    end_timestamp: float
    token_count: int
    position: int                      # Chunk index within the video
    created_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    course_id: str
    video_id: Optional[str] = None
    topics_studied: List[str] = []
    videos_completed: List[str] = []
    last_activity: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


# ═══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ═══════════════════════════════════════════════════════════════════════════════

class QuizQuestion(BaseModel):
    question_type: QuestionType
    question: str
    options: Optional[List[str]] = None     # MCQ only
    correct_answer: str
    explanation: Optional[str] = None


class QuizResultInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    course_id: str
    video_id: Optional[str] = None
    questions: List[QuizQuestion] = []
    score: Optional[float] = None
    created_at: datetime = Field(default_factory=utcnow)

    class Config:
        populate_by_name = True


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
