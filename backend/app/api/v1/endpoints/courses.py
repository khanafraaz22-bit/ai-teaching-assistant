"""
Course management endpoints.

POST  /courses/            — Submit a YouTube playlist for ingestion
GET   /courses/            — List all courses for the current user
GET   /courses/{id}        — Get a single course + status
GET   /courses/{id}/videos — List videos in a course
"""

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from app.database.mongodb import (
    get_courses_collection,
    get_videos_collection,
)
from app.models.schemas import CourseInDB, ProcessingStatus
from app.core.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class AddPlaylistRequest(BaseModel):
    playlist_url: str


class AddPlaylistResponse(BaseModel):
    course_id: str
    message: str
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=AddPlaylistResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a YouTube playlist for processing",
)
async def add_playlist(
    payload: AddPlaylistRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])

    if "list=" not in payload.playlist_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL does not appear to be a YouTube playlist (missing 'list=' parameter).",
        )

    from app.services.youtube.playlist_parser import YouTubePlaylistParser

    try:
        parser = YouTubePlaylistParser()
        playlist_meta = parser.parse(payload.playlist_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Playlist parse error: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch playlist from YouTube.")

    # Insert course document immediately — caller gets an ID right away
    courses = get_courses_collection()
    doc = CourseInDB(
        user_id=user_id,
        playlist_id=playlist_meta.playlist_id,
        playlist_url=payload.playlist_url,
        title=playlist_meta.title,
        description=playlist_meta.description,
        channel_name=playlist_meta.channel_name,
        thumbnail_url=playlist_meta.thumbnail_url,
        total_videos=playlist_meta.total_videos,
        status=ProcessingStatus.PENDING,
    )
    result = await courses.insert_one(
        doc.model_dump(by_alias=False, exclude={"id"})
    )
    course_id = str(result.inserted_id)

    # Enqueue Celery task — runs in a separate worker process
    from app.workers.tasks import process_playlist
    task = process_playlist.delay(
        playlist_url=payload.playlist_url,
        user_id=user_id,
        course_id=course_id,
    )
    logger.info(f"Enqueued Celery task {task.id} for course {course_id}")

    # Store task ID so we can query progress later
    await courses.update_one(
        {"_id": result.inserted_id},
        {"$set": {"celery_task_id": task.id}},
    )

    return AddPlaylistResponse(
        course_id=course_id,
        message=f"Processing started for '{playlist_meta.title}' ({playlist_meta.total_videos} videos).",
        status=ProcessingStatus.PENDING.value,
    )


@router.get("/", summary="List all courses for a user")
async def list_courses(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    courses = get_courses_collection()
    cursor = courses.find({"user_id": user_id}).sort("created_at", -1)
    results = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        results.append(doc)
    return {"courses": results, "total": len(results)}


@router.get("/{course_id}", summary="Get course details and processing status")
async def get_course(course_id: str, current_user: dict = Depends(get_current_user)):
    from bson import ObjectId
    courses = get_courses_collection()
    try:
        doc = await courses.find_one({
            "_id": ObjectId(course_id),
            "user_id": str(current_user["_id"]),   # Scope to owner
        })
    except Exception:
        raise HTTPException(status_code=404, detail="Invalid course ID.")
    if not doc:
        raise HTTPException(status_code=404, detail="Course not found.")
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("/{course_id}/videos", summary="List all videos in a course")
async def get_course_videos(course_id: str, current_user: dict = Depends(get_current_user)):
    videos = get_videos_collection()
    cursor = videos.find({"course_id": course_id}).sort("position", 1)
    results = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        results.append(doc)
    return {"videos": results, "total": len(results)}


@router.get(
    "/{course_id}/job-status",
    summary="Get real-time processing progress for a course",
)
async def get_job_status(
    course_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Returns live Celery task progress while a course is being processed.

    Response during processing:
        { "state": "PROGRESS", "current": 5, "total": 20, "current_video": "..." }

    Response when done:
        { "state": "SUCCESS", "result": { "total_chunks": 312, ... } }

    Response on failure:
        { "state": "FAILURE", "error": "..." }
    """
    from bson import ObjectId
    from app.workers.celery_app import celery_app as _celery

    # Verify course belongs to user
    courses = get_courses_collection()
    try:
        doc = await courses.find_one({
            "_id": ObjectId(course_id),
            "user_id": str(current_user["_id"]),
        })
    except Exception:
        raise HTTPException(status_code=404, detail="Invalid course ID.")
    if not doc:
        raise HTTPException(status_code=404, detail="Course not found.")

    # Find the latest Celery task for this course
    # Task ID is stored in the course document when enqueued
    task_id = doc.get("celery_task_id")
    if not task_id:
        # No task ID stored — return MongoDB status as fallback
        return {
            "state":     doc.get("status", "unknown").upper(),
            "course_id": course_id,
        }

    task_result = _celery.AsyncResult(task_id)

    if task_result.state == "PROGRESS":
        meta = task_result.info or {}
        return {
            "state":         "PROGRESS",
            "current":       meta.get("current", 0),
            "total":         meta.get("total", 0),
            "current_video": meta.get("current_video", ""),
            "course_id":     course_id,
        }
    elif task_result.state == "SUCCESS":
        return {
            "state":    "SUCCESS",
            "result":   task_result.result,
            "course_id": course_id,
        }
    elif task_result.state == "FAILURE":
        return {
            "state":    "FAILURE",
            "error":    str(task_result.result),
            "course_id": course_id,
        }
    else:
        return {
            "state":    task_result.state,   # PENDING, STARTED, RETRY
            "course_id": course_id,
        }

# ── Progress endpoints ────────────────────────────────────────────────────────

@router.get(
    "/progress/dashboard",
    summary="Get full learning dashboard for the current user",
)
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    """
    Returns a summary of all courses the user has added, with:
    - Videos completed per course
    - Completion percentage
    - Topics studied
    - Quiz history
    - Overall stats
    """
    from app.services.learning.progress_service import ProgressService
    user_id = str(current_user["_id"])
    return await ProgressService().get_user_dashboard(user_id)


@router.get(
    "/{course_id}/progress",
    summary="Get learning progress for a specific course",
)
async def get_course_progress(
    course_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Detailed progress for one course:
    - Which videos are completed
    - Topics the user has asked about
    - Quiz results
    - Completion percentage
    """
    from app.services.learning.progress_service import ProgressService
    user_id = str(current_user["_id"])
    return await ProgressService().get_course_progress(user_id, course_id)


@router.post(
    "/{course_id}/videos/{video_id}/complete",
    summary="Mark a video as completed",
    status_code=201,
)
async def mark_video_completed(
    course_id: str,
    video_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Called by the frontend when a user finishes watching a video.
    Updates the user's completion percentage for the course.
    """
    from app.services.learning.progress_service import ProgressService
    user_id = str(current_user["_id"])
    await ProgressService().record_video_completed(user_id, course_id, video_id)
    return {"message": f"Video '{video_id}' marked as completed."}


@router.post(
    "/{course_id}/quiz-result",
    summary="Submit a quiz result",
    status_code=201,
)
async def submit_quiz_result(
    course_id: str,
    current_user: dict = Depends(get_current_user),
    score: float = 0.0,
    video_id: str = None,
):
    """
    Store a completed quiz attempt with a score (0–100).
    Called by the frontend after the user finishes an exam.
    """
    from app.services.learning.progress_service import ProgressService
    user_id = str(current_user["_id"])
    result_id = await ProgressService().record_quiz_result(
        user_id=user_id,
        course_id=course_id,
        questions=[],
        score=score,
        video_id=video_id,
    )
    return {"message": "Quiz result saved.", "result_id": result_id}
