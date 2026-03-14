"""
Progress Tracking Service
-------------------------
Tracks everything a user does in the system:
  - Which courses they've added
  - Which videos they've completed
  - Which topics they've studied (from chat questions)
  - Quiz scores over time

All writes are upserts — safe to call multiple times.
All reads are aggregated into a single dashboard response.
"""

import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId

from app.database.mongodb import (
    get_progress_collection,
    get_quiz_results_collection,
    get_courses_collection,
    get_videos_collection,
)
from app.models.schemas import ProgressInDB, QuizResultInDB, QuizQuestion

logger = logging.getLogger(__name__)


class ProgressService:

    # ── Record events ─────────────────────────────────────────────────────────

    async def record_video_completed(
        self,
        user_id: str,
        course_id: str,
        video_id: str,
    ) -> None:
        """Mark a video as completed for a user."""
        col = get_progress_collection()
        await col.update_one(
            {"user_id": user_id, "course_id": course_id},
            {
                "$addToSet": {"videos_completed": video_id},
                "$set":      {"last_activity": datetime.utcnow()},
                "$setOnInsert": {
                    "created_at":    datetime.utcnow(),
                    "topics_studied": [],
                },
            },
            upsert=True,
        )
        logger.info(f"Video completed: user={user_id} video={video_id}")

    async def record_topic_studied(
        self,
        user_id: str,
        course_id: str,
        topic: str,
    ) -> None:
        """
        Record a topic the user asked about.
        Called automatically from the chat endpoint after each question.
        """
        col = get_progress_collection()
        topic_clean = topic.strip().lower()[:100]   # Normalize + cap length
        await col.update_one(
            {"user_id": user_id, "course_id": course_id},
            {
                "$addToSet": {"topics_studied": topic_clean},
                "$set":      {"last_activity": datetime.utcnow()},
                "$setOnInsert": {
                    "created_at":        datetime.utcnow(),
                    "videos_completed":  [],
                },
            },
            upsert=True,
        )

    async def record_quiz_result(
        self,
        user_id: str,
        course_id: str,
        questions: list[QuizQuestion],
        score: Optional[float],
        video_id: Optional[str] = None,
    ) -> str:
        """
        Store a quiz attempt and return the result document ID.
        Score is 0.0–100.0 (percentage). None if not yet graded.
        """
        col = get_quiz_results_collection()
        doc = QuizResultInDB(
            user_id=user_id,
            course_id=course_id,
            video_id=video_id,
            questions=questions,
            score=score,
        )
        result = await col.insert_one(
            doc.model_dump(by_alias=False, exclude={"id"})
        )

        # Also update last_activity in progress
        await get_progress_collection().update_one(
            {"user_id": user_id, "course_id": course_id},
            {
                "$set": {"last_activity": datetime.utcnow()},
                "$setOnInsert": {
                    "created_at":        datetime.utcnow(),
                    "videos_completed":  [],
                    "topics_studied":    [],
                },
            },
            upsert=True,
        )

        logger.info(f"Quiz result saved: user={user_id} score={score}")
        return str(result.inserted_id)

    # ── Read progress ─────────────────────────────────────────────────────────

    async def get_course_progress(
        self,
        user_id: str,
        course_id: str,
    ) -> dict:
        """
        Progress for a single course — videos completed, topics studied,
        quiz history, and a completion percentage.
        """
        # Progress document
        progress_col = get_progress_collection()
        progress = await progress_col.find_one(
            {"user_id": user_id, "course_id": course_id}
        )

        # Total video count for this course
        videos_col = get_videos_collection()
        total_videos = await videos_col.count_documents({"course_id": course_id})

        videos_completed  = progress.get("videos_completed", []) if progress else []
        topics_studied    = progress.get("topics_studied",   []) if progress else []
        last_activity     = progress.get("last_activity")        if progress else None

        completion_pct = (
            round(len(videos_completed) / total_videos * 100, 1)
            if total_videos > 0 else 0.0
        )

        # Quiz history for this course
        quiz_history = await self._get_quiz_history(user_id, course_id)

        return {
            "course_id":          course_id,
            "total_videos":       total_videos,
            "videos_completed":   videos_completed,
            "videos_completed_count": len(videos_completed),
            "completion_percentage":  completion_pct,
            "topics_studied":     topics_studied,
            "topics_studied_count":   len(topics_studied),
            "quiz_history":       quiz_history,
            "last_activity":      last_activity,
        }

    async def get_user_dashboard(self, user_id: str) -> dict:
        """
        Full dashboard for a user — all courses with progress summaries.
        """
        # All courses for this user
        courses_col = get_courses_collection()
        cursor = courses_col.find({"user_id": user_id}).sort("created_at", -1)
        courses = [doc async for doc in cursor]

        course_summaries = []
        total_videos_completed = 0
        total_topics_studied   = 0
        total_quizzes_taken    = 0

        for course in courses:
            course_id  = str(course["_id"])
            progress   = await self.get_course_progress(user_id, course_id)

            total_videos_completed += progress["videos_completed_count"]
            total_topics_studied   += progress["topics_studied_count"]
            total_quizzes_taken    += len(progress["quiz_history"])

            course_summaries.append({
                "course_id":             course_id,
                "title":                 course.get("title", ""),
                "thumbnail_url":         course.get("thumbnail_url"),
                "status":                course.get("status"),
                "total_videos":          progress["total_videos"],
                "videos_completed":      progress["videos_completed_count"],
                "completion_percentage": progress["completion_percentage"],
                "topics_studied":        progress["topics_studied_count"],
                "quizzes_taken":         len(progress["quiz_history"]),
                "last_activity":         progress["last_activity"],
            })

        # Average quiz score across all courses
        all_scores = [
            q["score"]
            for cs in course_summaries
            for q in (await self._get_quiz_history(user_id, cs["course_id"]))
            if q.get("score") is not None
        ]
        avg_quiz_score = (
            round(sum(all_scores) / len(all_scores), 1) if all_scores else None
        )

        return {
            "user_id":               user_id,
            "total_courses":         len(courses),
            "total_videos_completed": total_videos_completed,
            "total_topics_studied":  total_topics_studied,
            "total_quizzes_taken":   total_quizzes_taken,
            "average_quiz_score":    avg_quiz_score,
            "courses":               course_summaries,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    async def _get_quiz_history(user_id: str, course_id: str) -> list[dict]:
        """Fetch quiz results for a user/course, most recent first."""
        col = get_quiz_results_collection()
        cursor = col.find(
            {"user_id": user_id, "course_id": course_id},
            {"questions": 0},           # Exclude full question bodies from summary
        ).sort("created_at", -1).limit(20)

        results = []
        async for doc in cursor:
            results.append({
                "quiz_id":    str(doc["_id"]),
                "score":      doc.get("score"),
                "video_id":   doc.get("video_id"),
                "created_at": doc.get("created_at"),
            })
        return results
