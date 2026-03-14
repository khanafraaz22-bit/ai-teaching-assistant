"""
Celery Application
------------------
Celery is the task queue. Redis is the broker (sends tasks)
and the result backend (stores task results).

Workers are started separately from the FastAPI server:
    celery -A app.workers.celery_app worker --loglevel=info

The FastAPI app only *enqueues* tasks — it never runs them directly.
"""

from celery import Celery
from app.config.settings import settings

# Create the Celery app
celery_app = Celery(
    "yt_course_assistant",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],      # Where task functions live
)

celery_app.conf.update(
    # ── Serialization ─────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Reliability ───────────────────────────────────────────────────────────
    task_acks_late=True,               # Only ack after task completes (not on receipt)
    task_reject_on_worker_lost=True,   # Re-queue if worker dies mid-task
    worker_prefetch_multiplier=1,      # One task at a time per worker process

    # ── Retries ───────────────────────────────────────────────────────────────
    task_max_retries=3,
    task_default_retry_delay=60,       # Seconds between retries

    # ── Result expiry ─────────────────────────────────────────────────────────
    result_expires=86400,              # Keep results for 24 hours (seconds)

    # ── Timeouts ──────────────────────────────────────────────────────────────
    task_soft_time_limit=3600,         # 1 hour soft limit (sends warning)
    task_time_limit=4200,              # 1h10m hard limit (kills task)

    # ── Timezone ──────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
)
