"""
Startup Validator
-----------------
Checks critical configuration before the server starts accepting traffic.
Catches misconfiguration early — much better than failing on the first request.

Called from main.py lifespan before any other startup step.
"""

import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)


def validate_config() -> None:
    """
    Raise a clear RuntimeError if the config is invalid for the environment.
    Logs all issues before raising so every problem is visible at once.
    """
    errors = []

    # ── Secret key must be strong ─────────────────────────────────────────────
    if len(settings.SECRET_KEY) < 32:
        errors.append(
            f"SECRET_KEY is too short ({len(settings.SECRET_KEY)} chars). "
            f"Minimum 32. Generate with: openssl rand -hex 32"
        )

    if settings.SECRET_KEY in (
        "your-secret-key-here", "secret", "changeme", "dev", "test"
    ):
        errors.append("SECRET_KEY is a placeholder. Set a real secret key.")

    # ── OpenAI key must be present ────────────────────────────────────────────
    if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.startswith("sk-"):
        errors.append("OPENAI_API_KEY is missing or invalid.")

    # ── Production-only checks ────────────────────────────────────────────────
    if settings.ENVIRONMENT == "production":

        if settings.DEBUG:
            errors.append(
                "DEBUG=true in production exposes API docs and stack traces."
            )

        if "*" in settings.ALLOWED_ORIGINS:
            errors.append(
                "ALLOWED_ORIGINS contains '*' — this allows any domain to call "
                "your API. Set specific origins in production."
            )

        if not settings.SMTP_HOST:
            logger.warning(
                "SMTP_HOST is not set in production. "
                "Email verification and password reset will not work."
            )

    # ── Report all errors ─────────────────────────────────────────────────────
    if errors:
        for error in errors:
            logger.critical(f"❌ Config error: {error}")
        raise RuntimeError(
            f"Server startup aborted due to {len(errors)} configuration "
            f"error(s). Check logs above."
        )

    logger.info(
        f"✅ Config validated — environment={settings.ENVIRONMENT}, "
        f"debug={settings.DEBUG}"
    )
