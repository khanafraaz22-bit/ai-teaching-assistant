"""
Global Exception Handlers
--------------------------
Registered on the FastAPI app at startup.
Catches errors that slip through endpoint-level try/except blocks
and returns consistent, safe JSON responses.

Rules:
  - Never expose internal error details to the client in production
  - Always log the full error server-side
  - Return structured { "detail": "...", "code": "..." } responses
"""

import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI app."""

    # ── Validation errors (422) ───────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Pydantic validation failures — missing fields, wrong types, etc.
        Reformat into a clean list of field errors.
        """
        errors = []
        for error in exc.errors():
            field = " → ".join(str(loc) for loc in error.get("loc", []))
            errors.append({
                "field":   field,
                "message": error.get("msg", "Invalid value"),
                "type":    error.get("type", ""),
            })

        logger.warning(f"Validation error on {request.url}: {errors}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Request validation failed.",
                "code":   "VALIDATION_ERROR",
                "errors": errors,
            },
        )

    # ── HTTP exceptions (4xx, 5xx) ────────────────────────────────────────────
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        Handle all HTTPExceptions raised anywhere in the app.
        Log 5xx errors as errors, 4xx as warnings.
        """
        if exc.status_code >= 500:
            logger.error(
                f"HTTP {exc.status_code} on {request.method} {request.url}: "
                f"{exc.detail}"
            )
        else:
            logger.warning(
                f"HTTP {exc.status_code} on {request.method} {request.url}: "
                f"{exc.detail}"
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "code":   _status_to_code(exc.status_code),
            },
        )

    # ── Catch-all (500) ───────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Any exception not caught elsewhere.
        Log the full traceback server-side, return a safe generic message.
        """
        logger.exception(
            f"Unhandled exception on {request.method} {request.url}: {exc}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected error occurred. Please try again later.",
                "code":   "INTERNAL_SERVER_ERROR",
            },
        )


def _status_to_code(status_code: int) -> str:
    """Map HTTP status codes to readable error codes."""
    return {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }.get(status_code, "HTTP_ERROR")
