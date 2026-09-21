"""Structured application errors and FastAPI exception handlers."""

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ApplicationError(Exception):
    """Expected domain error safe to return to an API client."""

    status_code: int
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


def error_payload(
    code: str, message: str, details: list[dict[str, Any]] | None = None
) -> dict[str, dict[str, object]]:
    """Build the common error envelope."""
    return {"error": {"code": code, "message": message, "details": details}}


def register_exception_handlers(app: FastAPI) -> None:
    """Install stable JSON handlers for domain, HTTP, validation, and server errors."""

    @app.exception_handler(ApplicationError)
    async def application_error_handler(_request: Request, exc: ApplicationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(error_payload(exc.code, exc.message, exc.details)),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = jsonable_encoder(exc.errors())
        return JSONResponse(
            status_code=422,
            content=error_payload("validation_error", "Request validation failed", details),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload("http_error", message),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled application error on path %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=error_payload("internal_error", "An unexpected error occurred"),
        )
