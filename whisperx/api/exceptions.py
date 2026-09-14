import logging
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from whisperx.api.models import ErrorDetail, ErrorResponse


class APIError(Exception):
    status_code = 500
    code = "internal_error"
    message = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details
        super().__init__(self.message)


class UnsupportedOption(APIError):
    status_code = 422
    code = "unsupported_option"


class ModelUnavailable(APIError):
    status_code = 503
    code = "model_unavailable"


class InvalidAudio(APIError):
    status_code = 422
    code = "invalid_audio"


async def api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error = exc if isinstance(exc, APIError) else APIError()
    # Log server faults once, preserving the original traceback and chained cause.
    if error.status_code >= 500:
        logging.getLogger(__name__).error(
            "request_id=%s method=%s path=%s status=%s API request failed",
            getattr(request.state, "request_id", None),
            request.method,
            request.url.path,
            error.status_code,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    payload = ErrorResponse(
        error=ErrorDetail(
            code=error.code,
            message=error.message,
            request_id=getattr(request.state, "request_id", None),
            details=error.details,
        )
    )
    return JSONResponse(
        status_code=error.status_code,
        content=payload.model_dump(mode="json"),
        headers={"X-Request-ID": payload.error.request_id or ""},
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(
        error=ErrorDetail(
            code="validation_error",
            message="The request did not pass validation.",
            request_id=request_id,
            details={"errors": jsonable_encoder(exc.errors())},
        )
    )
    return JSONResponse(
        status_code=422,
        content=payload.model_dump(mode="json"),
        headers={"X-Request-ID": request_id or ""},
    )
