from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from whisperx_api.models import ErrorDetail, ErrorResponse


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


class Unauthorized(APIError):
    status_code = 401
    code = "unauthorized"
    message = "A valid API key is required."


class ResourceNotFound(APIError):
    status_code = 404
    code = "not_found"
    message = "The requested resource was not found."


class Conflict(APIError):
    status_code = 409
    code = "conflict"
    message = "The resource state conflicts with this operation."


class UnsupportedOption(APIError):
    status_code = 422
    code = "unsupported_option"


class UploadTooLarge(APIError):
    status_code = 413
    code = "upload_too_large"
    message = "The uploaded file exceeds the configured size limit."


class ModelUnavailable(APIError):
    status_code = 503
    code = "model_unavailable"


class InferenceTimeout(APIError):
    status_code = 504
    code = "inference_timeout"
    message = "The synchronous inference deadline was exceeded."


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            request_id=getattr(request.state, "request_id", None),
            details=exc.details,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
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
