import secrets
from typing import Annotated

from fastapi import Depends, Header, Request

from whisperx_api.config import Settings
from whisperx_api.exceptions import Unauthorized


async def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDependency = Annotated[Settings, Depends(get_app_settings)]


async def require_api_key(
    settings: SettingsDependency,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if settings.api_key is None:
        return
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:]
    supplied = x_api_key or bearer
    if supplied is None or not secrets.compare_digest(supplied, settings.api_key):
        raise Unauthorized()


APIKeyDependency = Annotated[None, Depends(require_api_key)]
