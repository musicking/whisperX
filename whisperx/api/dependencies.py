from typing import Annotated

from fastapi import Depends, Request

from whisperx.api.config import Settings


async def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDependency = Annotated[Settings, Depends(get_app_settings)]
