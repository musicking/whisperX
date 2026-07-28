from typing import Annotated

from fastapi import Depends, Request

from whisperx_api.audio.engine import SpeechEngine


async def get_engine(request: Request) -> SpeechEngine:
    return request.app.state.engine


EngineDependency = Annotated[SpeechEngine, Depends(get_engine)]
