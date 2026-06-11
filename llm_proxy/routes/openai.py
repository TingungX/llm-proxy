"""OpenAI Chat Completions API 路由 (/v1/chat/completions)"""

import logging

from fastapi import Request

from llm_proxy.main import app
from llm_proxy.handlers.openai_handler import OpenAIHandler

logger = logging.getLogger(__name__)

_handler = OpenAIHandler()


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    return await _handler.handle(request)

