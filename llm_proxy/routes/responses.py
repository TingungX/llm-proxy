"""OpenAI Responses API 路由 (/v1/responses)

将 Codex Desktop 的 Responses API 请求转换为 Chat Completions 格式进行代理，
再将上游响应转换回 Responses API 格式。仅支持 OpenAI 上游。
"""

import logging

from fastapi import Request

from llm_proxy.main import app
from llm_proxy.handlers.responses_handler import ResponsesHandler

logger = logging.getLogger(__name__)

_handler = ResponsesHandler()


@app.post("/v1/responses")
async def openai_responses(request: Request):
    return await _handler.handle(request)
