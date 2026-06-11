"""POST /v1/messages — 消息代理入口"""

import logging

from fastapi import Request

from llm_proxy.main import app
from llm_proxy.handlers.messages_handler import MessagesHandler

logger = logging.getLogger(__name__)

_handler = MessagesHandler()


@app.api_route("/v1/messages", methods=["POST"])
async def proxy_messages(request: Request):
    return await _handler.handle(request)

