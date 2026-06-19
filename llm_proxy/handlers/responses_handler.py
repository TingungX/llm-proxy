"""/v1/responses 路由 Pipeline"""

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.handlers.base import PipelineContext, Pipeline
from llm_proxy.handlers.shared.auth import AuthStep
from llm_proxy.handlers.shared.model_resolve import ModelResolveStep
from llm_proxy.handlers.shared.vision_fallback import VisionFallbackStep
from llm_proxy.handlers.shared.compression import CompressionStep
from llm_proxy.handlers.shared.ir_proxy import IRProxyStep
from llm_proxy.protocol.errors import make_openai_error

logger = logging.getLogger(__name__)


class ResponsesHandler:
    def __init__(self):
        self.pipeline = Pipeline([
            AuthStep(),
            ModelResolveStep(),
            # 工具降级逻辑已内置于 IRProxyStep
            VisionFallbackStep(),
            CompressionStep(),
            IRProxyStep(client_protocol="openai/responses"),
        ])

    async def handle(self, request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raw_body = await request.body()
            logger.warning(f"Empty or invalid JSON body: {raw_body[:100] if raw_body else '<empty>'}")
            return make_openai_error(
                "Request body must be valid JSON",
                "invalid_request_error",
                400,
            )

        ctx = PipelineContext(
            request=request,
            body=body,
            headers=dict(request.headers),
            error_protocol="openai",
        )
        return await self.pipeline.execute(ctx)
