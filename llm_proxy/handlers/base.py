"""Pipeline 基础设施 — PipelineContext, HandlerStep, Pipeline, PipelineStop"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse


class PipelineStop(Exception):
    """Pipeline 中断信号 — 步骤可抛出此异常直接返回响应

    用法:
        raise PipelineStop(JSONResponse({"error": "..."}, status_code=401))
    """

    def __init__(self, response: JSONResponse | StreamingResponse):
        self.response = response
        super().__init__()


@dataclass
class PipelineContext:
    """Pipeline 执行上下文 — 在步骤间传递状态"""

    request: Request
    body: dict
    headers: dict[str, str]
    error_protocol: str = "openai"  # "openai" | "anthropic" — 决定错误响应格式

    # 以下字段由各步骤填充
    api_key: str = ""
    endpoint: dict | None = None
    resolved: tuple | None = None  # resolve_model_for_endpoint 的六元组结果
    upstream_protocol: str = ""
    converter: str | None = None  # "responses_to_chat" | "chat_to_responses" | None
    reverse_tool_map: dict | None = None
    namespace_map: dict | None = None  # 子工具名 → server_label（namespace 展开时填充）
    response: JSONResponse | StreamingResponse | None = None

    # 额外数据
    extra: dict[str, Any] = field(default_factory=dict)


class HandlerStep:
    """Pipeline 步骤基类"""

    async def execute(self, ctx: PipelineContext) -> None:
        """执行步骤，修改 ctx 或抛出 PipelineStop"""
        raise NotImplementedError


class Pipeline:
    """有序步骤管道 — 依次执行每个步骤"""

    def __init__(self, steps: list[HandlerStep]):
        self.steps = steps

    async def execute(self, ctx: PipelineContext) -> JSONResponse | StreamingResponse:
        for step in self.steps:
            await step.execute(ctx)
            if ctx.response is not None:
                return ctx.response
        # 如果没有步骤设置 response，返回 500
        from llm_proxy.protocol.errors import make_openai_error, make_anthropic_error
        if ctx.error_protocol == "anthropic":
            raise PipelineStop(make_anthropic_error("No response generated", "server_error", 500))
        raise PipelineStop(make_openai_error("No response generated", "server_error", 500))

