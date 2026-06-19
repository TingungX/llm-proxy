"""视觉降级步骤 — 对不支持视觉的模型替换图片为文本描述"""

import logging

from llm_proxy.handlers.base import PipelineContext, HandlerStep
from llm_proxy.state import get_state
from llm_proxy.services.vision_service import (
    replace_images_in_anthropic_messages,
    replace_images_in_chat_messages,
    replace_images_in_responses_input,
)

logger = logging.getLogger(__name__)


class VisionFallbackStep(HandlerStep):
    """视觉降级步骤：对不支持视觉的模型，将图片替换为文本描述

    根据 error_protocol 和 converter 决定替换方式：
    - anthropic: 替换 Anthropic messages 中的图片
    - openai + responses_to_chat: 替换 Responses input 中的图片
    - openai + chat passthrough: 替换 Chat messages 中的图片
    """

    async def execute(self, ctx: PipelineContext) -> None:
        _, _, _, model_id, _, _ = ctx.resolved
        s = get_state()

        if s.vision_map.get(model_id.lower(), False):
            return  # 模型支持视觉，无需降级

        if ctx.error_protocol == "anthropic":
            if isinstance(ctx.body.get("messages"), list):
                ctx.body["messages"] = await replace_images_in_anthropic_messages(ctx.body["messages"])
        elif isinstance(ctx.body.get("input"), list):
            # Responses 格式（有 input 数组）：替换 input 中的图片
            if isinstance(ctx.body.get("input"), list):
                ctx.body["input"] = await replace_images_in_responses_input(ctx.body["input"])
        else:
            # Chat Completions 同协议透传
            if isinstance(ctx.body.get("messages"), list):
                ctx.body["messages"] = await replace_images_in_chat_messages(ctx.body["messages"])

        logger.debug(f"Vision fallback applied for model {model_id}")
