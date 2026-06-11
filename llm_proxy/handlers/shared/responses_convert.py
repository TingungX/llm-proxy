"""Responses→Chat 转换步骤 — 将 Responses API 请求转为 Chat Completions 格式"""

import logging

from llm_proxy.handlers.base import PipelineContext, HandlerStep
from llm_proxy.protocol.responses_chat.request import (
    convert_input_to_messages,
    convert_tools_to_chat,
)

logger = logging.getLogger(__name__)


class ResponsesConvertStep(HandlerStep):
    """Responses→Chat 转换步骤

    仅在 converter == "responses_to_chat" 时执行：
    - 将 Responses API input 转为 Chat messages
    - 将 Responses API tools 转为 Chat function tools
    - 保留 reasoning.effort、tool_choice 等参数
    - 构建完整 Chat Completions 请求体
    """

    async def execute(self, ctx: PipelineContext) -> None:
        if ctx.converter != "responses_to_chat":
            return  # 不需要转换

        body = ctx.body
        _, _, actual_model, _, _, _ = ctx.resolved

        input_data = body.get("input", [])
        instructions = body.get("instructions")

        if isinstance(input_data, list):
            logger.info(f"Responses input_items={len(input_data)}")
            for idx, item in enumerate(input_data[:3]):
                if isinstance(item, dict):
                    t = item.get('type', '?')
                    c = item.get('content', '')
                    if isinstance(c, str):
                        clen = len(c)
                    elif isinstance(c, list):
                        clen = sum(len(p.get('text', '')) for p in c if isinstance(p, dict))
                    else:
                        clen = 0
                    logger.info(f"  input[{idx}]: type={t}, role={item.get('role', '-')}, content_len={clen}")

        if instructions:
            logger.info(f"  instructions_len={len(instructions)}")

        messages = convert_input_to_messages(input_data, instructions)

        chat_body = {"model": actual_model, "messages": messages}
        if "max_output_tokens" in body and body["max_output_tokens"] is not None:
            chat_body["max_tokens"] = body["max_output_tokens"]
        for key in ("temperature", "top_p"):
            if key in body:
                chat_body[key] = body[key]

        # Tool 转换
        reverse_tool_map: dict[str, str] = {}
        # Codex 在 compact 完成后的下一轮请求会省略 `tools` 字段；
        # 因此 namespace_map 必须在 if 块外初始化为 {}，否则函数末尾的
        # `ctx.namespace_map = namespace_map or None` 会抛 UnboundLocalError
        # （参考 /private/tmp/llm-proxy.log 16:42 / 17:00 的 30+ 6-item 500 链）。
        namespace_map: dict[str, str] = {}
        if body.get("tools"):
            # DEBUG: 记录原始工具列表的类型和名称
            for t in body["tools"]:
                if isinstance(t, dict):
                    tt = t.get('type','?')
                    tn = t.get('name','?')
                    sub_names = ""
                    if tt == "namespace":
                        subs = t.get("tools") or []
                        sub_names = ", subtools=[" + "; ".join(
                            f"name={s.get('name','?')},type={s.get('type','?')},defer={s.get('deferLoading',s.get('defer_loading','?'))}" 
                            for s in subs if isinstance(s, dict)
                        ) + "]"
                    logger.info(f"  RAW tool: type={tt}, name={tn}{sub_names}")
            chat_tools, reverse_tool_map, namespace_map = convert_tools_to_chat(body["tools"])
            if chat_tools:
                chat_body["tools"] = chat_tools
            if reverse_tool_map:
                logger.info(f"Custom tools replaced: {list(reverse_tool_map.keys())}")
            if namespace_map:
                logger.info(f"Namespace tools mapped: {namespace_map}")

        # tool_choice 透传
        if body.get("tool_choice"):
            chat_body["tool_choice"] = body["tool_choice"]

        # reasoning.effort 映射
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict) and "effort" in reasoning:
            effort = reasoning["effort"]
            effort_map = {
                "none": "none",
                "auto": "auto",
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
            }
            chat_body["reasoning_effort"] = effort_map.get(effort, "auto")

        # 流式请求
        stream = body.get("stream", False)
        if stream:
            chat_body["stream"] = True
            chat_body["stream_options"] = {"include_usage": True}

        ctx.body = chat_body
        ctx.reverse_tool_map = reverse_tool_map or None
        ctx.namespace_map = namespace_map or None

        logger.info(f"Converted Responses → Chat: {len(messages)} messages, "
                     f"tools={len(chat_body.get('tools', []))}, "
                     f"reasoning_effort={chat_body.get('reasoning_effort')}")
