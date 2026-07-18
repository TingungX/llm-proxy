"""Responses→Chat 转换步骤 — 将 Responses API 请求转为 Chat Completions 格式"""

import logging

from llm_proxy.handlers.base import PipelineContext, HandlerStep
from llm_proxy.protocol.effort_mapping import apply_effort_mapping
from llm_proxy.protocol.provider_profiles import (
    apply_thinking_to_chat_body,
    get_provider_profile,
)
from llm_proxy.protocol.responses_chat.request import (
    CodexToolSpec,
    convert_input_to_messages,
    convert_tools_to_chat,
)
from llm_proxy.state import get_state

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
        _, _, actual_model, model_id, _, _ = ctx.resolved

        input_data = body.get("input", [])
        instructions = body.get("instructions")

        if isinstance(input_data, list):
            logger.debug(f"Responses input_items={len(input_data)}")
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
                    logger.debug(f"  input[{idx}]: type={t}, role={item.get('role', '-')}, content_len={clen}")

        if instructions:
            logger.debug(f"  instructions_len={len(instructions)}")

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
        # 因此 tool_spec_map 必须在 if 块外初始化为 {}，否则函数末尾的
        # `ctx.tool_spec_map = tool_spec_map or None` 会抛 UnboundLocalError
        # （参考 /private/tmp/llm-proxy.log 16:42 / 17:00 的 30+ 6-item 500 链）。
        tool_spec_map: dict[str, CodexToolSpec] = {}
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
                    logger.debug(f"  RAW tool: type={tt}, name={tn}{sub_names}")
            chat_tools, reverse_tool_map, tool_spec_map = convert_tools_to_chat(body["tools"])
            if chat_tools:
                chat_body["tools"] = chat_tools
            if reverse_tool_map:
                logger.debug(f"Custom tools replaced: {list(reverse_tool_map.keys())}")
            if tool_spec_map:
                ns_entries = {k: v for k, v in tool_spec_map.items() if v.kind == "namespace"}
                if ns_entries:
                    logger.debug(f"Namespace tools mapped: {ns_entries}")

        # tool_choice 透传
        if body.get("tool_choice"):
            chat_body["tool_choice"] = body["tool_choice"]

        # reasoning.effort 映射（按模型 preset）
        reasoning = body.get("reasoning")
        mapped_effort: str | None = None
        model_effort_mapping = get_state().get_model_effort_mapping(model_id)
        if isinstance(reasoning, dict) and "effort" in reasoning:
            effort = reasoning["effort"]
            mapped_effort = apply_effort_mapping(effort, model_effort_mapping)
            # 未命中规则且无 "*" 兜底时回退到 "auto"（与旧硬编码 map.get(effort, "auto") 一致）
            chat_body["reasoning_effort"] = mapped_effort if mapped_effort else "auto"

        # 按厂商 profile 编码 thinking/reasoning（含 MiniMax reasoning_split）
        profile = get_provider_profile(
            get_state().get_model_provider(model_id), get_state().provider_profiles
        )
        apply_thinking_to_chat_body(
            chat_body,
            mapped_effort or (reasoning.get("effort") if isinstance(reasoning, dict) else None),
            profile,
        )

        # 流式请求
        stream = body.get("stream", False)
        if stream:
            chat_body["stream"] = True
            chat_body["stream_options"] = {"include_usage": True}

        ctx.body = chat_body
        ctx.reverse_tool_map = reverse_tool_map or None
        ctx.tool_spec_map = tool_spec_map or None

        logger.debug(f"Converted Responses → Chat: {len(messages)} messages, "
                     f"tools={len(chat_body.get('tools', []))}, "
                     f"reasoning_effort={chat_body.get('reasoning_effort')}")
