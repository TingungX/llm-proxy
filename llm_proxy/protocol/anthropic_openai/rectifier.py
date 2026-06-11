"""Thinking Signature 整流器

自动修复 Anthropic API 中因签名校验失败导致的请求错误。
当上游返回签名相关错误时，移除有问题的 thinking/signature 字段并重试。

参考: CCS (cc-switch) proxy/thinking_rectifier.rs
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RectifyResult:
    """整流结果"""
    applied: bool = False
    removed_thinking_blocks: int = 0
    removed_redacted_thinking_blocks: int = 0
    removed_signature_fields: int = 0
    removed_top_level_thinking: bool = False


# ── 错误检测 ──────────────────────────────────────────────────────────

def should_rectify(error_message: str | None) -> bool:
    """检测错误信息是否需要触发 thinking 签名整流器

    检测 7 种错误模式:
    1. Invalid 'signature' in 'thinking' block
    2. Thought signature is not valid (Gemini/第三方渠道)
    3. must start with a thinking block
    4. Expected thinking or redacted_thinking, but found tool_use
    5. signature: Field required
    6. signature: Extra inputs are not permitted
    7. thinking or redacted_thinking blocks ... cannot be modified
    """
    if not error_message:
        return False

    lower = error_message.lower()

    # 1. Invalid signature in thinking block
    if ("invalid" in lower and "signature" in lower
            and "thinking" in lower and "block" in lower):
        return True

    # 2. Thought signature is not valid
    if "thought signature" in lower and ("not valid" in lower or "invalid" in lower):
        return True

    # 3. must start with a thinking block
    if "must start with a thinking block" in lower:
        return True

    # 4. Expected thinking or redacted_thinking, but found tool_use
    if ("expected" in lower
            and ("thinking" in lower or "redacted_thinking" in lower)
            and "found" in lower
            and "tool_use" in lower):
        return True

    # 5. signature: Field required
    if "signature" in lower and "field required" in lower:
        return True

    # 6. signature: Extra inputs are not permitted
    if "signature" in lower and "extra inputs are not permitted" in lower:
        return True

    # 7. thinking/redacted_thinking cannot be modified
    if ("thinking" in lower or "redacted_thinking" in lower) and "cannot be modified" in lower:
        return True

    return False


# ── 请求体整流 ────────────────────────────────────────────────────────

def rectify_request(body: dict) -> RectifyResult:
    """对 Anthropic 请求体做最小侵入整流

    - 移除 messages[*].content 中的 thinking/redacted_thinking block
    - 移除非 thinking block 上遗留的 signature 字段
    - 特定条件下删除顶层 thinking 字段

    Args:
        body: Anthropic 请求体（会被原地修改）

    Returns:
        RectifyResult 整流结果
    """
    result = RectifyResult()

    messages = body.get("messages")
    if not isinstance(messages, list):
        return result

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        new_content = []
        modified = False

        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else ""

            # 移除 thinking blocks
            if block_type == "thinking":
                result.removed_thinking_blocks += 1
                modified = True
                continue

            # 移除 redacted_thinking blocks
            if block_type == "redacted_thinking":
                result.removed_redacted_thinking_blocks += 1
                modified = True
                continue

            # 移除非 thinking block 上的 signature 字段
            if isinstance(block, dict) and "signature" in block:
                block = dict(block)
                block.pop("signature", None)
                result.removed_signature_fields += 1
                modified = True

            new_content.append(block)

        if modified:
            msg["content"] = new_content
            result.applied = True

    # 兜底：thinking.type=enabled + 最后一条 assistant 消息不以 thinking 开头 + 有 tool_use
    if _should_remove_top_level_thinking(body):
        if "thinking" in body:
            body.pop("thinking")
            result.removed_top_level_thinking = True
            result.applied = True

    return result


def _should_remove_top_level_thinking(body: dict) -> bool:
    """判断是否需要删除顶层 thinking 字段

    条件:
    - thinking.type == "enabled"（adaptive 不删）
    - 最后一条 assistant 消息的 content[0] 不是 thinking/redacted_thinking
    - 且该消息包含 tool_use
    """
    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        return False

    thinking_type = thinking.get("type")
    if thinking_type != "enabled":
        return False

    # 找最后一条 assistant 消息
    messages = body.get("messages", [])
    last_assistant = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_assistant = msg
            break

    if not last_assistant:
        return False

    content = last_assistant.get("content")
    if not isinstance(content, list) or not content:
        return False

    # 首块类型
    first_type = content[0].get("type") if isinstance(content[0], dict) else ""
    if first_type in ("thinking", "redacted_thinking"):
        return False

    # 是否包含 tool_use
    has_tool_use = any(
        isinstance(b, dict) and b.get("type") == "tool_use"
        for b in content
    )
    return has_tool_use

