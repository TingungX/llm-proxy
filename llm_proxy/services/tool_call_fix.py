"""修复对话历史中 tool_calls 与 tool response 的不匹配问题。

Codex 等客户端可能在中断 tool call 后保留 history，导致：
1. assistant message 带 tool_calls 但缺少对应 tool response（orphaned）
2. tool response 出现在错误的 assistant 之后（misplaced）

DeepSeek 等上游严格校验：每个 assistant(tool_calls) 必紧接对应 tool response，
且 tool response 不能归属到更早的 assistant。本模块在发送到上游前统一修复。

参考 ccx 的做法：孤立 tool response 降级为 user 消息保留上下文，
而非直接删除（DeepSeek 等上游会拒绝孤立的 tool 消息）。
"""

import logging

logger = logging.getLogger(__name__)

_PLACEHOLDER_CONTENT = "[Tool call was interrupted]"


def _downgrade_orphan_tool_message(msg: dict) -> dict:
    """将孤立的 tool 消息降级为 user 消息，保留上下文而非删除。

    DeepSeek 等上游拒绝孤立的 tool 消息；降级为普通历史文本可保留上下文。
    """
    call_id = msg.get("tool_call_id", "")
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    else:
        text = str(content) if content else ""
    if call_id:
        text = f"Function call output ({call_id}): {text}"
    else:
        text = f"Function call output: {text}"
    return {"role": "user", "content": text}


def fix_orphaned_tool_calls(messages: list[dict]) -> list[dict]:
    """修复 messages 中 tool_calls/tool response 不匹配，返回新列表。

    修复策略（参考 ccx downgradeOrphanOpenAIToolMessages）：
    1. 收集每个 assistant(tool_calls) 的 required_ids，将紧跟的 tool response
       按归属分配：属于当前 assistant 的保留，misplaced 的延后处理。
    2. 对缺失 response 的 tool_call_id 插入占位符。
    3. 对错位的 tool response 降级为 user 消息保留上下文（而非直接删除）。

    最终保证：每个 assistant(tool_calls) 后紧跟所有对应 tool response，
    不多不少。孤立 tool 不会直接丢弃，保留上下文给上游。

    Args:
        messages: OpenAI Chat Completions 格式的 messages 列表

    Returns:
        修复后的 messages 列表（新列表，不修改原列表）
    """
    if not messages:
        return messages

    result = []
    i = 0
    placeholders_inserted = 0
    misplaced_downgraded = 0
    pending_orphans: list[dict] = []

    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for orphan in pending_orphans:
                    result.append(orphan)
                    misplaced_downgraded += 1
                pending_orphans = []

                result.append(msg)
                required_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
                if not required_ids:
                    i += 1
                    continue

                responded_ids = set()
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tc_id = messages[j].get("tool_call_id")
                    if tc_id in required_ids:
                        responded_ids.add(tc_id)
                        result.append(messages[j])
                    else:
                        pending_orphans.append(_downgrade_orphan_tool_message(messages[j]))
                        logger.debug(
                            "Misplaced tool response for %s after assistant requiring %s, downgrading",
                            tc_id, required_ids,
                        )
                    j += 1

                missing_ids = required_ids - responded_ids
                if missing_ids:
                    for mid in sorted(missing_ids):
                        placeholder = {
                            "role": "tool",
                            "tool_call_id": mid,
                            "content": _PLACEHOLDER_CONTENT,
                        }
                        result.append(placeholder)
                        placeholders_inserted += 1

                i = j
            else:
                for orphan in pending_orphans:
                    result.append(orphan)
                    misplaced_downgraded += 1
                pending_orphans = []
                result.append(msg)
                i += 1
        elif msg.get("role") == "tool":
            pending_orphans.append(_downgrade_orphan_tool_message(msg))
            logger.debug(
                "Orphan tool response for %s without preceding assistant(tool_calls), downgrading",
                msg.get("tool_call_id", ""),
            )
            i += 1
        else:
            for orphan in pending_orphans:
                result.append(orphan)
                misplaced_downgraded += 1
            pending_orphans = []
            result.append(msg)
            i += 1

    for orphan in pending_orphans:
        result.append(orphan)
        misplaced_downgraded += 1

    if placeholders_inserted or misplaced_downgraded:
        logger.warning(
            "Fixed tool_calls: inserted %d placeholder(s), downgraded %d orphan response(s)",
            placeholders_inserted, misplaced_downgraded,
        )

    return result
