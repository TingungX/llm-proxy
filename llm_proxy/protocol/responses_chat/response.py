"""Chat Completions → Responses API 转换适配器

将 Chat Completions 格式的请求转换为 Responses API 格式，
并将 Responses API 响应转换回 Chat Completions 格式。
"""

import json
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


def convert_chat_to_responses_request(chat_body: dict) -> dict:
    """将 Chat Completions 请求转换为 Responses API 格式。

    转换规则：
    - messages → input（保留完整对话历史）
    - system 消息 → instructions
    - max_tokens → max_output_tokens
    - temperature, top_p, stream 保持不变

    Args:
        chat_body: Chat Completions API 的请求体

    Returns:
        Responses API 格式的请求体
    """
    result = {"model": chat_body.get("model")}

    messages = chat_body.get("messages", [])

    # 提取 system 消息作为 instructions
    instructions = None
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system" and instructions is None:
            instructions = msg.get("content", "")
        else:
            filtered_messages.append(msg)

    if instructions:
        result["instructions"] = instructions

    # 转换 messages → input
    if len(filtered_messages) == 1 and filtered_messages[0].get("role") == "user":
        # 单条 user 消息：直接使用字符串
        content = filtered_messages[0].get("content", "")
        if isinstance(content, str):
            result["input"] = content
        else:
            result["input"] = filtered_messages
    else:
        # 多条消息：保留完整对话历史
        result["input"] = filtered_messages

    # 参数映射
    if "max_tokens" in chat_body:
        result["max_output_tokens"] = chat_body["max_tokens"]

    for key in ("temperature", "top_p", "stream"):
        if key in chat_body:
            result[key] = chat_body[key]

    return result


def convert_responses_to_chat_response(resp_body: dict, original_model: str) -> dict:
    """将 Responses API 响应转换为 Chat Completions 格式。

    转换规则：
    - output[0].content → choices[0].message.content
    - usage.input_tokens → usage.prompt_tokens
    - usage.output_tokens → usage.completion_tokens

    Args:
        resp_body: Responses API 的响应体
        original_model: 原始请求的模型名

    Returns:
        Chat Completions 格式的响应体
    """
    outputs = resp_body.get("output", [])
    output = outputs[0] if outputs else {}

    # 提取文本内容
    content_parts = output.get("content", [])
    text_content = ""
    for part in content_parts:
        if part.get("type") == "output_text":
            text_content += part.get("text", "")

    usage = resp_body.get("usage", {})

    return {
        "id": resp_body.get("id", ""),
        "object": "chat.completion",
        "created": resp_body.get("created", 0),
        "model": original_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def stream_responses_to_chat(
    resp,
    model: str,
    endpoint_id: str,
    model_id: str,
    request_id: str = "",
    client_ip: str = "",
    user_agent: str = "",
) -> AsyncGenerator[bytes, None]:
    """将 Responses API SSE 流转换为 Chat Completions SSE 流。

    事件映射：
    - response.output_text.delta → choices[0].delta.content
    - response.done → data: [DONE]

    Args:
        resp: httpx 流式响应对象
        model: 原始模型名
        endpoint_id: 端点 ID（用于用量记录）
        model_id: 模型 ID（用于用量记录）
    """
    from llm_proxy.infra import db  # 延迟导入避免循环依赖

    usage = {"input_tokens": 0, "output_tokens": 0}
    tc_id_to_index: dict[str, int] = {}
    had_error = False

    try:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                yield b"data: [DONE]\n\n"
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            # 处理 usage
            if "usage" in event:
                u = event["usage"]
                usage["input_tokens"] = u.get("input_tokens", 0)
                usage["output_tokens"] = u.get("output_tokens", 0)

            # 转换事件
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                chunk = {
                    "id": event.get("id", ""),
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None,
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif event_type == "response.function_call.started":
                call_id = event.get("id", "")
                func_name = event.get("name", "")
                idx = tc_id_to_index.setdefault(call_id, len(tc_id_to_index))
                chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "id": call_id,
                                "function": {"name": func_name, "arguments": ""},
                            }]
                        },
                        "finish_reason": None,
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("id", "")
                args_delta = event.get("delta", "")
                idx = tc_id_to_index.get(call_id, 0)
                chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "function": {"arguments": args_delta},
                            }]
                        },
                        "finish_reason": None,
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif event_type == "response.function_call.done":
                call_id = event.get("id", "")
                func_name = event.get("name", "")
                args = event.get("arguments", "")
                idx = tc_id_to_index.get(call_id, 0)
                chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "id": call_id,
                                "function": {"name": func_name, "arguments": args},
                            }]
                        },
                        "finish_reason": "tool_calls",
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif event_type in ("response.done", "response.completed"):
                yield b"data: [DONE]\n\n"

            elif event_type == "response.error":
                had_error = True
                logger.error(f"Upstream error: {event.get('error')}")
                yield f"data: {json.dumps({'error': event.get('error')})}\n\n".encode()

    except Exception as e:
        had_error = True
        logger.error(f"Stream error: {e}", exc_info=True)
        error_chunk = {
            "error": {"message": str(e) or type(e).__name__, "type": "proxy_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n".encode()
    finally:
        # 记录用量
        status = "error" if had_error else "success"
        if usage["input_tokens"] > 0:
            db.record_usage(
                endpoint_id=endpoint_id,
                model_id=model_id,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                status=status,
                error_type=status if status == "error" else None,
                request_id=request_id,
                client_ip=client_ip,
                user_agent=user_agent,
            )
        else:
            db.record_usage(
                endpoint_id=endpoint_id,
                model_id=model_id,
                input_tokens=1,
                output_tokens=0,
                status=status,
                error_type=status if status == "error" else None,
                request_id=request_id,
                client_ip=client_ip,
                user_agent=user_agent,
            )
