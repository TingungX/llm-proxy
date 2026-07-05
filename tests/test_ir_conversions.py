"""统一协议 IR 抽象层 — 单元测试。

测试覆盖：
1. 三个协议的 to_ir / to_upstream 双向
2. 跨协议直接转换（Anthropic ↔ Responses）
3. 响应方向（response_to_ir / response_from_ir）
4. 特殊场景：apply_patch 展开、think 标签提取、cache_control 保留、tool_choice 映射
"""

import json
import pytest

from llm_proxy.protocol.ir import (
    REGISTRY,
    convert_request,
    convert_response,
)
from llm_proxy.protocol.ir.anthropic import (
    to_ir as anthropic_to_ir,
    to_upstream as anthropic_to_upstream,
    response_from_ir as anthropic_response_from_ir,
    response_to_ir as anthropic_response_to_ir,
)
from llm_proxy.protocol.ir.chat import (
    to_ir as chat_to_ir,
    to_upstream as chat_to_upstream,
    response_from_ir as chat_response_from_ir,
    response_to_ir as chat_response_to_ir,
)
from llm_proxy.protocol.ir.responses import (
    to_ir as responses_to_ir,
    to_upstream as responses_to_upstream,
    response_from_ir as responses_response_from_ir,
    response_to_ir as responses_response_to_ir,
)
from llm_proxy.protocol.ir.types import (
    IRImageBlock,
    IRMessage,
    IRRedactedThinkingBlock,
    IRRequest,
    IRResponse,
    IRTextBlock,
    IRThinkingBlock,
    IRToolDef,
    IRToolResultBlock,
    IRToolUseBlock,
)

# ── Anthropic → IR → Chat（与 anthropic_to_chat 行为对齐）────


class TestAnthropicIRRoundtripRequest:
    """Anthropic → IR → Chat 应与原 anthropic_to_chat() 输出一致。"""

    def test_basic_text_messages(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What's 2+2?"},
            ],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)

        assert chat_body["model"] == "claude-3-5-sonnet"
        assert chat_body["max_tokens"] == 1024
        assert len(chat_body["messages"]) == 3
        assert chat_body["messages"][0] == {"role": "user", "content": "Hello"}
        assert chat_body["messages"][1] == {"role": "assistant", "content": "Hi there!"}
        assert chat_body["messages"][2] == {"role": "user", "content": "What's 2+2?"}

    def test_top_level_system_string(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert chat_body["messages"][1] == {"role": "user", "content": "Hi"}

    def test_top_level_system_list_with_cache_control(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Always be concise."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        ir = anthropic_to_ir(body)
        # system 在 IR 中合并到 extensions
        assert "system_cache_control" in ir.extensions
        # 转换到 Chat 时，由于 system 来自 list 形式，预期会保留 cache_control
        chat_body = chat_to_upstream(ir)
        sys_msg = chat_body["messages"][0]
        assert sys_msg["role"] == "system"
        assert "You are helpful." in sys_msg["content"]
        assert "Always be concise." in sys_msg["content"]

    def test_thinking_block_converts_to_reasoning_content(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "User said hi."},
                    {"type": "text", "text": "Hello!"},
                ]},
            ],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        # 找 assistant 消息
        assistant_msg = next(m for m in chat_body["messages"] if m["role"] == "assistant")
        assert assistant_msg.get("reasoning_content") == "User said hi."
        assert assistant_msg["content"] == "Hello!"

    def test_tool_use_and_tool_result(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 200,
            "messages": [
                {"role": "user", "content": "What's the weather in SF?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "San Francisco"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": "72°F, sunny"},
                ]},
            ],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        roles = [m["role"] for m in chat_body["messages"]]
        assert roles == ["user", "assistant", "tool"]
        tool_msg = chat_body["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_1"
        assert tool_msg["content"] == "72°F, sunny"
        # assistant 应有 tool_calls
        assistant = chat_body["messages"][1]
        assert "tool_calls" in assistant
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
            "city": "San Francisco"
        }

    def test_image_block_converts_to_image_url(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "abc123"
                    }},
                ]},
            ],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        user_msg = chat_body["messages"][0]
        assert user_msg["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert user_msg["content"][1]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in user_msg["content"][1]["image_url"]["url"]

    def test_tool_choice_any_maps_to_required(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "any",
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["tool_choice"] == "required"

    def test_tool_choice_tool_maps_to_function(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_billing_header_stripped(self):
        """billing header 包含 rotating cch= 值，必须剥离。"""
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "system": "x-anthropic-billing-header:cch=abc123\nYou are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        sys_msg = chat_body["messages"][0]
        # billing header 不应在 system 中
        assert "x-anthropic-billing-header" not in sys_msg["content"]
        assert "You are helpful." in sys_msg["content"]

    def test_stop_sequences(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["END", "STOP"],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["stop"] == ["END", "STOP"]

    def test_batch_tool_filtered(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {"type": "BatchTool", "name": "ignored"},
                {"name": "kept", "description": "kept", "input_schema": {}},
            ],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert len(chat_body["tools"]) == 1
        assert chat_body["tools"][0]["function"]["name"] == "kept"


# ── Responses → IR → Chat ─────────────────────────────────────


class TestResponsesIRRoundtripRequest:
    """Responses → IR → Chat 行为。"""

    def test_simple_string_input(self):
        body = {
            "model": "gpt-5",
            "input": "What is 2+2?",
            "max_output_tokens": 100,
        }
        ir = responses_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["model"] == "gpt-5"
        assert chat_body["max_tokens"] == 100
        assert chat_body["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_instructions_become_system(self):
        body = {
            "model": "gpt-5",
            "instructions": "Be concise.",
            "input": "Hi",
        }
        ir = responses_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["messages"][0] == {"role": "system", "content": "Be concise."}
        assert chat_body["messages"][1] == {"role": "user", "content": "Hi"}

    def test_input_array_with_messages(self):
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Hi"},
                ]},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Hello!"},
                ]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "How are you?"},
                ]},
            ],
        }
        ir = responses_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert len(chat_body["messages"]) == 3
        assert chat_body["messages"][0] == {"role": "user", "content": "Hi"}
        assert chat_body["messages"][1] == {"role": "assistant", "content": "Hello!"}

    def test_reasoning_effort_mapping(self):
        body = {
            "model": "gpt-5",
            "input": "Hi",
            "reasoning": {"effort": "minimal"},
        }
        ir = responses_to_ir(body)
        assert ir.reasoning_effort == "low"

    def test_function_call_input(self):
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "What's 2+2?"},
                ]},
                {"type": "function_call", "call_id": "call_1",
                 "name": "add", "arguments": '{"a": 2, "b": 2}'},
                {"type": "function_call_output", "call_id": "call_1",
                 "output": "4"},
            ],
        }
        ir = responses_to_ir(body)
        chat_body = chat_to_upstream(ir)
        # 应有 user, assistant(tool_calls), tool
        roles = [m["role"] for m in chat_body["messages"]]
        assert roles == ["user", "assistant", "tool"]
        tool_msg = chat_body["messages"][2]
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "4"

    def test_orphaned_function_call_inserts_placeholder(self):
        """孤立 function_call（无匹配 function_call_output）应插入占位 tool result。

        连续 function_call 合并到同一条 assistant 消息，缺失 call_A 的 result 插入占位符。
        """
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Use tool A"},
                ]},
                {"type": "function_call", "call_id": "call_a",
                 "name": "tool_a", "arguments": '{"x": 1}'},
                # call_a 没有对应的 function_call_output
                {"type": "function_call", "call_id": "call_b",
                 "name": "tool_b", "arguments": '{"y": 2}'},
                {"type": "function_call_output", "call_id": "call_b",
                 "output": "done"},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Now what?"},
                ]},
            ],
        }
        ir = responses_to_ir(body)
        chat_body = chat_to_upstream(ir)
        roles = [m["role"] for m in chat_body["messages"]]
        # 连续 function_call 合并为同一条 assistant，call_a 缺 result → 占位
        assert roles == ["user", "assistant", "tool", "tool", "user"]
        assistant = chat_body["messages"][1]
        assert assistant["role"] == "assistant"
        assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_a", "call_b"]
        # call_a 的占位符（先出现，因为有缺失）
        assert chat_body["messages"][2]["role"] == "tool"
        assert chat_body["messages"][2]["tool_call_id"] == "call_a"
        assert chat_body["messages"][2]["content"] == "[Tool call was interrupted]"
        # call_b 的正常 result
        assert chat_body["messages"][3]["role"] == "tool"
        assert chat_body["messages"][3]["tool_call_id"] == "call_b"
        assert chat_body["messages"][3]["content"] == "done"

    def test_apply_patch_passthrough_in_ir(self):
        """apply_patch 透传：单 function tool + reverse_tool_map 自映射。"""
        body = {
            "model": "gpt-5",
            "input": "Add a file",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
            }],
        }
        ir = responses_to_ir(body)
        assert len(ir.tools) == 1
        assert ir.tools[0].name == "apply_patch"
        rtm = ir.extensions.get("reverse_tool_map", {})
        assert rtm.get("apply_patch") == "apply_patch"


# ── Anthropic → IR → Responses（核心：拼图最后一块）──


class TestAnthropicToResponsesRequest:
    """Anthropic → Responses 直接转换（不经过 Chat）。"""

    def test_simple_message(self):
        body = {
            "model": "claude-3-5-sonnet",
            "system": "You are helpful.",
            "max_tokens": 200,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hello!"},
                ]},
            ],
        }
        upstream = convert_request("anthropic", "openai/responses", body)
        assert upstream["model"] == "claude-3-5-sonnet"
        assert upstream["instructions"] == "You are helpful."
        assert upstream["max_output_tokens"] == 200
        # input 数组
        items = upstream["input"]
        assert len(items) >= 2
        # 第一条应是 user message
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"

    def test_thinking_becomes_reasoning(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Internal reasoning here."},
                    {"type": "text", "text": "Public answer."},
                ]},
            ],
        }
        upstream = convert_request("anthropic", "openai/responses", body)
        # 应有 reasoning item
        reasoning_items = [i for i in upstream["input"] if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert "Internal reasoning here." in reasoning_items[0]["summary"][0]["text"]
        # 应有 message item
        msg_items = [i for i in upstream["input"] if i.get("type") == "message"]
        assert any(
            m.get("role") == "assistant" and
            m.get("content", [{}])[0].get("text") == "Public answer."
            for m in msg_items
        )

    def test_image_becomes_input_image(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": "imgdata"
                    }},
                ]},
            ],
        }
        upstream = convert_request("anthropic", "openai/responses", body)
        user_msg = next(i for i in upstream["input"]
                        if i.get("type") == "message" and i.get("role") == "user")
        content_types = [c.get("type") for c in user_msg["content"]]
        assert "input_text" in content_types
        assert "input_image" in content_types

    def test_tool_choice_preserved(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "search"},
        }
        upstream = convert_request("anthropic", "openai/responses", body)
        # Anthropic {"type":"tool","name":"X"} → Responses 扁平形式
        assert upstream["tool_choice"] == {
            "type": "function",
            "name": "search",
        }


# ── Responses → IR → Anthropic ─────────────────────────────────


class TestResponsesToAnthropicRequest:
    """Responses → Anthropic 直接转换。"""

    def test_simple_input(self):
        body = {
            "model": "gpt-5",
            "instructions": "Be helpful.",
            "input": "Hi",
            "max_output_tokens": 100,
        }
        upstream = convert_request("openai/responses", "anthropic", body)
        assert upstream["model"] == "gpt-5"
        assert upstream["system"] == "Be helpful."
        assert upstream["max_tokens"] == 100
        assert len(upstream["messages"]) == 1
        assert upstream["messages"][0] == {"role": "user", "content": "Hi"}

    def test_apply_patch_passthrough_to_anthropic(self):
        """透传模式：Responses apply_patch → Anthropic 单个 apply_patch function tool。"""
        body = {
            "model": "gpt-5",
            "input": "Edit file",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply patch",
            }],
        }
        upstream = convert_request("openai/responses", "anthropic", body)
        tool_names = {t["name"] for t in upstream["tools"]}
        assert tool_names == {"apply_patch"}


# ── 响应方向 ─────────────────────────────────────────────────────


class TestChatToAnthropicResponse:
    def test_basic_response(self):
        chat_resp = {
            "id": "chatcmpl-123",
            "model": "gpt-5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        anth = convert_response("openai/chat-completions", "anthropic", chat_resp)
        assert anth["role"] == "assistant"
        assert anth["stop_reason"] == "end_turn"
        assert anth["content"] == [{"type": "text", "text": "Hello!"}]
        assert anth["usage"]["input_tokens"] == 10
        assert anth["usage"]["output_tokens"] == 5

    def test_think_tags_extracted(self):
        chat_resp = {
            "id": "x",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think>Reasoning</think>Public answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        anth = convert_response("openai/chat-completions", "anthropic", chat_resp)
        content_types = [b["type"] for b in anth["content"]]
        assert "thinking" in content_types
        assert "text" in content_types
        thinking = next(b for b in anth["content"] if b["type"] == "thinking")
        assert thinking["thinking"] == "Reasoning"
        text = next(b for b in anth["content"] if b["type"] == "text")
        assert text["text"] == "Public answer"

    def test_tool_calls_to_tool_use(self):
        chat_resp = {
            "id": "x",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        anth = convert_response("openai/chat-completions", "anthropic", chat_resp)
        assert anth["stop_reason"] == "tool_use"
        tool_use = next(b for b in anth["content"] if b["type"] == "tool_use")
        assert tool_use["id"] == "call_1"
        assert tool_use["name"] == "search"
        assert tool_use["input"] == {"q": "test"}


class TestResponsesToAnthropicResponse:
    def test_basic_response(self):
        resp_body = {
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Hello!"},
                ]},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        anth = convert_response("openai/responses", "anthropic", resp_body)
        assert anth["stop_reason"] == "end_turn"
        assert anth["content"] == [{"type": "text", "text": "Hello!"}]
        assert anth["usage"]["input_tokens"] == 10

    def test_function_call_to_tool_use(self):
        resp_body = {
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Let me check."},
                ]},
                {"type": "function_call", "id": "call_1", "call_id": "call_1",
                 "name": "search", "arguments": '{"q": "test"}'},
            ],
            "usage": {},
        }
        anth = convert_response("openai/responses", "anthropic", resp_body)
        assert anth["stop_reason"] == "tool_use"
        tool_uses = [b for b in anth["content"] if b["type"] == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "search"
        assert tool_uses[0]["input"] == {"q": "test"}


class TestAnthropicResponseToChat:
    def test_basic_response(self):
        anth_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        chat = convert_response("anthropic", "openai/chat-completions", anth_resp)
        assert chat["choices"][0]["message"]["content"] == "Hello!"
        assert chat["choices"][0]["finish_reason"] == "stop"
        assert chat["usage"]["prompt_tokens"] == 10
        assert chat["usage"]["completion_tokens"] == 5

    def test_thinking_block_to_reasoning_content(self):
        anth_resp = {
            "id": "x",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Internal thought"},
                {"type": "text", "text": "Public reply"},
            ],
            "stop_reason": "end_turn",
            "usage": {},
        }
        chat = convert_response("anthropic", "openai/chat-completions", anth_resp)
        msg = chat["choices"][0]["message"]
        assert msg["content"] == "Public reply"
        assert msg.get("reasoning_content") == "Internal thought"

    def test_tool_use_to_tool_calls(self):
        anth_resp = {
            "id": "x",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "search",
                 "input": {"q": "test"}},
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        chat = convert_response("anthropic", "openai/chat-completions", anth_resp)
        msg = chat["choices"][0]["message"]
        assert chat["choices"][0]["finish_reason"] == "tool_calls"
        assert msg["tool_calls"][0]["id"] == "toolu_1"
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "test"}


# ── IR 类型直接测试 ─────────────────────────────────────────────


class TestIRRequestDataclass:
    def test_default_values(self):
        ir = IRRequest()
        assert ir.model == ""
        assert ir.messages == []
        assert ir.system_prompt is None
        assert ir.tools is None
        assert ir.stream is False
        assert ir.extensions == {}

    def test_message_construction(self):
        msg = IRMessage(role="user", content="Hi")
        assert msg.role == "user"
        assert msg.content == "Hi"

    def test_content_blocks(self):
        ir = IRRequest(
            model="m",
            messages=[
                IRMessage(role="user", content=[
                    IRTextBlock(text="Look"),
                    IRImageBlock(base64_data="x", media_type="image/png"),
                ]),
                IRMessage(role="assistant", content=[
                    IRThinkingBlock(thinking="..."),
                    IRTextBlock(text="Answer"),
                ]),
                IRMessage(role="tool", content=[
                    IRToolResultBlock(tool_use_id="t1", content="result"),
                ]),
            ],
        )
        assert len(ir.messages) == 3
        assert isinstance(ir.messages[0].content[0], IRTextBlock)
        assert isinstance(ir.messages[0].content[1], IRImageBlock)
        assert isinstance(ir.messages[1].content[0], IRThinkingBlock)
        assert isinstance(ir.messages[2].content[0], IRToolResultBlock)


# ── 特殊场景 ─────────────────────────────────────────────────────


class TestApplyPatchExpansion:
    def test_apply_patch_passthrough_to_ir(self):
        """透传：Responses apply_patch → 1 个 IRToolDef + reverse_tool_map 自映射。"""
        body = {
            "model": "gpt-5",
            "input": "Edit",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply patch",
            }],
        }
        ir = responses_to_ir(body)
        assert len(ir.tools) == 1
        assert ir.tools[0].name == "apply_patch"
        rtm = ir.extensions["reverse_tool_map"]
        assert rtm.get("apply_patch") == "apply_patch"

    def test_4_tools_no_longer_collapse_to_apply_patch(self):
        """反向：4 个独立 function tool（不是 apply_patch 派生）→ Responses function tools，透传模式不再塌缩。"""
        body = {
            "model": "gpt-5",
            "input": "Edit",
            "tools": [
                {"type": "function", "name": "write_to_file",
                 "description": "Write", "parameters": {}},
                {"type": "function", "name": "replace_in_file",
                 "description": "Replace", "parameters": {}},
            ],
        }
        ir = chat_to_ir(body)
        assert len(ir.tools) == 2
        # 转回 Responses：透传模式不再塌缩为 apply_patch
        upstream = responses_to_upstream(ir)
        upstream_names = {t["name"] for t in upstream["tools"]}
        assert upstream_names == {"write_to_file", "replace_in_file"}


class TestThinkTagExtraction:
    def test_think_tags_stripped_from_string_content(self):
        """Chat 响应含 <think> 标签 → IRThinkingBlock + IRTextBlock。"""
        chat_resp = {
            "id": "x",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think>Let me think</think>The answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        ir = chat_response_to_ir(chat_resp)
        assert len(ir.content_blocks) == 2
        assert isinstance(ir.content_blocks[0], IRThinkingBlock)
        assert ir.content_blocks[0].thinking == "Let me think"
        assert isinstance(ir.content_blocks[1], IRTextBlock)
        assert ir.content_blocks[1].text == "The answer"


class TestCacheControlPreservation:
    def test_cache_control_preserved_in_chat(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Be helpful.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        sys_msg = chat_body["messages"][0]
        assert sys_msg.get("cache_control") == {"type": "ephemeral"}

    def test_cache_control_preserved_on_tool(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "search",
                "description": "Search",
                "input_schema": {},
                "cache_control": {"type": "ephemeral"},
            }],
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["tools"][0]["cache_control"] == {"type": "ephemeral"}


class TestToolChoiceMapping:
    @pytest.mark.parametrize("input_choice,expected", [
        ("auto", "auto"),
        ("none", "none"),
        ("any", "required"),
    ])
    def test_anthropic_string_choices(self, input_choice, expected):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": input_choice,
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["tool_choice"] == expected

    def test_anthropic_dict_choice(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "search"},
        }
        ir = anthropic_to_ir(body)
        chat_body = chat_to_upstream(ir)
        assert chat_body["tool_choice"] == {
            "type": "function", "function": {"name": "search"},
        }


# ── 端到端双向一致性 ─────────────────────────────────────────────


class TestRoundTripStability:
    """Anthropic → IR → Chat → IR → Chat 应与原 Chat 体一致（双向稳定）。"""

    def test_anthropic_to_chat_then_back_to_chat(self):
        original = {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{
                     "id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q": "test"}'},
                 }]},
                {"role": "tool", "tool_call_id": "c1", "content": "found"},
            ],
        }
        # Chat → IR → Chat
        ir = chat_to_ir(original)
        again = chat_to_upstream(ir)
        # 验证关键字段
        assert again["model"] == original["model"]
        assert len(again["messages"]) == len(original["messages"])
        # system 应被抽取为 IRRequest.system_prompt
        assert ir.system_prompt == "Be helpful."
        # 重新生成时 system 会被放回 messages[0]
        assert again["messages"][0]["role"] == "system"
        assert again["messages"][0]["content"] == "Be helpful."


class TestRegistryConsistency:
    def test_all_three_protocols_registered(self):
        assert "anthropic" in REGISTRY
        assert "openai/chat-completions" in REGISTRY
        assert "openai/responses" in REGISTRY

    def test_openai_alias_resolves(self):
        """历史别名 'openai' → 'openai/chat-completions'."""
        # 通过 convert_request 间接验证
        result = convert_request("openai", "openai/responses", {
            "model": "gpt-5",
            "input": "Hi",
        })
        assert result["model"] == "gpt-5"


# ── 回归测试（review 修复点）───────────────────────────────────────


class TestRedactedThinkingRoundTrip:
    """N1 修复：redacted_thinking 在 IR ↔ Anthropic 双向必须保留。"""

    def test_response_to_ir_recognizes_redacted_thinking(self):
        """Anthropic 响应含 redacted_thinking → IRRedactedThinkingBlock（不丢 data）。"""
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "redacted_thinking", "data": "encrypted_blob_xyz"},
            ],
            "stop_reason": "end_turn",
            "model": "claude",
        }
        ir = anthropic_response_to_ir(body)
        types = [b.type for b in ir.content_blocks]
        assert "redacted_thinking" in types
        # data 必须保留
        redacted = next(b for b in ir.content_blocks if b.type == "redacted_thinking")
        assert redacted.data == "encrypted_blob_xyz"

    def test_response_from_ir_emits_redacted_thinking(self):
        """IRRedactedThinkingBlock → Anthropic 响应的 redacted_thinking block。"""
        ir = IRResponse(
            id="msg_1",
            model="claude",
            content_blocks=[
                IRTextBlock(text="hi"),
                IRRedactedThinkingBlock(data="encrypted_blob_xyz"),
            ],
            stop_reason="end_turn",
        )
        body = anthropic_response_from_ir(ir)
        blocks = body["content"]
        redacted = [b for b in blocks if b["type"] == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "encrypted_blob_xyz"


# ── 第二轮 review 修复点（边缘情况）───────────────────────────────────


class TestToolChoiceRequiredToAnthropic:
    """Chat 'required' → Anthropic 必须是 'any'（不能透传 'required'）。"""

    def test_required_string_maps_to_any(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "required",  # Chat 风格
        }
        ir = chat_to_ir(body)
        anthropic_body = anthropic_to_upstream(ir)
        assert anthropic_body["tool_choice"] == "any"


class TestDictToolChoiceFormatCompat:
    """dict tool_choice 在三协议间转换时格式必须正确。"""

    def test_chat_dict_to_responses_uses_flat_name(self):
        """Chat 格式 {'type':'function','function':{'name':'X'}} → Responses 格式 {'type':'function','name':'X'}."""
        body = {
            "model": "gpt-5",
            "input": "Hi",
            "tool_choice": {"type": "function", "function": {"name": "search"}},
        }
        # Chat → IR → Responses
        ir = chat_to_ir(body)
        upstream = responses_to_upstream(ir)
        assert upstream["tool_choice"] == {"type": "function", "name": "search"}

    def test_responses_dict_to_anthropic_uses_tool_type(self):
        """Responses 格式 {'type':'function','name':'X'} → Anthropic 格式 {'type':'tool','name':'X'}."""
        body = {
            "model": "gpt-5",
            "input": "Hi",
            "tool_choice": {"type": "function", "name": "search"},
        }
        ir = responses_to_ir(body)
        upstream = anthropic_to_upstream(ir)
        assert upstream["tool_choice"] == {"type": "tool", "name": "search"}

    def test_anthropic_dict_to_responses_uses_flat_name(self):
        """Anthropic {'type':'tool','name':'X'} → Responses {'type':'function','name':'X'}."""
        body = {
            "model": "claude",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "search"},
        }
        upstream = convert_request("anthropic", "openai/responses", body)
        assert upstream["tool_choice"] == {"type": "function", "name": "search"}


class TestChatStreamingToolUseEndFallback:
    """Chat 流式：tool_use_end 携带完整 input 但未通过 delta 流式传输时，必须补发 arguments。"""

    @pytest.mark.asyncio
    async def test_tool_use_end_emits_arguments_when_no_delta(self):
        from llm_proxy.protocol.ir.chat import format_ir_as_sse
        from llm_proxy.protocol.ir.types import IRStreamEvent

        events = [
            IRStreamEvent(type="message_start", data={"id": "x", "model": "m"}),
            IRStreamEvent(type="tool_use_start", data={"id": "call_1", "name": "search"}),
            # 没有 tool_use_delta，直接 end 携带完整 input
            IRStreamEvent(type="tool_use_end", data={"id": "call_1", "input": {"q": "test"}}),
            IRStreamEvent(type="message_stop", data={"stop_reason": "tool_use"}),
        ]

        chunks: list[dict] = []
        async for chunk in format_ir_as_sse(_async_iter(events), "m"):
            text = chunk.decode("utf-8")
            for line in text.split("\n"):
                if line.startswith("data: ") and "[DONE]" not in line:
                    chunks.append(json.loads(line[6:]))

        # 找出所有 tool_calls delta
        tool_args_deltas = []
        for c in chunks:
            for choice in c.get("choices", []):
                delta = choice.get("delta", {})
                for tc in delta.get("tool_calls", []):
                    args = tc.get("function", {}).get("arguments")
                    if args is not None and args != "":
                        tool_args_deltas.append(args)
        # 必须有至少一个非空 arguments delta
        assert any(tool_args_deltas), f"Expected non-empty arguments delta, got {tool_args_deltas}"
        # 拼接应能解析为完整 input
        combined = "".join(tool_args_deltas)
        assert json.loads(combined) == {"q": "test"}


class TestResponsesRefusalMapsIncomplete:
    """非流式 Responses 响应：refusal stop_reason 必须 → status=incomplete + reason=content_filter。"""

    def test_refusal_status_is_incomplete(self):
        ir = IRResponse(
            id="r1",
            model="m",
            content_blocks=[IRTextBlock(text="I can't help with that.")],
            stop_reason="refusal",
            usage={"input_tokens": 5, "output_tokens": 3},
        )
        body = responses_response_from_ir(ir)
        assert body["status"] == "incomplete"
        assert body.get("incomplete_details", {}).get("reason") == "content_filter"


class TestStripThinkTagsUnclosed:
    """未闭合 <think> 标签：reasoning 不应被双倍计数。"""

    def test_unclosed_think_no_double_count(self):
        from llm_proxy.protocol.think_tag import strip_think_tags
        reasoning, content = strip_think_tags("<think>reasoning here")
        assert reasoning == "reasoning here"
        assert content == ""

    def test_unclosed_think_with_trailing_content(self):
        from llm_proxy.protocol.think_tag import strip_think_tags
        # 未闭合时所有后续内容都视为 reasoning
        reasoning, content = strip_think_tags("<think>still thinking")
        assert reasoning == "still thinking"
        assert content == ""


class TestChatStreamingToolCallWithoutName:
    """Chat 流式：上游 chunk 携带 arguments 但无 name 时不应静默丢弃。"""

    @pytest.mark.asyncio
    async def test_arguments_without_name_still_emitted(self):
        from llm_proxy.protocol.ir.chat import parse_stream_to_ir, format_ir_as_sse
        from llm_proxy.protocol.ir.types import IRStreamEvent

        # 模拟上游：name 缺失，只有 arguments delta
        upstream_lines = [
            'data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\\"q\\":\\"test\\"}"}}]}}]}',
            'data: {"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            'data: [DONE]',
        ]

        events = []
        async for ev in parse_stream_to_ir(_FakeResp(upstream_lines), "m"):
            events.append(ev)

        # 必须有 tool_use_end 事件携带完整 input
        end_events = [e for e in events if e.type == "tool_use_end"]
        assert len(end_events) == 1
        assert end_events[0].data.get("input") == {"q": "test"}


# ── 辅助 ────────────────────────────────────────────────────────────


async def _async_iter(items):
    for x in items:
        yield x


class _FakeResp:
    """模拟 httpx 流式响应，按行产出 SSE 数据。"""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


# ── 第三轮 review 修复点（数据丢失类）───────────────────────────────


class TestAnthropicToolResultImagePreserved:
    """Anthropic tool_result 含 image block 时，Anthropic→IR→Anthropic 应保留图片。"""

    def test_image_in_tool_result_roundtrip(self):
        body = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": [
                        {"type": "text", "text": "screenshot below"},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        }},
                    ],
                }],
            }],
        }
        # Anthropic → IR → Anthropic
        ir = anthropic_to_ir(body)
        upstream = anthropic_to_upstream(ir)
        # 找到 tool_result block
        msg = upstream["messages"][0]
        tr_block = next(b for b in msg["content"] if b.get("type") == "tool_result")
        # content 应保留为 list（含 image block），不应被压成纯文本
        assert isinstance(tr_block["content"], list)
        types_in_result = [b.get("type") for b in tr_block["content"]]
        assert "image" in types_in_result, f"image block dropped: {types_in_result}"


class TestStreamingUsageCacheTokens:
    """流式 usage 事件应保留 cache_read_input_tokens / cache_creation_input_tokens。"""

    @pytest.mark.asyncio
    async def test_anthropic_streaming_usage_preserves_cache_tokens(self):
        from llm_proxy.protocol.ir.anthropic import format_ir_as_sse
        from llm_proxy.protocol.ir.types import IRStreamEvent

        events = [
            IRStreamEvent(type="message_start", data={"id": "x", "model": "m"}),
            IRStreamEvent(type="usage", data={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20,
            }),
            IRStreamEvent(type="message_stop", data={"stop_reason": "end_turn"}),
        ]

        chunks: list[dict] = []
        async for chunk in format_ir_as_sse(_async_iter(events), "m"):
            text = chunk.decode("utf-8")
            for line in text.split("\n"):
                if line.startswith("data: ") and "[DONE]" not in line:
                    chunks.append(json.loads(line[6:]))

        # 找出 message_delta 中的 usage
        usage_chunk = next(
            (c for c in chunks if c.get("type") == "message_delta"),
            None,
        )
        assert usage_chunk is not None, "no message_delta chunk emitted"
        usage = usage_chunk.get("usage", {})
        assert usage.get("cache_read_input_tokens") == 80, f"cache_read lost: {usage}"
        assert usage.get("cache_creation_input_tokens") == 20, f"cache_creation lost: {usage}"


class TestRedactedThinkingInChatResponses:
    """IRRedactedThinkingBlock 在 Chat/Responses 响应方向不应被静默丢弃。"""

    def test_redacted_thinking_in_chat_response(self):
        """Chat 响应：redacted_thinking 应转换为 reasoning 占位（而非静默丢弃）。"""
        ir = IRResponse(
            id="r1",
            model="m",
            content_blocks=[
                IRTextBlock(text="answer"),
                IRRedactedThinkingBlock(data="encrypted_blob"),
            ],
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 3},
        )
        body = chat_response_from_ir(ir)
        # 至少不应静默丢弃：reasoning_content 或 message content 应反映 redacted thinking 存在
        # Chat 没有原生 redacted_thinking 概念，转换为 reasoning_content 占位
        msg = body["choices"][0]["message"]
        has_reasoning = bool(msg.get("reasoning_content"))
        has_text = "answer" in (msg.get("content") or "")
        assert has_text, f"text lost: {msg}"
        assert has_reasoning, f"redacted_thinking silently dropped: {msg}"

    def test_redacted_thinking_in_responses_response(self):
        """Responses 命令响应：redacted_thinking 应转换为 reasoning 占位（而非静默丢弃）。"""
        ir = IRResponse(
            id="r1",
            model="m",
            content_blocks=[
                IRRedactedThinkingBlock(data="encrypted_blob"),
                IRTextBlock(text="answer"),
            ],
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 3},
        )
        body = responses_response_from_ir(ir)
        # 至少应有 reasoning item（不被静默丢弃）
        output = body.get("output", [])
        types_in_output = [item.get("type") for item in output]
        has_reasoning = any("reasoning" in t for t in types_in_output)
        assert has_reasoning, f"redacted_thinking silently dropped: {types_in_output}"


class TestResponsesFailedStatusMapping:
    """Responses 上游 status=failed/cancelled 不应被映射为 end_turn。"""

    def test_failed_status_mapped_to_refusal(self):
        """status=failed → IR stop_reason 应反映错误，不应是 end_turn。"""
        # 模拟上游 Responses 响应 status=failed
        upstream = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5",
            "status": "failed",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 0},
            "error": {"code": "server_error", "message": "upstream failed"},
        }
        ir = responses_response_to_ir(upstream)
        # stop_reason 不应是 end_turn（应反映失败）
        assert ir.stop_reason != "end_turn", f"failed status mapped to end_turn: {ir.stop_reason}"

    def test_cancelled_status_mapped_to_refusal(self):
        upstream = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5",
            "status": "cancelled",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        ir = responses_response_to_ir(upstream)
        assert ir.stop_reason != "end_turn", f"cancelled status mapped to end_turn: {ir.stop_reason}"
