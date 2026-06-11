"""测试 fix_orphaned_tool_calls 修复逻辑"""

from llm_proxy.services.tool_call_fix import fix_orphaned_tool_calls


def test_no_messages():
    assert fix_orphaned_tool_calls([]) == []


def test_no_tool_calls():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert result == messages


def test_complete_tool_calls():
    messages = [
        {"role": "user", "content": "run tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "user", "content": "thanks"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert result == messages


def test_single_orphaned_tool_call():
    messages = [
        {"role": "user", "content": "run tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "next message"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert len(result) == 4
    assert result[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "[Tool call was interrupted]",
    }
    assert result[3] == {"role": "user", "content": "next message"}


def test_partial_orphaned_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
        {"role": "user", "content": "next"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert len(result) == 4
    assert result[2] == {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": "[Tool call was interrupted]",
    }
    assert result[3] == {"role": "user", "content": "next"}


def test_multiple_assistant_with_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "done"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert len(result) == 6
    assert result[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "[Tool call was interrupted]",
    }
    assert result[4] == {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": "[Tool call was interrupted]",
    }


def test_does_not_modify_original():
    original = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "next"},
    ]
    original_copy = [dict(m) for m in original]
    fix_orphaned_tool_calls(original)
    assert original == original_copy


def test_tool_calls_without_id():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "next"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert result == messages


def test_empty_tool_calls_list():
    messages = [
        {"role": "assistant", "content": "hi", "tool_calls": []},
        {"role": "user", "content": "next"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert result == messages


def test_misplaced_tool_response_removed():
    """Codex 场景：call_1 的 response 出现在下一个 assistant(tool_calls) 之后
    call_1 的 late response 降级为 user 消息保留上下文"""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "late result for call_1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "result for call_2"},
        {"role": "user", "content": "next"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert len(result) == 6
    assert result[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "[Tool call was interrupted]",
    }
    assert result[2]["tool_calls"][0]["id"] == "call_2"
    assert result[3] == {"role": "tool", "tool_call_id": "call_2", "content": "result for call_2"}
    assert result[4]["role"] == "user"
    assert "call_1" in result[4]["content"]
    assert "late result for call_1" in result[4]["content"]
    assert result[5] == {"role": "user", "content": "next"}


def test_misplaced_tool_response_with_own_responses():
    """第二个 assistant 有自己的 response，但还夹着上一个 assistant 的迟到的 response
    迟到的 response 降级为 user 消息保留上下文"""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "late result for call_1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "result for call_2"},
        {"role": "user", "content": "next"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert len(result) == 6
    assert result[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "[Tool call was interrupted]",
    }
    assert result[3] == {"role": "tool", "tool_call_id": "call_2", "content": "result for call_2"}
    assert result[4]["role"] == "user"
    assert "call_1" in result[4]["content"]


def test_misplaced_tool_response_dangling():
    """迟到的 tool response 对应的 assistant 不在前面，降级为 user 消息保留上下文"""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "orphaned response"},
        {"role": "user", "content": "next"},
    ]
    result = fix_orphaned_tool_calls(messages)
    assert len(result) == 4
    assert result[0]["tool_calls"][0]["id"] == "call_1"
    assert result[1] == {"role": "tool", "tool_call_id": "call_1", "content": "result1"}
    assert result[2]["role"] == "user"
    assert "call_2" in result[2]["content"]
    assert "orphaned response" in result[2]["content"]
    assert result[3] == {"role": "user", "content": "next"}
