"""测试 input_compressor 输入压缩逻辑"""

from llm_proxy.services.input_compressor import (
    CompressionConfig,
    CompressionStats,
    InputCompressor,
)


# ─── 辅助 ────────────────────────────────────────────────────────────

def _make_compressor(strategies=None, **kwargs) -> InputCompressor:
    cfg = CompressionConfig(enabled=True, strategies=strategies or [
        "drop_progress", "truncate", "collapse", "shorten_paths",
    ], **kwargs)
    return InputCompressor(cfg)


# ─── CompressionConfig ───────────────────────────────────────────────

def test_config_defaults():
    cfg = CompressionConfig()
    assert not cfg.enabled
    assert cfg.strategies == []  # 默认空，启用时需显式指定
    assert cfg.truncate_max_lines == 200


def test_config_has_strategy():
    cfg = CompressionConfig(strategies=["drop_progress", "collapse"])
    assert cfg.has_strategy("drop_progress")
    assert not cfg.has_strategy("truncate")


# ─── drop_progress ───────────────────────────────────────────────────

def test_drop_progress_removes_compiling():
    c = _make_compressor(strategies=["drop_progress"])
    text = "Compiling serde v1.0.0\nCompiling tokio v1.0.0\nerror: mismatched types"
    result = c.compress_tool_result(text)
    assert "Compiling" not in result
    assert "error: mismatched types" in result


def test_drop_progress_removes_docker_steps():
    c = _make_compressor(strategies=["drop_progress"])
    text = "Step 1/10 : FROM ubuntu\nStep 2/10 : RUN apt-get update\nSuccessfully built abc123"
    result = c.compress_tool_result(text)
    assert "Step 1/10" not in result
    assert "Step 2/10" not in result
    assert "Successfully built" in result


def test_drop_progress_removes_npm_added():
    c = _make_compressor(strategies=["drop_progress"])
    text = "added 150 packages in 3s\nnpm notice created a lockfile\nerror: missing dependency"
    result = c.compress_tool_result(text)
    assert "added 150 packages" not in result
    assert "error: missing dependency" in result


def test_drop_progress_removes_cargo_fresh():
    c = _make_compressor(strategies=["drop_progress"])
    text = "Fresh serde v1.0.0\nFresh tokio v1.0.0\nRunning target/debug/main"
    result = c.compress_tool_result(text)
    assert "Fresh" not in result


def test_drop_progress_preserves_errors():
    c = _make_compressor(strategies=["drop_progress"])
    text = "Compiling foo\nerror[E0308]: mismatched types\n  --> src/main.rs:4:5"
    result = c.compress_tool_result(text)
    assert "error[E0308]" in result
    assert "src/main.rs" in result


def test_drop_progress_short_text_unchanged():
    c = _make_compressor(strategies=["drop_progress"])
    text = "ok"  # < 50 chars
    assert c.compress_tool_result(text) == text


# ─── truncate ────────────────────────────────────────────────────────

def test_truncate_code_block():
    c = _make_compressor(strategies=["truncate"], truncate_max_lines=10)
    # 20 行代码块
    code_lines = "\n".join(f"line {i}" for i in range(20))
    text = f"```python\n{code_lines}\n```"
    result = c.compress_tool_result(text)
    assert "line 0" in result   # 前 5 行保留
    assert "line 19" in result  # 后 10 行保留
    assert "truncated" in result


def test_truncate_short_code_block_unchanged():
    c = _make_compressor(strategies=["truncate"], truncate_max_lines=200)
    text = "```python\nprint('hello')\n```"
    result = c.compress_tool_result(text)
    assert result == text


def test_truncate_no_code_block_no_truncate():
    c = _make_compressor(strategies=["truncate"], truncate_max_lines=10)
    # 纯文本，没有代码块围栏
    text = "\n".join(f"line {i}" for i in range(300))
    result = c.compress_tool_result(text)
    # 纯文本不做 truncate
    assert result == text


def test_truncate_multiple_code_blocks():
    c = _make_compressor(strategies=["truncate"], truncate_max_lines=10)
    long_code = "\n".join(f"line {i}" for i in range(20))
    text = f"```python\n{long_code}\n```\nSome text\n```js\n{long_code}\n```"
    result = c.compress_tool_result(text)
    # 两个代码块都应该被截断
    assert result.count("truncated") == 2


def test_truncate_indicator_format():
    c = _make_compressor(strategies=["truncate"], truncate_max_lines=10,
                         truncate_indicator="... [{n} lines cut]")
    long_code = "\n".join(f"line {i}" for i in range(20))
    text = f"```python\n{long_code}\n```"
    result = c.compress_tool_result(text)
    assert "lines cut" in result


# ─── collapse ────────────────────────────────────────────────────────

def test_collapse_blank_lines():
    c = _make_compressor(strategies=["collapse"])
    text = "line1\n\n\n\n\nline2"
    result = c.compress_tool_result(text)
    # collapse_max_blank_lines=2, 最多 2 个空行（3 个换行符）
    # 原文 5 个换行（4 个空行），应折叠为 3 个换行（2 个空行）
    assert "\n\n\n\n" not in result
    assert "line1" in result
    assert "line2" in result


def test_collapse_no_change_when_within_limit():
    c = _make_compressor(strategies=["collapse"])
    text = "line1\n\nline2"  # 只有 1 个空行
    result = c.compress_tool_result(text)
    assert result == text


def test_collapse_custom_max():
    c = _make_compressor(strategies=["collapse"], collapse_max_blank_lines=1)
    text = "line1\n\n\nline2"  # 2 个空行
    result = c.compress_tool_result(text)
    # collapse_max_blank_lines=1, 最多 1 个空行（2 个换行符）
    assert "\n\n\n" not in result


# ─── shorten_paths ───────────────────────────────────────────────────

def test_shorten_paths_unix():
    c = _make_compressor(strategies=["shorten_paths"])
    text = "Error in /Users/tingung/Projects/myapp/src/main.rs:4:5"
    result = c.compress_tool_result(text)
    # 应该将 /Users/tingung/Projects/myapp/ 替换为 ./
    assert "./src/main.rs" in result
    assert "/Users/tingung/Projects/myapp/" not in result


def test_shorten_paths_no_match():
    c = _make_compressor(strategies=["shorten_paths"])
    text = "Error in ./src/main.rs:4:5"  # 已经是相对路径
    result = c.compress_tool_result(text)
    assert result == text


def test_shorten_paths_disabled():
    c = _make_compressor(strategies=["shorten_paths"], shorten_paths_enabled=False)
    text = "Error in /Users/tingung/Projects/myapp/src/main.rs:4:5"
    result = c.compress_tool_result(text)
    # 禁用后不替换
    assert "/Users/tingung/Projects/myapp/" in result


# ─── compress_tool_result（组合策略）────────────────────────────────

def test_compress_tool_result_combined():
    c = _make_compressor(strategies=["drop_progress", "collapse", "truncate"],
                         truncate_max_lines=10)
    text = (
        "Compiling serde v1.0.0\n"
        "Compiling tokio v1.0.0\n"
        "\n\n\n\n"
        "```rust\n"
        + "\n".join(f"fn func_{i}() {{}}" for i in range(20))
        + "\n```\n"
        "error: mismatched types"
    )
    result = c.compress_tool_result(text)
    assert "Compiling" not in result
    assert "error: mismatched types" in result
    assert "truncated" in result


def test_compress_tool_result_no_change_short():
    c = _make_compressor()
    text = "ok"  # < 50 chars
    assert c.compress_tool_result(text) == text


def test_compress_tool_result_empty():
    c = _make_compressor()
    assert c.compress_tool_result("") == ""
    assert c.compress_tool_result(None) is None


# ─── compress_text（通用文本，仅 collapse）──────────────────────────

def test_compress_text_only_collapse():
    c = _make_compressor(strategies=["collapse"])
    text = "Hello\n\n\n\nWorld"
    result = c.compress_text(text)
    assert "\n\n\n\n" not in result


def test_compress_text_no_truncate():
    c = _make_compressor(strategies=["truncate", "collapse"])
    # 通用文本不做 truncate，即使很长
    text = "A" * 10000
    result = c.compress_text(text)
    assert len(result) == len(text)  # 不截断


# ─── CompressionStats ────────────────────────────────────────────────

def test_stats_record():
    stats = CompressionStats()
    stats.record("hello world this is long text", "hello world")
    assert stats.original_chars == len("hello world this is long text")
    assert stats.compressed_chars == len("hello world")
    assert stats.items_compressed == 1


def test_stats_no_record_when_unchanged():
    c = _make_compressor(strategies=["collapse"])
    text = "no blank lines here at all"
    c.compress_tool_result(text)
    # 文本没变，不记录统计
    assert c.stats.items_compressed == 0


# ─── Anthropic 格式 ─────────────────────────────────────────────────

def test_compress_anthropic_body_tool_result_string():
    c = _make_compressor(strategies=["drop_progress", "collapse"])
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "Compiling serde v1.0.0\n\n\n\nerror: failed to compile the project"}
            ]},
        ]
    }
    c.compress_anthropic_body(body)
    block = body["messages"][0]["content"][0]
    assert "Compiling" not in block["content"]
    assert "error: failed" in block["content"]


def test_compress_anthropic_body_tool_result_blocks():
    c = _make_compressor(strategies=["drop_progress"])
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [
                     {"type": "text", "text": "Compiling serde v1.0.0\nerror: failed to compile the project"}
                 ]}
            ]},
        ]
    }
    c.compress_anthropic_body(body)
    text_block = body["messages"][0]["content"][0]["content"][0]
    assert "Compiling" not in text_block["text"]
    assert "error: failed" in text_block["text"]


def test_compress_anthropic_body_system_string():
    c = _make_compressor(strategies=["collapse"])
    body = {
        "system": "You are helpful.\n\n\n\nFollow these instructions carefully.",
        "messages": [],
    }
    c.compress_anthropic_body(body)
    assert "\n\n\n\n" not in body["system"]


def test_compress_anthropic_body_system_blocks():
    c = _make_compressor(strategies=["collapse"])
    body = {
        "system": [{"text": "You are helpful.\n\n\n\nFollow these instructions carefully."}],
        "messages": [],
    }
    c.compress_anthropic_body(body)
    assert "\n\n\n\n" not in body["system"][0]["text"]


def test_compress_anthropic_body_thinking_unchanged():
    c = _make_compressor(strategies=["drop_progress"])
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Compiling my thoughts..."},
                {"type": "text", "text": "Here is the result."},
            ]},
        ]
    }
    original_thinking = body["messages"][0]["content"][0]["thinking"]
    c.compress_anthropic_body(body)
    # thinking 不应被压缩
    assert body["messages"][0]["content"][0]["thinking"] == original_thinking


# ─── Chat 格式 ──────────────────────────────────────────────────────

def test_compress_chat_body_tool_message():
    c = _make_compressor(strategies=["drop_progress", "collapse"])
    body = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_1",
             "content": "Compiling serde v1.0.0\n\n\n\nerror: failed to compile the project"},
        ]
    }
    c.compress_chat_body(body)
    assert "Compiling" not in body["messages"][0]["content"]
    assert "error: failed" in body["messages"][0]["content"]


def test_compress_chat_body_user_message_collapse_only():
    c = _make_compressor(strategies=["truncate", "collapse"])
    long_text = "Hello\n\n\n\nWorld" + "A" * 200
    body = {
        "messages": [
            {"role": "user", "content": long_text},
        ]
    }
    c.compress_chat_body(body)
    # user 消息只做 collapse，不做 truncate
    assert "\n\n\n\n" not in body["messages"][0]["content"]


def test_compress_chat_body_tool_content_list():
    c = _make_compressor(strategies=["drop_progress"])
    body = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_1",
             "content": [{"type": "text", "text": "Compiling serde v1.0.0\nerror: failed to compile the project"}]},
        ]
    }
    c.compress_chat_body(body)
    text_part = body["messages"][0]["content"][0]
    assert "Compiling" not in text_part["text"]


# ─── Responses 格式 ─────────────────────────────────────────────────

def test_compress_responses_body_function_call_output():
    c = _make_compressor(strategies=["drop_progress", "collapse"])
    body = {
        "input": [
            {"type": "function_call_output", "call_id": "fc_1",
             "output": "Compiling serde v1.0.0\n\n\n\nerror: failed to compile the project"},
        ]
    }
    c.compress_responses_body(body)
    assert "Compiling" not in body["input"][0]["output"]
    assert "error: failed" in body["input"][0]["output"]


def test_compress_responses_body_instructions():
    c = _make_compressor(strategies=["collapse"])
    body = {
        "instructions": "You are helpful.\n\n\n\nFollow these instructions carefully.",
        "input": [],
    }
    c.compress_responses_body(body)
    assert "\n\n\n\n" not in body["instructions"]


def test_compress_responses_body_message_items():
    c = _make_compressor(strategies=["collapse"])
    body = {
        "input": [
            {"type": "message", "content": [
                {"type": "input_text", "text": "Hello\n\n\n\nWorld" + "A" * 200}
            ]},
        ]
    }
    c.compress_responses_body(body)
    text_part = body["input"][0]["content"][0]
    assert "\n\n\n\n" not in text_part["text"]


# ─── 端到端：完整请求体 ─────────────────────────────────────────────

def test_full_anthropic_request():
    c = _make_compressor(strategies=["drop_progress", "truncate", "collapse"],
                         truncate_max_lines=10)
    body = {
        "model": "claude-sonnet-4-6",
        "system": "You are a coding assistant.\n\n\n\nBe helpful and concise in your responses.",
        "messages": [
            {"role": "user", "content": "Run cargo build"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "bash",
                 "input": {"cmd": "cargo build"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": (
                     "Compiling serde v1.0.0\n"
                     "Compiling tokio v1.0.0\n"
                     "```rust\n"
                     + "\n".join(f"fn func_{i}() {{}}" for i in range(20))
                     + "\n```\n"
                     "error: mismatched types in main function"
                 )},
            ]},
        ],
    }
    c.compress_anthropic_body(body)

    # system prompt 被折叠
    assert "\n\n\n\n" not in body["system"]

    # tool_result 被压缩
    tool_result = body["messages"][2]["content"][0]["content"]
    assert "Compiling" not in tool_result
    assert "error: mismatched types" in tool_result
    assert "truncated" in tool_result

    # tool_use 不受影响
    tool_use = body["messages"][1]["content"][0]
    assert tool_use["name"] == "bash"
    assert tool_use["input"]["cmd"] == "cargo build"


def test_disabled_config_no_compression():
    c = InputCompressor(CompressionConfig(enabled=False))
    body = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_1",
             "content": "Compiling serde v1.0.0\n\n\n\nerror: failed"},
        ]
    }
    c.compress_chat_body(body)
    # 禁用时仍然压缩（enabled 只在 CompressionStep 中检查）
    # 但 enabled=False 的默认 strategies 为空
    assert "Compiling" in body["messages"][0]["content"]
