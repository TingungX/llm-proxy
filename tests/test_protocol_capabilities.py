"""protocol/capabilities.py 的单元测试

覆盖：
  - is_reachable 所有 9 种 (client, upstream) 组合
  - select_upstream 同协议优先 / 跨协议按表选 / 不可达 / 空集合
  - NoReachableProtocol 错误信息
  - 别名规范化（"openai" → "openai/chat-completions"）
"""

import pytest

from llm_proxy.protocol.capabilities import (
    IMPLEMENTED_CONVERSIONS,
    NoReachableProtocol,
    is_reachable,
    normalize,
    normalize_set,
    select_upstream,
)


# ─── is_reachable ──────────────────────────────────────────────────

@pytest.mark.parametrize("client,upstream,expected", [
    # 同协议：永远可达
    ("anthropic", "anthropic", True),
    ("openai/chat-completions", "openai/chat-completions", True),
    ("openai/responses", "openai/responses", True),
    # 显式实现的跨协议对
    ("anthropic", "openai/chat-completions", True),
    ("openai/responses", "openai/chat-completions", True),
    ("openai/chat-completions", "openai/responses", True),
    # 未实现的跨协议对
    ("anthropic", "openai/responses", False),
    ("openai/chat-completions", "anthropic", False),
    ("openai/responses", "anthropic", False),
    # 别名：客户端用 "openai" 视为 "openai/chat-completions"
    ("openai", "openai/chat-completions", True),  # 别名同协议
    ("openai", "anthropic", False),  # 别名 vs 真正的 anthropic
])
def test_is_reachable(client, upstream, expected):
    assert is_reachable(client, upstream) is expected


# ─── select_upstream: 同协议优先 ─────────────────────────────────────

def test_select_same_protocol_anthropic():
    """client=anthropic, available={anthropic, chat, responses} → anthropic"""
    assert select_upstream("anthropic", {"anthropic", "openai/chat-completions", "openai/responses"}) == "anthropic"


def test_select_same_protocol_chat():
    """client=chat, available={chat, responses} → chat（同协议优先于 list 顺序）"""
    assert select_upstream("openai/chat-completions", {"openai/chat-completions", "openai/responses"}) == "openai/chat-completions"


def test_select_same_protocol_responses():
    """client=responses, available={chat, responses} → responses（透传）"""
    assert select_upstream("openai/responses", {"openai/chat-completions", "openai/responses"}) == "openai/responses"


# ─── select_upstream: 跨协议按表选 ─────────────────────────────────

def test_select_cross_protocol_anthropic_prefers_chat():
    """client=anthropic, available={chat, responses} → chat（表里 anthropic→chat 在前）"""
    assert select_upstream("anthropic", {"openai/chat-completions", "openai/responses"}) == "openai/chat-completions"


def test_select_cross_protocol_chat_prefers_responses():
    """client=chat, available={responses} → responses（chat→responses 转换）"""
    assert select_upstream("openai/chat-completions", {"openai/responses"}) == "openai/responses"


def test_select_cross_protocol_responses_falls_back_to_chat():
    """client=responses, available={chat} → chat（responses→chat 转换）"""
    assert select_upstream("openai/responses", {"openai/chat-completions"}) == "openai/chat-completions"


# ─── select_upstream: 不可达 ───────────────────────────────────────

def test_select_unreachable_anthropic_to_responses():
    """client=anthropic, available={responses} → NoReachableProtocol（anthropic→responses 未实现）"""
    with pytest.raises(NoReachableProtocol) as exc_info:
        select_upstream("anthropic", {"openai/responses"})
    assert exc_info.value.client == "anthropic"
    assert exc_info.value.available == {"openai/responses"}


def test_select_unreachable_chat_to_anthropic():
    """client=chat, available={anthropic} → NoReachableProtocol"""
    with pytest.raises(NoReachableProtocol):
        select_upstream("openai/chat-completions", {"anthropic"})


def test_select_empty_available():
    """available={} → NoReachableProtocol"""
    with pytest.raises(NoReachableProtocol):
        select_upstream("anthropic", set())


# ─── select_upstream: 集合类型容忍 ─────────────────────────────────

def test_select_accepts_list_input():
    """available 可以是 list（兼容历史 data shape）"""
    assert select_upstream("anthropic", ["openai/chat-completions"]) == "openai/chat-completions"


def test_select_accepts_iterable():
    """available 可以是任意 iterable"""
    result = select_upstream("anthropic", (p for p in ["openai/chat-completions"]))
    assert result == "openai/chat-completions"


# ─── 别名规范化 ────────────────────────────────────────────────────

def test_normalize_openai_alias():
    assert normalize("openai") == "openai/chat-completions"


def test_normalize_passthrough():
    assert normalize("anthropic") == "anthropic"
    assert normalize("openai/responses") == "openai/responses"


def test_normalize_set():
    assert normalize_set({"openai", "openai/responses"}) == {
        "openai/chat-completions",
        "openai/responses",
    }


def test_select_with_openai_alias():
    """available={"openai"} (alias) → 规范化为 chat"""
    assert select_upstream("openai/chat-completions", {"openai"}) == "openai/chat-completions"


# ─── NoReachableProtocol 错误信息 ──────────────────────────────────

def test_no_reachable_protocol_message_includes_diagnostics():
    err = NoReachableProtocol("anthropic", {"openai/responses"})
    msg = str(err)
    assert "anthropic" in msg
    assert "openai/responses" in msg
    assert "Implemented conversions" in msg
    # 至少要有一行"修复建议"
    assert "Fix:" in msg


def test_no_reachable_protocol_strips_client_from_available():
    """available 中如果包含 client 自身（不该有但防御性），构造时去掉"""
    err = NoReachableProtocol("anthropic", {"anthropic"})
    assert "anthropic" not in err.available


# ─── IMPLEMENTED_CONVERSIONS 表完整性 ─────────────────────────────

def test_implemented_conversions_count():
    """表的大小应保持稳定（加新转换时显式更新）"""
    assert len(IMPLEMENTED_CONVERSIONS) == 3


def test_implemented_conversions_no_duplicates():
    pairs = list(IMPLEMENTED_CONVERSIONS)
    assert len(pairs) == len(set(pairs))


def test_implemented_conversions_no_self_loops():
    """同协议对不应出现在转换表里（隐含可达）"""
    for src, tgt in IMPLEMENTED_CONVERSIONS:
        assert src != tgt, f"self-loop detected: ({src}, {tgt})"

