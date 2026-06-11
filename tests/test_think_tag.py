"""Tests for ThinkTagStateMachine"""
import pytest
from llm_proxy.protocol.think_tag import ThinkTagStateMachine, strip_think_tags


class TestThinkTagBasicExtraction:
    def test_think_tag_at_start_extracts_reasoning(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("<think>I need to calculate</think>def fib(n):")
        assert "I need to calculate" in "".join(reasoning)
        assert "def fib(n):" in "".join(content)
        assert m.state == "done"

    def test_no_think_tag_passes_through(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("just plain text")
        assert reasoning == []
        assert content == ["just plain text"]
        assert m.state == "none"

    def test_empty_think_region(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("<think></think>after")
        assert reasoning == []
        assert content == ["after"]
        assert m.state == "done"

    def test_unclosed_think_tag_all_reasoning(self):
        m = ThinkTagStateMachine()
        m.feed("<think>")
        reasoning, content = m.feed("still thinking")
        assert "still thinking" in "".join(reasoning)
        assert content == []
        assert m.state == "inside"


class TestThinkTagStartPositionOnly:
    def test_think_tag_in_middle_not_recognized(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("hello <think>not think</think> more")
        assert reasoning == []
        assert "hello <think>not think</think> more" in "".join(content)
        assert m.state == "none"
        assert m.can_start is False

    def test_leading_whitespace_then_think_tag_recognized(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("   <think>thinking</think>text")
        assert "thinking" in "".join(reasoning)
        assert "text" in "".join(content)

    def test_leading_whitespace_then_normal_text(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("   hello world")
        assert reasoning == []
        assert "   hello world" in "".join(content)

    def test_can_start_closes_after_non_whitespace(self):
        m = ThinkTagStateMachine()
        reasoning, content = m.feed("text")
        assert m.can_start is False
        reasoning2, content2 = m.feed("<think>should not match</think>more")
        assert reasoning2 == []
        assert "<think>should not match</think>more" in "".join(content2)


class TestThinkTagCrossChunkBoundary:
    def test_partial_open_tag_across_chunks(self):
        m = ThinkTagStateMachine()
        r1, c1 = m.feed("<thi")
        assert r1 == []
        assert c1 == []
        r2, c2 = m.feed("nk>thinking</think>done")
        assert "thinking" in "".join(r2)
        assert "done" in "".join(c2)
        assert m.state == "done"

    def test_partial_close_tag_across_chunks(self):
        m = ThinkTagStateMachine()
        r1, c1 = m.feed("<think>thinking</thi")
        r2, c2 = m.feed("nk>done")
        assert "thinking" in "".join(r1)
        assert "done" in "".join(c2)

    def test_isolated_lt_buffered(self):
        m = ThinkTagStateMachine()
        r1, c1 = m.feed("<")
        assert r1 == []
        assert c1 == []
        r2, c2 = m.feed("some text")
        assert r2 == []
        assert "<some text" in "".join(c2)
        assert m.can_start is False


class TestThinkTagDrain:
    def test_drain_unbuffered_returns_empty(self):
        m = ThinkTagStateMachine()
        remaining, to_reasoning = m.drain()
        assert remaining == ""
        assert to_reasoning is False

    def test_drain_pending_buf_as_content(self):
        m = ThinkTagStateMachine()
        m.feed("<thi")
        remaining, to_reasoning = m.drain()
        assert "thi" in remaining
        assert to_reasoning is False

    def test_drain_inside_state_as_reasoning(self):
        m = ThinkTagStateMachine()
        m.feed("<think>still thinking")
        remaining, to_reasoning = m.drain()
        assert "still thinking" in remaining
        assert to_reasoning is True

    def test_drain_leading_ws_with_buf(self):
        m = ThinkTagStateMachine()
        m.feed("   <thi")
        remaining, to_reasoning = m.drain()
        assert "   " in remaining
        assert "thi" in remaining
        assert to_reasoning is False

    def test_drain_after_done_returns_empty(self):
        m = ThinkTagStateMachine()
        m.feed("<think>think</think>text")
        remaining, to_reasoning = m.drain()
        assert remaining == ""
        assert to_reasoning is False


class TestThinkTagReset:
    def test_reset_restores_initial_state(self):
        m = ThinkTagStateMachine()
        m.feed("<think>thinking</think>text")
        m.reset()
        assert m.state == "none"
        assert m.can_start is True
        assert m.buf == ""
        assert m.leading_ws == ""


class TestStripThinkTags:
    def test_double_reasoning(self):
        """content 同时包含 reasoning_content 和 <think> tag"""
        from llm_proxy.protocol.anthropic_openai.response import _build_content_blocks
        msg = {"reasoning_content": "official", "content": "<think>extra</think>visible"}
        blocks = _build_content_blocks(msg)
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "official"
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "visible"

    def test_content_only_think_tag(self):
        """content 只有 <think> tag，没有 reasoning_content"""
        from llm_proxy.protocol.anthropic_openai.response import _build_content_blocks
        msg = {"content": "<think>reasoning</think>visible"}
        blocks = _build_content_blocks(msg)
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "reasoning"
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "visible"

    def test_clean_content(self):
        """content 没有 think tag，保持不变"""
        from llm_proxy.protocol.anthropic_openai.response import _build_content_blocks
        msg = {"content": "just text"}
        blocks = _build_content_blocks(msg)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "just text"

    def test_strip_think_tags_double(self):
        reasoning, content = strip_think_tags("<think>hidden</think>visible")
        assert reasoning == "hidden"
        assert content == "visible"

    def test_strip_think_tags_content_only(self):
        reasoning, content = strip_think_tags("just text")
        assert reasoning == ""
        assert content == "just text"

    def test_strip_think_tags_clean(self):
        reasoning, content = strip_think_tags("")
        assert reasoning == ""
        assert content == ""

    def test_responses_to_responses_with_think_tag(self):
        """to_responses_response 正确处理 think tag"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "content": "<think>think</think>text",
                },
                "finish_reason": "stop",
            }],
            "model": "test-model",
        }
        resp = to_responses_response(chat_body, "test-model")
        outputs = resp["output"]
        assert outputs[0]["type"] == "reasoning"
        assert outputs[0]["summary"][0]["text"] == "think"
        assert outputs[1]["type"] == "message"
        assert outputs[1]["content"][0]["text"] == "text"

    def test_responses_to_responses_clean(self):
        """to_responses_response 在无 think tag 时不变"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "content": "plain text",
                },
                "finish_reason": "stop",
            }],
            "model": "test-model",
        }
        resp = to_responses_response(chat_body, "test-model")
        outputs = resp["output"]
        assert len(outputs) == 1
        assert outputs[0]["type"] == "message"

    def test_needs_reasoning_split(self):
        from llm_proxy.handlers.shared.proxy import _needs_reasoning_split
        assert _needs_reasoning_split("https://api.minimaxi.com") is True
        assert _needs_reasoning_split("https://api.openai.com") is False
        assert _needs_reasoning_split("") is False
        assert _needs_reasoning_split("https://api.MiniMaxi.com/v1") is True
