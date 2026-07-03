"""apply_patch 在 IR 通道的端到端测试。

覆盖两个症状：
- 症状 A：Codex 发回的 apply_patch 工具调用历史（custom_tool_call +
  custom_tool_call_output）应正确转换为 IRToolUseBlock + IRToolResultBlock，
  这样下游 IR Chat 转换能转发给上游，模型能看到工具结果。
- 症状 B：上游输出的不完整 DSL（缺 *** End Patch）应被
  repair_apply_patch_dsl 自动修复。
"""

import json

import pytest

from llm_proxy.protocol.ir.responses import to_ir
from llm_proxy.protocol.ir.types import (
    IRMessage,
    IRToolResultBlock,
    IRToolUseBlock,
)
from llm_proxy.protocol.responses_chat.tool_replacement import repair_apply_patch_dsl


# ────────────────────────────────────────────────────────────────────
# 症状 A：custom_tool_call_output 必须被 IR 通道正确转发
# ────────────────────────────────────────────────────────────────────


def _build_codex_apply_patch_history() -> list[dict]:
    """Codex 在多轮对话中发回的 apply_patch 历史。"""
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Add a function to foo.py"}],
        },
        {
            "type": "custom_tool_call",
            "call_id": "call_apply_1",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** Add File: foo.py\n+def hello():\n+    return 1\n*** End Patch",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "call_apply_1",
            "output": "File created successfully at: foo.py",
        },
    ]


class TestCustomToolCallForwarding:
    """IR 通道必须正确处理 custom_tool_call + custom_tool_call_output。"""

    def test_custom_tool_call_creates_ir_tool_use_block(self):
        body = {"model": "gpt-5", "input": _build_codex_apply_patch_history()}
        ir = to_ir(body)

        tool_uses = [
            b for m in ir.messages for b in m.content
            if isinstance(b, IRToolUseBlock)
        ]
        apply_blocks = [b for b in tool_uses if b.name == "apply_patch"]
        assert len(apply_blocks) == 1
        assert apply_blocks[0].id == "call_apply_1"

    def test_custom_tool_call_input_wrapped_in_dict(self):
        """apply_patch 的 DSL input 被解析为结构化参数（含 action 字段），
        匹配 Chat Completions 协议（tool call 的 arguments 是 JSON 对象）。"""
        body = {"model": "gpt-5", "input": _build_codex_apply_patch_history()}
        ir = to_ir(body)

        apply_block = next(
            b for m in ir.messages for b in m.content
            if isinstance(b, IRToolUseBlock) and b.name == "apply_patch"
        )
        assert isinstance(apply_block.input, dict)
        assert apply_block.input.get("action") == "add_file"
        assert apply_block.input.get("filePath") == "foo.py"
        assert "def hello():" in apply_block.input.get("content", "")

    def test_custom_tool_call_output_creates_ir_tool_result_block(self):
        """这是症状 A 的核心：custom_tool_call_output 必须被转换为
        IRToolResultBlock，让下游能把工具结果发给上游。"""
        body = {"model": "gpt-5", "input": _build_codex_apply_patch_history()}
        ir = to_ir(body)

        tool_results = [
            b for m in ir.messages if m.role == "tool"
            for b in (m.content if isinstance(m.content, list) else [])
            if isinstance(b, IRToolResultBlock)
        ]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "call_apply_1"
        assert "File created successfully" in tool_results[0].content

    def test_output_dict_with_text_field_extracted(self):
        """Codex 可能发 dict 形式的 output：{'text': '...'}。"""
        history = [
            {
                "type": "custom_tool_call",
                "call_id": "call_apply_2",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** Add File: a.py\n+x\n*** End Patch",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_apply_2",
                "output": {"text": "ok"},
            },
        ]
        body = {"model": "gpt-5", "input": history}
        ir = to_ir(body)

        result = next(
            b for m in ir.messages if m.role == "tool"
            for b in (m.content if isinstance(m.content, list) else [])
            if isinstance(b, IRToolResultBlock) and b.tool_use_id == "call_apply_2"
        )
        assert result.content == "ok"


# ────────────────────────────────────────────────────────────────────
# 症状 B：缺 *** End Patch 必须被自动修复
# ────────────────────────────────────────────────────────────────────


class TestRepairApplyPatchDslReenabled:
    """修复症状 B：repair_apply_patch_dsl 不能是 bypass。"""

    def test_missing_end_patch_is_appended(self):
        """上游最常见的错误：缺 *** End Patch。"""
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+line1\n+line2"
        result = repair_apply_patch_dsl(dsl)
        assert result.was_repaired
        assert "appended missing *** End Patch" in result.repairs
        assert result.dsl.rstrip().endswith("*** End Patch")

    def test_missing_begin_patch_is_inserted(self):
        dsl = "*** Add File: /a.py\n+x\n*** End Patch"
        result = repair_apply_patch_dsl(dsl)
        assert result.was_repaired
        assert result.dsl.startswith("*** Begin Patch")

    def test_end_lowercase_is_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** end patch"
        result = repair_apply_patch_dsl(dsl)
        assert "*** end patch" not in result.dsl
        assert "*** End Patch" in result.dsl

    def test_unified_diff_hunk_header_is_normalized(self):
        dsl = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@ -19,5 +19,6 @@\n"
            "-a\n"
            "+b\n"
            "*** End Patch"
        )
        result = repair_apply_patch_dsl(dsl)
        assert "@@ -19,5" not in result.dsl
        assert "*** End Patch" in result.dsl

    def test_complete_dsl_passes_through_unchanged(self):
        """完整 DSL 不应触发任何修复。"""
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** End Patch"
        result = repair_apply_patch_dsl(dsl)
        assert not result.was_repaired
        assert result.dsl == dsl
