"""Tests for llm_proxy.adapters.tool_replacement."""

import pytest
from llm_proxy.protocol.responses_chat.tool_replacement import (
    WRITE_TOOL_DEF,
    REPLACE_TOOL_DEF,
    DELETE_TOOL_DEF,
    parse_apply_patch_to_simple,
    write_to_apply_patch,
    delete_to_apply_patch,
    replace_to_apply_patch,
    build_reverse_tool_map,
    reverse_tool_args_to_apply_patch,
    ReverseConversionError,
)


class TestParseApplyPatchToSimple:
    """parse_apply_patch_to_simple 现在返回 list[dict] | None。"""

    def test_add_file_basic(self):
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n*** Add File: /tmp/a.py\n+print(1)\n*** End Patch"
        )
        assert result == [
            {"tool": "write_to_file", "args": {"filePath": "/tmp/a.py", "content": "print(1)"}},
        ]

    def test_add_file_multiline(self):
        result = parse_apply_patch_to_simple(
            "*** Add File: /tmp/a.py\n+def foo():\n+    return 1\n*** End Patch"
        )
        assert result is not None and len(result) == 1
        assert result[0]["tool"] == "write_to_file"
        assert result[0]["args"]["filePath"] == "/tmp/a.py"
        assert result[0]["args"]["content"] == "def foo():\n    return 1"

    def test_add_file_without_end_marker(self):
        result = parse_apply_patch_to_simple(
            "*** Add File: /tmp/a.py\n+print(1)"
        )
        assert result == [
            {"tool": "write_to_file", "args": {"filePath": "/tmp/a.py", "content": "print(1)"}},
        ]

    def test_multi_file_no_begin_no_end(self):
        """完全裸格式：多文件，无 Begin Patch 也无 End Patch。"""
        result = parse_apply_patch_to_simple(
            "*** Add File: /tmp/a.py\n+print(1)\n"
            "*** Update File: /tmp/b.py\n@@\n-old\n+new\n"
            "*** Delete File: /tmp/c.py"
        )
        assert result is not None and len(result) == 3
        assert result[0] == {"tool": "write_to_file", "args": {"filePath": "/tmp/a.py", "content": "print(1)"}}
        assert result[1] == {"tool": "replace_in_file", "args": {"filePath": "/tmp/b.py", "old_str": "old", "new_str": "new"}}
        assert result[2] == {"tool": "delete_file", "args": {"filePath": "/tmp/c.py"}}

    def test_update_file_basic(self):
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n*** Update File: /tmp/a.py\n@@\n-old_line\n+new_line\n*** End Patch"
        )
        assert result == [
            {"tool": "replace_in_file", "args": {"filePath": "/tmp/a.py", "old_str": "old_line", "new_str": "new_line"}},
        ]

    def test_update_file_multiline(self):
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n@@\n-line1\n-line2\n+lineA\n+lineB\n*** End Patch"
        )
        assert result is not None and len(result) == 1
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["filePath"] == "/tmp/a.py"
        assert result[0]["args"]["old_str"] == "line1\nline2"
        assert result[0]["args"]["new_str"] == "lineA\nlineB"

    def test_delete_file(self):
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n*** Delete File: /tmp/a.py\n*** End Patch"
        )
        assert result == [
            {"tool": "delete_file", "args": {"filePath": "/tmp/a.py"}},
        ]

    def test_multi_file_patch(self):
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Add File: /tmp/a.py\n+print(1)\n"
            "*** Update File: /tmp/b.py\n@@\n-old\n+new\n"
            "*** Delete File: /tmp/c.py\n"
            "*** End Patch"
        )
        assert result is not None
        assert len(result) == 3
        assert result[0] == {"tool": "write_to_file", "args": {"filePath": "/tmp/a.py", "content": "print(1)"}}
        assert result[1] == {"tool": "replace_in_file", "args": {"filePath": "/tmp/b.py", "old_str": "old", "new_str": "new"}}
        assert result[2] == {"tool": "delete_file", "args": {"filePath": "/tmp/c.py"}}

    def test_empty_input_returns_none(self):
        assert parse_apply_patch_to_simple("") is None
        assert parse_apply_patch_to_simple(None) is None

    def test_malformed_returns_none(self):
        assert parse_apply_patch_to_simple("not a patch") is None
        assert parse_apply_patch_to_simple("*** Begin Patch\n*** End Patch") is None

    def test_ignores_context_and_at_lines(self):
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n@@\n keep\n-old\n+new\n keep2\n*** End Patch"
        )
        assert result is not None and len(result) == 1
        assert result[0]["args"]["old_str"] == "old"
        assert result[0]["args"]["new_str"] == "new"

    def test_update_only_plus_lines_converts_to_append(self):
        """Update File 只有 '+' 行 → 转为 append_to_file。"""
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n@@\n+def new_func():\n+    pass\n*** End Patch"
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["tool"] == "append_to_file"
        assert "def new_func():" in result[0]["args"]["content"]
        assert result[0]["args"]["filePath"] == "/tmp/a.py"

    def test_update_only_plus_lines_mixed_with_normal(self):
        """多文件 patch 中一个段只有 '+' 行，其他段正常。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Add File: /tmp/a.py\n+print(1)\n"
            "*** Update File: /tmp/b.py\n@@\n+appended\n"
            "*** End Patch"
        )
        assert result is not None
        assert len(result) == 2
        # 第一个段正常
        assert result[0]["tool"] == "write_to_file"
        # 第二个段转为 append_to_file
        assert result[1]["tool"] == "append_to_file"

    def test_move_to_with_content(self):
        """Update File + Move to + diff → replace_in_file 含 destinationPath。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Update File: /tmp/old.py\n"
            "*** Move to: /tmp/new.py\n"
            "@@\n-old_code\n+new_code\n"
            "*** End Patch"
        )
        assert result is not None and len(result) == 1
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["filePath"] == "/tmp/old.py"
        assert result[0]["args"]["destinationPath"] == "/tmp/new.py"
        assert result[0]["args"]["old_str"] == "old_code"
        assert result[0]["args"]["new_str"] == "new_code"

    def test_move_to_no_content(self):
        """Update File + Move to，无 diff 行 → replace_in_file 含 destinationPath，占位 old_str/new_str。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Update File: /tmp/old.py\n"
            "*** Move to: /tmp/new.py\n"
            "*** End Patch"
        )
        assert result is not None and len(result) == 1
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["filePath"] == "/tmp/old.py"
        assert result[0]["args"]["destinationPath"] == "/tmp/new.py"
        assert result[0]["args"]["old_str"] == " "
        assert result[0]["args"]["new_str"] == " "

    def test_move_to_only_no_diff(self):
        """Update File + Move to，无任何 diff 和 context → 仍应生成 replace_in_file。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Update File: src/app.py\n"
            "*** Move to: src/main.py\n"
            "*** End Patch"
        )
        assert result is not None and len(result) == 1
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["destinationPath"] == "src/main.py"

    def test_update_with_move_to_empty_old_str(self):
        """Update File + Move to + 只有 '+' 行 → 降级为 user message（append_to_file 不支持 Move to）。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Update File: /tmp/old.py\n"
            "*** Move to: /tmp/new.py\n"
            "@@\n+new_line\n"
            "*** End Patch"
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["tool"] == "_degraded_user_message"
        assert "new_line" in result[0]["args"]["content"]

    def test_end_of_file_marker(self):
        """*** End of File 标记正确分割段。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Update File: /tmp/a.py\n@@\n-old1\n+new1\n"
            "*** End of File\n"
            "*** Update File: /tmp/b.py\n@@\n-old2\n+new2\n"
            "*** End Patch"
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["args"]["filePath"] == "/tmp/a.py"
        assert result[0]["args"]["old_str"] == "old1"
        assert result[1]["args"]["filePath"] == "/tmp/b.py"
        assert result[1]["args"]["old_str"] == "old2"

    def test_update_only_plus_with_context_uses_context_as_anchor(self):
        """有 context 行但无 - 行 → 用 context 行构造 replace_in_file（对齐 Claude Code）。"""
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n@@\n existing line\n+new line\n*** End Patch"
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["old_str"] == "existing line"
        assert result[0]["args"]["new_str"] == "existing line\nnew line"

    def test_update_only_plus_with_multiline_context(self):
        """多行 context + 多行追加。"""
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n@@\n line1\n line2\n+new1\n+new2\n*** End Patch"
        )
        assert result is not None
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["old_str"] == "line1\nline2"
        assert result[0]["args"]["new_str"] == "line1\nline2\nnew1\nnew2"

    def test_update_only_plus_with_context_and_move_to(self):
        """context + Move to + 只有+行。"""
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n*** Move to: /tmp/b.py\n@@\n existing\n+appended\n*** End Patch"
        )
        assert result is not None
        assert result[0]["tool"] == "replace_in_file"
        assert result[0]["args"]["old_str"] == "existing"
        assert result[0]["args"]["new_str"] == "existing\nappended"
        assert result[0]["args"]["destinationPath"] == "/tmp/b.py"


class TestWriteToApplyPatch:
    def test_basic(self):
        result = write_to_apply_patch("/tmp/a.py", "print(1)")
        assert "*** Begin Patch" in result
        assert "*** Add File: /tmp/a.py" in result
        assert "+print(1)" in result
        assert "*** End Patch" in result

    def test_multiline(self):
        result = write_to_apply_patch("/tmp/a.py", "line1\nline2")
        assert "+line1\n+line2" in result


class TestDeleteToApplyPatch:
    def test_basic(self):
        result = delete_to_apply_patch("/tmp/a.py")
        assert "*** Begin Patch" in result
        assert "*** Delete File: /tmp/a.py" in result
        assert "*** End Patch" in result


class TestReplaceToApplyPatch:
    def test_basic_no_context(self):
        """old_str / new_str have no common prefix/suffix — no context lines."""
        result = replace_to_apply_patch("/tmp/a.py", "old", "new")
        assert "*** Begin Patch" in result
        assert "*** Update File: /tmp/a.py" in result
        assert "-old" in result
        assert "+new" in result
        assert "*** End Patch" in result

    def test_with_context_single_line_change(self):
        """Model included surrounding context in both old_str and new_str."""
        result = replace_to_apply_patch(
            "/tmp/a.py",
            "line1\nold2\nline3",
            "line1\nnew2\nline3",
        )
        lines = result.split("\n")
        # Verify context lines (space prefix)
        assert " line1" in result
        assert " line3" in result
        # Verify the change
        assert "-old2" in result
        assert "+new2" in result
        # Verify order: context before, diff, context after
        space_idx = [i for i, ln in enumerate(lines) if ln.startswith(" ")]
        minus_idx = [i for i, ln in enumerate(lines) if ln.startswith("-")]
        plus_idx = [i for i, ln in enumerate(lines) if ln.startswith("+")]
        assert len(space_idx) == 2
        assert space_idx[0] < minus_idx[0]  # context before change
        assert minus_idx[0] < space_idx[1]  # change before context after
        assert plus_idx[0] > minus_idx[0]   # + after -

    def test_pure_deletion(self):
        """Model removes a line, keeps surrounding context."""
        result = replace_to_apply_patch(
            "/tmp/a.py",
            "line1\nline2\nline3",
            "line1\nline3",
        )
        assert " line1" in result
        assert " line3" in result
        assert "-line2" in result
        assert "+" not in result.replace("*** Add File:", "").replace("*** Update File:", "").split("\n")[0]  # no + lines

    def test_pure_addition(self):
        """Model adds a line, keeps surrounding context."""
        result = replace_to_apply_patch(
            "/tmp/a.py",
            "line1\nline3",
            "line1\nline2\nline3",
        )
        assert " line1" in result
        assert " line3" in result
        assert "+line2" in result
        # No - lines (after removing context lines' leading space)
        non_context = [ln for ln in result.split("\n") if not ln.startswith(" ") and ln not in ("@@",)]
        assert not any(ln.startswith("-") for ln in non_context)

    def test_multiline_change_with_context(self):
        """Multiple lines changed, with context on both sides."""
        result = replace_to_apply_patch(
            "/tmp/a.py",
            "ctx1\nold1\nold2\nold3\nctx2",
            "ctx1\nnew1\nnew2\nctx2",
        )
        assert " ctx1" in result
        assert " ctx2" in result
        assert "-old1" in result
        assert "-old2" in result
        assert "-old3" in result
        assert "+new1" in result
        assert "+new2" in result

    def test_line_endings_preserved(self):
        """Content with trailing newlines handled correctly."""
        result = replace_to_apply_patch(
            "/tmp/a.py",
            "line1\nline2\n",
            "line1\nline2_new\n",
        )
        assert " line1" in result
        # The trailing empty string from split should be a common suffix line
        assert "+line2_new" in result
        assert "-line2" in result

    def test_with_destination_path(self):
        """dest_path 非空时，输出包含 *** Move to: 行。"""
        result = replace_to_apply_patch(
            "/tmp/old.py", "old_code", "new_code", dest_path="/tmp/new.py"
        )
        assert "*** Update File: /tmp/old.py" in result
        assert "*** Move to: /tmp/new.py" in result
        assert "-old_code" in result
        assert "+new_code" in result
        assert "*** End Patch" in result
        # Move to 行应在 Update File 行之后、@@ 之前
        lines = result.split("\n")
        update_idx = next(i for i, ln in enumerate(lines) if ln.startswith("*** Update File"))
        move_idx = next(i for i, ln in enumerate(lines) if ln.startswith("*** Move to"))
        at_idx = next(i for i, ln in enumerate(lines) if ln.startswith("@@"))
        assert update_idx < move_idx < at_idx

    def test_without_destination_path(self):
        """dest_path 为 None 时，不含 *** Move to: 行。"""
        result = replace_to_apply_patch("/tmp/a.py", "old", "new")
        assert "*** Move to:" not in result


class TestBuildReverseToolMap:
    def test_returns_expected(self):
        result = build_reverse_tool_map()
        assert result == {
            "write_to_file": "apply_patch",
            "replace_in_file": "apply_patch",
            "delete_file": "apply_patch",
            "append_to_file": "apply_patch",
        }


class TestReverseToolArgsToApplyPatch:
    def test_write_to_file(self):
        result = reverse_tool_args_to_apply_patch(
            "write_to_file",
            {"filePath": "/tmp/a.py", "content": "print(1)"},
        )
        assert "*** Add File: /tmp/a.py" in result
        assert "+print(1)" in result

    def test_write_to_file_alt_keys(self):
        result = reverse_tool_args_to_apply_patch(
            "write_to_file",
            {"file_path": "/tmp/a.py", "content": "x"},
        )
        assert "*** Add File: /tmp/a.py" in result

    def test_replace_in_file(self):
        result = reverse_tool_args_to_apply_patch(
            "replace_in_file",
            {"filePath": "/tmp/a.py", "old_str": "old", "new_str": "new"},
        )
        assert "*** Update File: /tmp/a.py" in result
        assert "-old" in result
        assert "+new" in result

    def test_replace_with_destination(self):
        result = reverse_tool_args_to_apply_patch(
            "replace_in_file",
            {"filePath": "/tmp/old.py", "old_str": "old", "new_str": "new", "destinationPath": "/tmp/new.py"},
        )
        assert "*** Update File: /tmp/old.py" in result
        assert "*** Move to: /tmp/new.py" in result
        assert "-old" in result
        assert "+new" in result

    def test_replace_without_destination(self):
        result = reverse_tool_args_to_apply_patch(
            "replace_in_file",
            {"filePath": "/tmp/a.py", "old_str": "old", "new_str": "new"},
        )
        assert "*** Move to:" not in result

    def test_delete_file(self):
        result = reverse_tool_args_to_apply_patch(
            "delete_file",
            {"filePath": "/tmp/a.py"},
        )
        assert "*** Delete File: /tmp/a.py" in result
        assert "*** End Patch" in result

    def test_delete_file_alt_keys(self):
        result = reverse_tool_args_to_apply_patch(
            "delete_file",
            {"file_path": "/tmp/b.py"},
        )
        assert "*** Delete File: /tmp/b.py" in result

    def test_missing_args_raises_error(self):
        with pytest.raises(ReverseConversionError):
            reverse_tool_args_to_apply_patch("write_to_file", {"filePath": "/tmp/a.py"})

    def test_unknown_tool_raises_error(self):
        with pytest.raises(ReverseConversionError):
            reverse_tool_args_to_apply_patch("unknown_tool", {})

    def test_replace_in_file_empty_old_str_raises_error(self):
        with pytest.raises(ReverseConversionError) as exc_info:
            reverse_tool_args_to_apply_patch(
                "replace_in_file",
                {"filePath": "/tmp/a.py", "old_str": "", "new_str": "new"},
            )
        assert "old_str must not be empty" in str(exc_info.value)

    def test_delete_file_missing_path_raises_error(self):
        with pytest.raises(ReverseConversionError):
            reverse_tool_args_to_apply_patch("delete_file", {})


class TestToolDefs:
    def test_write_tool_def(self):
        assert WRITE_TOOL_DEF["type"] == "function"
        fn = WRITE_TOOL_DEF["function"]
        assert fn["name"] == "write_to_file"
        assert "filePath" in fn["parameters"]["properties"]
        assert "content" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["filePath", "content"]
        # 安全加固：描述明确语义
        assert "overwritten" in fn["description"] or "overwrite" in fn["description"]

    def test_replace_tool_def(self):
        assert REPLACE_TOOL_DEF["type"] == "function"
        fn = REPLACE_TOOL_DEF["function"]
        assert fn["name"] == "replace_in_file"
        assert "filePath" in fn["parameters"]["properties"]
        assert "old_str" in fn["parameters"]["properties"]
        assert "new_str" in fn["parameters"]["properties"]
        assert fn["parameters"]["properties"]["old_str"].get("minLength") == 1
        # 安全加固：destinationPath 可选参数
        assert "destinationPath" in fn["parameters"]["properties"]
        assert "destinationPath" not in fn["parameters"]["required"]
        # 安全加固：描述引导模型行为
        assert "unique" in fn["description"].lower()

    def test_delete_tool_def(self):
        assert DELETE_TOOL_DEF["type"] == "function"
        fn = DELETE_TOOL_DEF["function"]
        assert fn["name"] == "delete_file"
        assert "filePath" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["filePath"]
        # 安全加固：描述强调破坏性
        assert "destructive" in fn["description"].lower()


class TestValidationSafety:
    """校验安全：借鉴 Claude Code 方案，对转换产出的参数做完整性校验。"""

    def test_add_file_empty_path_returns_degraded(self):
        """Add File 空 filePath → 降级段（不影响其他段）。"""
        result = parse_apply_patch_to_simple("*** Add File: \n+content\n*** End Patch")
        assert result is not None
        assert result[0]["tool"] == "_degraded_user_message"

    def test_update_file_empty_path_returns_none(self):
        """Update File 空 filePath → 降级段（不影响其他段）。"""
        result = parse_apply_patch_to_simple("*** Update File: \n@@\n-old\n+new\n*** End Patch")
        assert result is not None
        assert result[0]["tool"] == "_degraded_user_message"

    def test_delete_file_empty_path_returns_degraded(self):
        """Delete File 空 filePath → 降级段（不影响其他段）。"""
        result = parse_apply_patch_to_simple("*** Delete File: \n*** End Patch")
        assert result is not None
        assert result[0]["tool"] == "_degraded_user_message"

    def test_add_file_no_plus_lines_returns_degraded(self):
        """Add File 没有 '+' 行（空 content）→ 降级段。"""
        result = parse_apply_patch_to_simple("*** Add File: /tmp/a.py\n*** End Patch")
        assert result is not None
        assert result[0]["tool"] == "_degraded_user_message"

    def test_add_file_empty_plus_line_is_ok(self):
        """Add File 有 '+' 行但内容为空字符串（创建空文件）→ 合法。"""
        result = parse_apply_patch_to_simple("*** Add File: /tmp/a.py\n+\n*** End Patch")
        assert result is not None
        assert result[0]["tool"] == "write_to_file"
        assert result[0]["args"]["content"] == ""

    def test_reverse_write_empty_content_raises_error(self):
        """反向转换：write_to_file content 为空字符串 → ReverseConversionError。"""
        with pytest.raises(ReverseConversionError) as exc_info:
            reverse_tool_args_to_apply_patch("write_to_file", {"filePath": "/tmp/a.py", "content": ""})
        assert "content must not be empty" in str(exc_info.value)

    def test_reverse_replace_empty_old_str_raises_error(self):
        """反向转换：replace_in_file old_str 为空 → ReverseConversionError（已存在，验证仍在）。"""
        with pytest.raises(ReverseConversionError) as exc_info:
            reverse_tool_args_to_apply_patch("replace_in_file", {
                "filePath": "/tmp/a.py", "old_str": "", "new_str": "new"
            })
        assert "old_str must not be empty" in str(exc_info.value)

    def test_update_only_plus_lines_single_segment_converts_to_append(self):
        """单段 only-plus Update File → append_to_file。"""
        result = parse_apply_patch_to_simple(
            "*** Update File: /tmp/a.py\n@@\n+appended\n*** End Patch"
        )
        assert result is not None
        assert result[0]["tool"] == "append_to_file"
        assert result[0]["args"]["filePath"] == "/tmp/a.py"
        assert result[0]["args"]["content"] == "appended"

    def test_update_only_plus_lines_mixed_preserves_normal(self):
        """多段 patch 中正常段不受影响。"""
        result = parse_apply_patch_to_simple(
            "*** Begin Patch\n"
            "*** Add File: /tmp/a.py\n+print(1)\n"
            "*** Update File: /tmp/b.py\n@@\n+appended\n"
            "*** End Patch"
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["tool"] == "write_to_file"
        assert result[1]["tool"] == "append_to_file"
