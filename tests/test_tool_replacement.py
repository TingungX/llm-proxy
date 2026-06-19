"""Tests for llm_proxy.protocol.responses_chat.tool_replacement."""

from llm_proxy.protocol.responses_chat.tool_replacement import (
    APPLY_PATCH_TOOL_DESCRIPTION,
    RepairResult,
    repair_apply_patch_dsl,
)


class TestApplyPatchToolDescription:
    """APPLY_PATCH_TOOL_DESCRIPTION 内容契约。"""

    def test_is_nonempty_string(self):
        assert isinstance(APPLY_PATCH_TOOL_DESCRIPTION, str)
        assert len(APPLY_PATCH_TOOL_DESCRIPTION) > 50

    def test_mentions_add_update_delete_actions(self):
        for kw in ("Add File", "Update File", "Delete File"):
            assert kw in APPLY_PATCH_TOOL_DESCRIPTION

    def test_mentions_plus_prefix_rule(self):
        assert "plus" in APPLY_PATCH_TOOL_DESCRIPTION.lower()
        assert "minus" in APPLY_PATCH_TOOL_DESCRIPTION.lower()

    def test_mentions_move_to_rename(self):
        assert "Move to" in APPLY_PATCH_TOOL_DESCRIPTION

    def test_under_1500_characters(self):
        assert len(APPLY_PATCH_TOOL_DESCRIPTION) < 1500


def _dsl(result) -> str:
    """helper: extract .dsl from RepairResult"""
    if isinstance(result, RepairResult):
        return result.dsl
    return result  # backward compat for any leftover tests


def _repairs(result) -> list:
    if isinstance(result, RepairResult):
        return result.repairs
    return []


class TestRepairApplyPatchDsl:
    """repair_apply_patch_dsl 鲁棒性测试。"""

    # === 基本 Marker 修复 ===

    def test_both_begin_and_end_preserved(self):
        dsl = "*** Begin Patch\n*** Add File: /tmp/a.py\n+x\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert _dsl(r) == dsl
        assert not r.was_repaired

    def test_missing_end_appended(self):
        dsl = "*** Begin Patch\n*** Add File: /tmp/a.py\n+x"
        r = repair_apply_patch_dsl(dsl)
        assert _dsl(r).rstrip().endswith("*** End Patch")
        assert "appended missing *** End Patch" in r.repairs

    def test_missing_begin_inserted_before_first_header(self):
        dsl = "*** Add File: /tmp/a.py\n+x\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        # 移除可能的 # 注记行后,标准 DSL 开头
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert dsl_part.startswith("*** Begin Patch\n")
        assert "inserted missing *** Begin Patch" in r.repairs

    def test_missing_both_wrap_extracted_headers(self):
        dsl = "*** Add File: /tmp/a.py\n+x"
        r = repair_apply_patch_dsl(dsl)
        # B2：注记在尾部，按 # 前缀切
        dsl_part = r.dsl.split("\n# llm-proxy", 1)[0] if r.was_repaired else r.dsl
        assert dsl_part.startswith("*** Begin Patch\n")
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert "*** Add File: /tmp/a.py" in dsl_part
        assert "wrapped with Begin/End markers" in r.repairs

    def test_speech_text_around_patch_trimmed(self):
        dsl = "Here is the patch:\n*** Begin Patch\n*** Add File: /tmp/a.py\n+x\n*** End Patch\nHope that helps!"
        r = repair_apply_patch_dsl(dsl)
        dsl_part = r.dsl.split("\n# llm-proxy", 1)[0] if r.was_repaired else r.dsl
        assert dsl_part.startswith("*** Begin Patch")
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert "Here is the patch" not in dsl_part
        assert "Hope that helps" not in dsl_part
        assert "trimmed surrounding text" in r.repairs

    def test_case_and_whitespace_tolerance(self):
        dsl = "** * BEGIN  patch\n*** Add File: /tmp/a.py\n+x\n*** END  Patch"
        r = repair_apply_patch_dsl(dsl)
        # 归一化后：标准 *** Begin/End Patch
        assert "*** Begin Patch" in _dsl(r)
        assert "*** End Patch" in _dsl(r)
        assert "** * BEGIN  patch" not in _dsl(r)
        assert "*** END  Patch" not in _dsl(r)
        # Begin 和 End 都被归一化
        assert "normalized Begin marker" in r.repairs
        assert "normalized End marker" in r.repairs

    def test_duplicate_end_keeps_single(self):
        dsl = "*** Begin Patch\n*** Add File: /tmp/a.py\n+x\n*** End Patch\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert _dsl(r).count("*** End Patch") == 1
        # B2：注记在尾部，标准 DSL 部分以 End Patch 结尾
        dsl_part = _dsl(r).split("\n# llm-proxy", 1)[0]
        assert dsl_part.rstrip().endswith("*** End Patch")

    def test_no_file_headers_returns_unchanged(self):
        dsl = "no markers here, just text"
        r = repair_apply_patch_dsl(dsl)
        assert not r.was_repaired
        assert _dsl(r) == dsl

    def test_empty_input_returns_empty(self):
        r1 = repair_apply_patch_dsl("")
        r2 = repair_apply_patch_dsl("   \n  ")
        assert r1.dsl == ""
        assert r2.dsl == ""
        assert not r1.was_repaired
        assert not r2.was_repaired

    def test_none_input_returns_empty_repair_result(self):
        r = repair_apply_patch_dsl(None)
        # None 输入返回 RepairResult(dsl="", repairs=[]) — 调用方可以安全地访问 .dsl
        assert r.dsl == ""
        assert r.repairs == []
        assert not r.was_repaired

    def test_only_begin_no_content(self):
        dsl = "*** Begin Patch\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert not r.was_repaired  # 无文件头，无法修复
        assert r.dsl == dsl

    def test_multiple_begin(self):
        dsl = "*** Begin Patch\n*** Begin Patch\n*** Add File: /tmp/a.py\n+x\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert dsl_part.startswith("*** Begin Patch")
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert dsl_part.count("*** End Patch") == 1

    def test_newlines_before_begin(self):
        dsl = "\n\n\n*** Begin Patch\n*** Add File: /tmp/a.py\n+x\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        # 无修复时无 # 注记
        assert r.dsl == dsl.strip() or r.dsl.startswith("*** Begin Patch")

    def test_newlines_after_end(self):
        dsl = "*** Begin Patch\n*** Add File: /tmp/a.py\n+x\n*** End Patch\n\n\n"
        r = repair_apply_patch_dsl(dsl)
        # 实际 DSL 部分（去掉可能的注记）应该无尾随空行
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert dsl_part == dsl_part.strip()

    # === 模型"忘记 End Patch"的所有变体 — 最常见的错误 ===

    def test_missing_end_pure(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+line1\n+line2"
        r = repair_apply_patch_dsl(dsl)
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert dsl_part.startswith("*** Begin Patch")
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert "appended missing *** End Patch" in r.repairs

    def test_missing_end_multi_file(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+line1\n*** Update File: /b.py\n@@\n-old\n+new"
        r = repair_apply_patch_dsl(dsl)
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert "*** Add File: /a.py" in dsl_part
        assert "*** Update File: /b.py" in dsl_part

    def test_missing_end_with_trailing_text(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n\nThis is the new content."
        r = repair_apply_patch_dsl(dsl)
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert dsl_part.rstrip().endswith("*** End Patch")

    def test_end_of_file_not_end_patch(self):
        # *** End of File 是合法的段内结束标记
        dsl = "*** Begin Patch\n*** Update File: /a.py\n@@\n-old\n+new\n*** End of File"
        r = repair_apply_patch_dsl(dsl)
        dsl_part = r.dsl.split("\n\n", 1)[-1] if r.was_repaired else r.dsl
        assert "*** End of File" in dsl_part
        assert dsl_part.rstrip().endswith("*** End Patch")
        assert "appended missing *** End Patch" in r.repairs

    def test_end_lowercase_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** end patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** end patch" not in _dsl(r)
        assert "*** End Patch" in _dsl(r)
        assert "normalized End marker" in r.repairs

    def test_end_no_space_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** EndPatch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** EndPatch" not in _dsl(r)
        assert "*** End Patch" in _dsl(r)

    def test_end_uppercase_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** END PATCH"
        r = repair_apply_patch_dsl(dsl)
        assert "*** END PATCH" not in _dsl(r)
        assert "*** End Patch" in _dsl(r)

    def test_end_of_patch_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** End of Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** End of Patch" not in _dsl(r)
        assert "*** End Patch" in _dsl(r)

    def test_end_hash_prefix_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n### End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "### End Patch" not in _dsl(r)
        assert "*** End Patch" in _dsl(r)

    def test_end_underscore_normalized(self):
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** End_of_Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** End_of_Patch" not in _dsl(r)
        assert "*** End Patch" in _dsl(r)


class TestHunkHeaderNormalization:
    """@@ hunk header 归一化：模型可能误用 unified-diff 语法或 anchor 语法。"""

    def test_bare_atat_preserved(self):
        """裸 @@（无尾随内容）不应被修复。"""
        dsl = "*** Begin Patch\n*** Update File: foo.py\n@@\n-a\n+b\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert not any("@@" in rep and "hunk" in rep for rep in r.repairs)

    def test_unified_diff_style_atat_normalized(self):
        """@@ -19,5 +19,6 @@ → @@（unified-diff 误用）。"""
        dsl = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@ -19,5 +19,6 @@\n"
            "-a\n"
            "+b\n"
            "*** End Patch"
        )
        r = repair_apply_patch_dsl(dsl)
        dsl_body = _dsl(r)
        assert "@@ -19" not in dsl_body
        assert "+19,6" not in dsl_body
        assert "@@\n" in dsl_body
        assert any("@@" in rep and "hunk header" in rep for rep in r.repairs)

    def test_atat_with_function_anchor_normalized(self):
        """@@ def some_function: → @@（anchor 误用）。"""
        dsl = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@ def some_function:\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )
        r = repair_apply_patch_dsl(dsl)
        dsl_body = _dsl(r)
        assert "@@ def" not in dsl_body
        assert "some_function" not in dsl_body

    def test_multiple_bad_atat_all_normalized(self):
        """多个坏 @@ 行全部归一化，注记含数量。"""
        dsl = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-a\n"
            "+b\n"
            "@@ -10,3 +10,3 @@\n"
            "-c\n"
            "+d\n"
            "*** End Patch"
        )
        r = repair_apply_patch_dsl(dsl)
        dsl_body = _dsl(r)
        assert dsl_body.count("@@\n") >= 2
        assert "-1,3" not in dsl_body
        assert "-10,3" not in dsl_body
        assert any(rep == "normalized 2 @@ hunk headers" for rep in r.repairs)

    def test_added_line_with_atat_preserved(self):
        """hunk body 中 +@@marker 不应被误判为 hunk header。"""
        dsl = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@\n"
            "-old_marker\n"
            "+@@new_marker\n"
            "*** End Patch"
        )
        r = repair_apply_patch_dsl(dsl)
        assert "@@new_marker" in _dsl(r)
        assert not any("hunk header" in rep for rep in r.repairs)
class TestRepairResult:
    """RepairResult 封闭 DSL 契约（B1：修复不向 DSL 注入任何内容）。

    Codex harness 强校验首末两行字面匹配（首行 `*** Begin Patch`、
    末行 `*** End Patch`），所以修复后的 dsl 字符串不能注入任何注记；
    修复反馈改走 logger 留痕，repairs 字段保留供日志使用。
    """

    def test_no_repair_dsl_unchanged(self):
        """无修复时 .dsl 等于原文本。"""
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert not r.was_repaired
        assert r.repairs == []
        assert r.dsl == dsl

    def test_repaired_dsl_has_no_injected_lines(self):
        """修复后 dsl 不含 # llm-proxy auto-repaired 注记行。"""
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x"
        r = repair_apply_patch_dsl(dsl)
        assert r.was_repaired
        assert "# llm-proxy" not in r.dsl

    def test_repaired_dsl_starts_with_begin_patch(self):
        """修复后 dsl 首行必须是 *** Begin Patch（Codex harness 首行检查）。"""
        dsl = "*** Add File: /a.py\n+x\n*** End Patch"  # 缺 Begin
        r = repair_apply_patch_dsl(dsl)
        assert r.was_repaired
        assert r.dsl.startswith("*** Begin Patch"), r.dsl

    def test_repaired_dsl_ends_with_end_patch(self):
        """修复后 dsl 末行必须是 *** End Patch（Codex harness 末行检查）。"""
        dsl = "*** Begin Patch\n*** Add File: /a.py\n+x"  # 缺 End
        r = repair_apply_patch_dsl(dsl)
        assert r.was_repaired
        assert r.dsl.rstrip().endswith("*** End Patch"), r.dsl

    def test_multi_repair_dsl_keeps_outer_envelope(self):
        """多项修复时 dsl 仍保持首末两行字面匹配 + 内部不注 # 注记。"""
        dsl = "** * Begin Patch\n*** Add File: /a.py\n+x\n*** END PATCH"
        r = repair_apply_patch_dsl(dsl)
        assert r.was_repaired
        assert "normalized Begin marker" in r.repairs
        assert "normalized End marker" in r.repairs
        assert r.dsl.startswith("*** Begin Patch")
        assert r.dsl.rstrip().endswith("*** End Patch")
        # 内部不含 # 注记
        for line in r.dsl.split("\n"):
            assert not line.startswith("# llm-proxy"), f"unexpected note in DSL: {line!r}"

    def test_repairs_field_still_populated_for_logger(self):
        """repairs 字段保留供 logger 留痕用，不依赖 DSL 注记通道。"""
        dsl = "*** Begin Patch\n*** Update File: foo.py\n@@ -1,1 +1,1 @@\n-a\n+b\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert r.was_repaired
        assert "normalized @@ hunk header" in r.repairs
class TestFileHeaderCasingNormalization:
    """文件操作关键字大小写归一化：模型常见 update File / Update file 等错误。"""

    def test_update_file_lowercase_u_normalized(self):
        """update File: → Update File:"""
        dsl = "*** Begin Patch\n*** update File: /tmp/a.py\n@@\n-old\n+new\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** Update File: /tmp/a.py" in _dsl(r)
        assert "*** update File" not in _dsl(r)
        assert any("Update File" in rep and "casing" in rep for rep in r.repairs)

    def test_update_file_lowercase_f_normalized(self):
        """Update file: → Update File:"""
        dsl = "*** Begin Patch\n*** Update file: /tmp/a.py\n@@\n-old\n+new\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** Update File: /tmp/a.py" in _dsl(r)
        assert "*** Update file" not in _dsl(r)
        assert any("Update File" in rep and "casing" in rep for rep in r.repairs)

    def test_add_file_uppercase_normalized(self):
        """ADD FILE: → Add File:"""
        dsl = "*** Begin Patch\n*** ADD FILE: /tmp/a.py\n+x\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** Add File: /tmp/a.py" in _dsl(r)
        assert "*** ADD FILE" not in _dsl(r)
        assert any("Add File" in rep and "casing" in rep for rep in r.repairs)

    def test_delete_file_mixed_casing_normalized(self):
        """delete FILE: → Delete File:"""
        dsl = "*** Begin Patch\n*** delete FILE: /tmp/a.py\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** Delete File: /tmp/a.py" in _dsl(r)
        assert "*** delete FILE" not in _dsl(r)
        assert any("Delete File" in rep and "casing" in rep for rep in r.repairs)

    def test_move_to_lowercase_normalized(self):
        """move to: → Move to:"""
        dsl = "*** Begin Patch\n*** Update File: /tmp/a.py\n*** move to: /tmp/b.py\n@@\n-old\n+new\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert "*** Move to: /tmp/b.py" in _dsl(r)
        assert "*** move to" not in _dsl(r)
        assert any("Move to" in rep and "casing" in rep for rep in r.repairs)

    def test_correct_casing_no_repair(self):
        """标准大小写不应触发修复。"""
        dsl = "*** Begin Patch\n*** Update File: /tmp/a.py\n@@\n-old\n+new\n*** End Patch"
        r = repair_apply_patch_dsl(dsl)
        assert not any("casing" in rep for rep in r.repairs)

    def test_multiple_wrong_casing_all_normalized(self):
        """多个文件头大小写错误都应被纠正。"""
        dsl = (
            "*** Begin Patch\n"
            "*** update File: /tmp/a.py\n@@\n-old\n+new\n"
            "*** ADD FILE: /tmp/b.py\n+x\n"
            "*** End Patch"
        )
        r = repair_apply_patch_dsl(dsl)
        dsl_body = _dsl(r)
        assert "*** Update File: /tmp/a.py" in dsl_body
        assert "*** Add File: /tmp/b.py" in dsl_body
        assert "*** update File" not in dsl_body
        assert "*** ADD FILE" not in dsl_body
        casing_repairs = [rep for rep in r.repairs if "casing" in rep]
        assert len(casing_repairs) == 2
