"""apply_patch DSL 修复工具。

当上游模型以自定义工具透传模式调用 apply_patch 时，返回的 input 参数字符串
可能不完整（缺少 *** Begin Patch、*** End Patch，或夹杂语音文本）。
repair_apply_patch_dsl 负责修复这些格式问题，保证 Codex 能正确解析。

修复后会返回 RepairResult，stream.py 拿到后会用 .dsl 作为 custom_tool_call.input。
如果被修复，.dsl 顶部会嵌入一行紧凑的 # 注释，告知模型：
  - 工具修了什么
  - 让模型下次能写出正确格式
注释行被 apply_patch 解析器自动忽略，不影响 patch 执行。
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

APPLY_PATCH_TOOL_DESCRIPTION: str = (
    "Edit files using the apply_patch DSL. "
    "The single input string parameter must contain a patch with this structure: "
    "Begin Patch, Action File lines, change lines, End Patch. "
    "Actions are 'Add File' (new file, all lines start with plus), "
    "'Update File' (diff with minus for removed, plus for added, space for context; "
    "optional 'at' anchor and 'Move to: path' for rename), "
    "'Delete File' (no body). "
    "A patch can contain multiple file sections. "
    "Common failure: 'is not a valid hunk header' usually means a plus prefix "
    "is missing on a content line - every line inside Add/Update must start with plus, minus, or space."
)

# ---------------------------------------------------------------------------
# DSL marker 正则
# 必须行首匹配，避免在文件内容行中间误匹配（如 +// TODO: *** End Patch）
# 兼容各种不标准写法：***, ****, ** *, * * *
# ---------------------------------------------------------------------------

_BEGIN_RE = re.compile(
    r"^[ \t]*(?:\*{3,4}|\*\*\s\*|\*\s\*\s\*)\s*Begin\s+Patch",
    re.IGNORECASE | re.MULTILINE,
)
_END_RE = re.compile(
    r"^[ \t]*(?:\*{3,4}|\*\*\s\*|\*\s\*\s\*)\s*End\s+Patch",
    re.IGNORECASE | re.MULTILINE,
)
_ADD_FILE_RE = re.compile(r"\*{3}\s*Add\s+File\s*:\s*(.+)", re.IGNORECASE)
_UPDATE_FILE_RE = re.compile(r"\*{3}\s*Update\s+File\s*:\s*(.+)", re.IGNORECASE)
_DELETE_FILE_RE = re.compile(r"\*{3}\s*Delete\s+File\s*:\s*(.+)", re.IGNORECASE)
_MOVE_TO_RE = re.compile(r"\*{3}\s*Move\s+to\s*:\s*(.+)", re.IGNORECASE)
# ---------------------------------------------------------------------------
# 文件操作关键字大小写归一化
# apply_patch 严格校验 Add File / Update File / Delete File / Move to 的大小写，
# 模型常见错误：update File、Update file、ADD FILE 等。需纠正为标准 Title Case。
# ---------------------------------------------------------------------------
_HEADER_NORMALIZE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_ADD_FILE_RE, r"*** Add File: \1"),
    (_UPDATE_FILE_RE, r"*** Update File: \1"),
    (_DELETE_FILE_RE, r"*** Delete File: \1"),
    (_MOVE_TO_RE, r"*** Move to: \1"),
]
# ---------------------------------------------------------------------------
# @@ hunk header 归一化
# 模型有时把 @@ 当成 unified-diff 的 hunk header（@@ -19,5 +19,6 @@），
# 或当成 anchor（@@ def some_function:）。apply_patch 期望 @@ 单独成行。
# 匹配以 @@ 起始行、后跟空白 + 至少一个非空白字符的行，截断为裸 @@。
# ---------------------------------------------------------------------------
_HUNK_HEADER_REPAIR_RE = re.compile(r"^@@[ \t]+\S[^\n]*$", re.MULTILINE)

# 归一化 regex：捕获前导空白 + 非标准前缀 + Begin/End Patch
# [\s*#] 允许字符之间有空格，所以能匹配 ** * / * * *
_NORMALIZE_BEGIN = re.compile(
    r"^([ \t]*)(?:[\*#][\s*#]*){2,}[ \t_]*begin[ \t_]*(?:of[ \t_]*)?patch[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NORMALIZE_END = re.compile(
    r"^([ \t]*)(?:[\*#][\s*#]*){2,}[ \t_]*end[ \t_]*(?:of[ \t_]*)?patch[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# 修复结果
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """DSL 修复结果。

    Attributes:
        dsl: 修复后的完整 DSL。Codex harness 强校验首末两行（首行必须为
             ``*** Begin Patch``、末行必须为 ``*** End Patch``），所以
             修复结果保持 DSL 字符串本身的封闭性，不注入任何注记。
        repairs: 修复项描述列表（人类可读短句）。空列表表示无修复。
    """
    dsl: str
    repairs: list[str] = field(default_factory=list)

    @property
    def was_repaired(self) -> bool:
        return bool(self.repairs)


# 注记注入历史（已废弃，见 ad_hoc 笔记 2026-06-19T11-25）：
#   顶部 prepend → Codex harness 首行检查拒收
#   末尾 append  → Codex harness 末行检查拒收
# 修复反馈改走 logger 留痕，DSL 字符串保持首末两行字面匹配。


# ---------------------------------------------------------------------------
# DSL 修复
# ---------------------------------------------------------------------------


def repair_apply_patch_dsl(dsl: str) -> RepairResult:
    """修复 apply_patch DSL 格式问题，返回 RepairResult。

    上游模型有时输出不完整的 DSL：缺少 *** Begin Patch、
    缺少 *** End Patch，或在补丁前后夹杂语音文本。

    修复策略（按优先级）：
    - 两者都有 → 截断到 Begin..End 之间的纯净 DSL
    - 有 Begin 无 End → 补 End
    - 无 Begin 有 End → 在 End 之前的第一个文件头前插入 Begin
    - 无 Begin 无 End → 在第一个文件头前插入 Begin，末尾补 End
    - 无文件头 → 原样返回
    """
    if not dsl or not isinstance(dsl, str):
        return RepairResult(dsl=dsl or "", repairs=[])

    original = dsl
    text = dsl.strip()
    if not text:
        return RepairResult(dsl="", repairs=[])

    repairs: list[str] = []
    stripped = len(original) - len(text)
    if stripped:
        repairs.append(f"stripped {stripped} leading/trailing whitespace char(s)")

    # Step 1: 归一化 marker（** * Begin → *** Begin 等）
    before = text
    new_text = _NORMALIZE_BEGIN.sub(r"\1*** Begin Patch", text)
    new_text = _NORMALIZE_END.sub(r"\1*** End Patch", new_text)
    if new_text != before:
        # 区分 Begin / End 归一化
        end_only = _NORMALIZE_END.sub(r"\1*** End Patch", text)
        begin_only = _NORMALIZE_BEGIN.sub(r"\1*** Begin Patch", text)
        if begin_only != text:
            repairs.append("normalized Begin marker")
        if end_only != text:
            repairs.append("normalized End marker")
        text = new_text

    # Step 2: 归一化 @@ hunk header（@@ -19,5 +19,6 @@ → @@）
    text, hh_count = _normalize_hunk_headers(text)
    if hh_count:
        if hh_count == 1:
            repairs.append("normalized @@ hunk header")
        else:
            repairs.append(f"normalized {hh_count} @@ hunk headers")

    # Step 3: 归一化文件操作关键字大小写（update File → Update File 等）
    # 注：Step 2 和 Step 3 链式执行，Step 2 的输出是 Step 3 的输入，
    # 避免覆盖丢失修复结果。
    header_repairs = _normalize_file_headers(text)
    if header_repairs:
        text = header_repairs["text"]
        repairs.extend(header_repairs["repairs"])

    # Step 4: 找 Begin/End 位置（基于归一化后的文本）
    begin_matches = list(_BEGIN_RE.finditer(text))
    end_matches = list(_END_RE.finditer(text))
    begin_match = begin_matches[0] if begin_matches else None
    end_match = end_matches[0] if end_matches else None

    # Step 4: 按情况处理
    if begin_match and end_match and begin_match.start() < end_match.end():
        new_text = text[begin_match.start():end_match.end()]
        if new_text != text:
            repairs.append("trimmed surrounding text")
        text = new_text
    elif begin_match and not end_match:
        text = text.rstrip() + "\n*** End Patch"
        repairs.append("appended missing *** End Patch")
    elif not begin_match and end_match:
        first_header = _find_first_header(text, 0, end_match.start())
        if first_header:
            text = "*** Begin Patch\n" + text[first_header.start():]
            repairs.append("inserted missing *** Begin Patch")
        else:
            # End 在行首/文件头之前：降级到无 Begin 无 End 路径
            first_header = _find_first_header(text)
            if first_header:
                text = "*** Begin Patch\n" + text[first_header.start():]
                text = text.rstrip() + "\n*** End Patch"
                repairs.append("wrapped with Begin/End markers")
            else:
                return RepairResult(dsl=dsl, repairs=[])  # 无法定位插入位置
    elif not begin_match and not end_match:
        first_header = _find_first_header(text)
        if first_header:
            text = "*** Begin Patch\n" + text[first_header.start():]
            text = text.rstrip() + "\n*** End Patch"
            repairs.append("wrapped with Begin/End markers")
        else:
            return RepairResult(dsl=dsl, repairs=[])  # 无文件头，无法修复

    # Step 5: 组装最终 DSL。Codex harness 强校验首末两行（首行必须为
    #         ``*** Begin Patch``、末行必须为 ``*** End Patch``），所以
    #         修复后不在 DSL 内追加任何内容；修复信息走 logger 留痕。
    #
    # 临时 DEBUG 日志：触发时打 WARNING 级别（含 before/after），
    # 用于排查 `***` 标记丢失等真机问题。确认无问题后可降级回 INFO / 删掉。
    if repairs:
        _log_repair_diff(original, text, repairs)
        logger.info("repair_apply_patch_dsl: applied %d repair(s): %s",
                    len(repairs), "; ".join(repairs))
    final_dsl = text

    return RepairResult(dsl=final_dsl, repairs=repairs)


def _normalize_file_headers(text: str) -> dict | None:
    """纠正文件操作关键字的大小写。

    apply_patch 严格校验 ``*** Add File:`` / ``*** Update File:`` /
    ``*** Delete File:`` / ``*** Move to:`` 的大小写。
    模型常见错误：update File、Update file、ADD FILE 等。

    Returns:
        dict with "text" (str) and "repairs" (list[str]) if any normalization
        was applied, otherwise None.
    """
    repairs: list[str] = []
    new_text = text
    for pat, replacement in _HEADER_NORMALIZE_PATTERNS:
        changed = False
        lines = new_text.split("\n")
        for i, line in enumerate(lines):
            m = pat.match(line)
            if m:
                normalized = pat.sub(replacement, line)
                if normalized != line:
                    lines[i] = normalized
                    changed = True
        if changed:
            new_text = "\n".join(lines)
            # 提取关键字名用于日志：*** Update File: → "Update File"
            kw = replacement.split(":")[0].replace("*** ", "")
            repairs.append(f"normalized *** {kw}: casing")

    if not repairs:
        return None
    return {"text": new_text, "repairs": repairs}


def _log_repair_diff(original: str, repaired: str, repairs: list[str]) -> None:
    """临时 DEBUG：WARNING 级别打 before/after，用于真机排查 `***` 丢失等。

    包含：
    - 原 DSL 全文（截断到 1000 字符）
    - 修后 DSL 全文（截断到 1000 字符）
    - repairs 列表
    - 首末行各 80 字符（快速比对）
    """
    def _trunc(s: str, n: int = 1000) -> str:
        if len(s) <= n:
            return s
        return s[:n] + f"... [truncated, total {len(s)} chars]"

    first_orig = original.splitlines()[0][:80] if original else "(empty)"
    last_orig = original.rstrip().splitlines()[-1][:80] if original.strip() else "(empty)"
    first_repaired = repaired.splitlines()[0][:80] if repaired else "(empty)"
    last_repaired = repaired.rstrip().splitlines()[-1][:80] if repaired.strip() else "(empty)"

    logger.warning(
        "apply_patch repair fired: repairs=%s | "
        "original[first_line=%r, last_line=%r, total_len=%d] | "
        "repaired[first_line=%r, last_line=%r, total_len=%d] | "
        "original_full=%r | repaired_full=%r",
        repairs,
        first_orig, last_orig, len(original),
        first_repaired, last_repaired, len(repaired),
        _trunc(original),
        _trunc(repaired),
    )


def _find_first_header(text: str, start: int = 0, end: int = -1) -> re.Match | None:
    """扫描范围内的第一个文件头位置。"""
    first = None
    end = end if end >= 0 else len(text)
    for pat in (_ADD_FILE_RE, _UPDATE_FILE_RE, _DELETE_FILE_RE):
        m = pat.search(text, start, end)
        if m and (first is None or m.start() < first.start()):
            first = m
    return first


def _normalize_hunk_headers(text: str) -> tuple[str, int]:
    """截断 @@ hunk header 行的尾部内容。

    apply_patch 期望 @@ 单独成行（可选前导/尾随空白）。
    模型误把 @@ 当 unified-diff 头时常见三种：
      - "@@ -19,5 +19,6 @@"（unified diff 元数据）
      - "@@ def some_function:"（function anchor）
      - "@@ # comment"（任意文本）

    Returns:
        (new_text, count) — 归一化后的文本和被归一化的行数。
    """
    count = 0

    def _strip(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "@@"

    new_text = _HUNK_HEADER_REPAIR_RE.sub(_strip, text)
    return new_text, count
