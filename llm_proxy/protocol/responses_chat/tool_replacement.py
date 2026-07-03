"""apply_patch DSL ↔ 結構化參數轉換。

正向：解析 apply_patch 的 DSL 文本，提取每個文件段的操作類型和參數。
反向：將結構化參數（含 action 字段）生成 apply_patch 格式的 DSL 文本。

上游模型看到單個 apply_patch 工具，參數包含 action 枚舉區分操作類型：
  add_file / update_file / delete_file / append_to_file
Codex 看到 type=custom, name=apply_patch，input 為 DSL 文本。

apply_patch DSL 完整語法：
  patch: "*** Begin Patch" file_section+ "*** End Patch"
  file_section: add_hunk | delete_hunk | update_hunk
  add_hunk: "*** Add File: " filename LF add_line+
  add_line: "+" /(.*)/ LF
  delete_hunk: "*** Delete File: " filename LF
  update_hunk: "*** Update File: " filename LF change_move? change
  change_move: "*** Move to: " filename LF
  change: (change_context | change_line)+ eof_line?
  change_context: ("@@" | "@@ " /(.+)/) LF
  change_line: ("+" | "-" | " ") /(.*)/ LF
  eof_line: "*** End of File" LF
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# apply_patch 單工具定義（發給上游模型）
# 用 action 枚舉代替 4 個獨立工具，參數按操作類型組合
# ---------------------------------------------------------------------------

APPLY_PATCH_TOOL_DESCRIPTION: str = (
    "Apply file operations (add_file, update_file, delete_file, append_to_file) "
    "using structured parameters. Single action or use action='batch' "
    "with operations[] for multiple changes in one patch."
)

ACTION_TO_TOOL: dict[str, str] = {
    "add_file": "write_to_file",
    "update_file": "replace_in_file",
    "delete_file": "delete_file",
    "append_to_file": "append_to_file",
}

TOOL_TO_ACTION: dict[str, str] = {v: k for k, v in ACTION_TO_TOOL.items()}

# ---------------------------------------------------------------------------
# apply_patch 單工具定義（發給上游模型）
# 用 action 枚舉代替 4 個獨立工具，參數按操作類型組合
# ---------------------------------------------------------------------------

APPLY_PATCH_SINGLE_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": APPLY_PATCH_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "add_file", "update_file", "delete_file",
                        "append_to_file", "batch",
                    ],
                    "description": (
                        "Type of file operation to perform. "
                        "Use 'batch' to apply multiple operations in a single patch."
                    ),
                },
                "filePath": {
                    "type": "string",
                    "description": "Absolute path of the target file. Required for non-batch actions.",
                },
                "content": {
                    "type": "string",
                    "description": "File content. Required for add_file and append_to_file.",
                },
                "old_str": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "The exact text to be replaced. Must not be empty. "
                        "Include surrounding context lines to disambiguate. "
                        "Required for update_file."
                    ),
                },
                "new_str": {
                    "type": "string",
                    "description": (
                        "The replacement text. Include the same surrounding "
                        "context lines that were in old_str. Required for update_file."
                    ),
                },
                "destinationPath": {
                    "type": "string",
                    "description": (
                        "Optional new absolute path for the file. "
                        "When provided, the file is moved/renamed to this path after the update. "
                        "Only used with update_file action."
                    ),
                },
                "operations": {
                    "type": "array",
                    "description": (
                        "Array of operations for batch mode. "
                        "Required when action='batch'. "
                        "Each entry must contain 'action', 'filePath', "
                        "and action-specific parameters."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "add_file", "update_file",
                                    "delete_file", "append_to_file",
                                ],
                                "description": "Type of file operation for this batch entry.",
                            },
                            "filePath": {
                                "type": "string",
                                "description": "Absolute path of the target file.",
                            },
                            "content": {
                                "type": "string",
                                "description": "File content for add_file or append_to_file.",
                            },
                            "old_str": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "Text to replace. Required for update_file. "
                                    "Include surrounding context lines to disambiguate."
                                ),
                            },
                            "new_str": {
                                "type": "string",
                                "description": (
                                    "Replacement text. Required for update_file. "
                                    "Include the same surrounding context lines as old_str."
                                ),
                            },
                            "destinationPath": {
                                "type": "string",
                                "description": (
                                    "New absolute path to move/rename the file. "
                                    "Only for update_file."
                                ),
                            },
                        },
                        "required": ["action", "filePath"],
                    },
                },
            },
            "required": ["action"],
        },
    },
}

# ---------------------------------------------------------------------------
# 正向轉換輔助：將 parse_apply_patch_to_simple 結果轉為結構化 action 格式
# ---------------------------------------------------------------------------

def tool_result_to_action_call(op: dict) -> dict:
    """將 parse_apply_patch_to_simple 的單個結果轉為 {"action": ..., "filePath": ..., ...}。

    op 格式：{"tool": "write_to_file", "args": {"filePath": ..., "content": ...}}
    返回格式：{"action": "add_file", "filePath": ..., "content": ...}
    """
    tool = op.get("tool", "")
    action = TOOL_TO_ACTION.get(tool, tool)
    return {"action": action, **op.get("args", {})}


# ---------------------------------------------------------------------------
# 參數提取輔助
# ---------------------------------------------------------------------------

_PATH_KEYS = ("filePath", "file_path", "path", "filepath")
_CONTENT_KEYS = ("content", "contents", "text")
_OLD_STR_KEYS = ("old_str", "oldStr", "old_string", "old")
_NEW_STR_KEYS = ("new_str", "newStr", "new_string", "new")
_DEST_PATH_KEYS = ("destinationPath", "destination_path", "dest_path", "destPath", "new_path")


def _extract_arg(args: dict, keys: tuple[str, ...]) -> str | None:
    """從參數字典中按優先級取第一個存在的 key 值。"""
    for k in keys:
        v = args.get(k)
        if v is not None and isinstance(v, str):
            return v
    return None


# ---------------------------------------------------------------------------
# 正向轉換：apply_patch DSL → 結構化操作列表
# ---------------------------------------------------------------------------

_BEGIN_RE = re.compile(r"\*{3}\s*Begin\s+Patch", re.IGNORECASE)
_ADD_FILE_RE = re.compile(r"\*{3}\s*Add\s+File\s*:\s*(.+)", re.IGNORECASE)
_UPDATE_FILE_RE = re.compile(r"\*{3}\s*Update\s+File\s*:\s*(.+)", re.IGNORECASE)
_DELETE_FILE_RE = re.compile(r"\*{3}\s*Delete\s+File\s*:\s*(.+)", re.IGNORECASE)
_MOVE_TO_RE = re.compile(r"\*{3}\s*Move\s+to\s*:\s*(.+)", re.IGNORECASE)
_END_OF_FILE_RE = re.compile(r"\*{3}\s*End\s+of\s+File", re.IGNORECASE)
_END_RE = re.compile(r"\*{3}\s*End\s+Patch", re.IGNORECASE)


def parse_apply_patch_to_simple(input_text: str) -> list[dict] | None:
    """嘗試將 apply_patch 的 patch 文本解析為標準工具調用參數列表。

    返回列表中每個元素格式：
        {"tool": "write_to_file", "args": {"filePath": ..., "content": ...}}
        {"tool": "replace_in_file", "args": {"filePath": ..., "old_str": ..., "new_str": ..., destinationPath?}}
        {"tool": "delete_file", "args": {"filePath": ...}}
        {"tool": "_degraded_user_message", "args": {"content": "..."}}

    返回 None — 解析失敗（格式錯誤或無法定位文件頭）
    """
    if not input_text or not isinstance(input_text, str):
        return None

    # 提取 Begin Patch 和 End Patch 之間的內容
    # 兼容無 Begin Patch 前綴的輸入（Codex 有時不帶前綴）
    begin_match = _BEGIN_RE.search(input_text)
    if begin_match:
        body_start = begin_match.end()
    else:
        # 無 Begin Patch：嘗試從第一個文件頭開始
        first_header = None
        for pat in (_ADD_FILE_RE, _UPDATE_FILE_RE, _DELETE_FILE_RE):
            m = pat.search(input_text)
            if m and (first_header is None or m.start() < first_header.start()):
                first_header = m
        if first_header is None:
            return None
        body_start = first_header.start()

    end_match = _END_RE.search(input_text, body_start)
    if end_match:
        body_text = input_text[body_start:end_match.start()]
    else:
        body_text = input_text[body_start:]

    # 按文件頭分割為段
    segments: list[tuple[str, str, list[str]]] = []  # [(action, path, lines)]
    current_action: str | None = None
    current_path: str | None = None
    current_lines: list[str] = []

    for line in body_text.splitlines():
        add_match = _ADD_FILE_RE.match(line)
        update_match = _UPDATE_FILE_RE.match(line)
        delete_match = _DELETE_FILE_RE.match(line)

        if add_match:
            if current_action is not None:
                segments.append((current_action, current_path, current_lines))
            current_action = "add"
            current_path = add_match.group(1).strip()
            current_lines = []
        elif update_match:
            if current_action is not None:
                segments.append((current_action, current_path, current_lines))
            current_action = "update"
            current_path = update_match.group(1).strip()
            current_lines = []
        elif delete_match:
            if current_action is not None:
                segments.append((current_action, current_path, current_lines))
            current_action = "delete"
            current_path = delete_match.group(1).strip()
            current_lines = []
        elif _END_OF_FILE_RE.match(line):
            # End of File 標記段結束
            if current_action is not None:
                segments.append((current_action, current_path, current_lines))
            current_action = None
            current_path = None
            current_lines = []
        else:
            # diff 內容行：@@ / +/-/空格前綴 / *** Move to:
            stripped = line.strip()
            move_match = _MOVE_TO_RE.match(line)
            if move_match or stripped.startswith("@@") or line.startswith("+") or line.startswith("-") or line.startswith(" "):
                current_lines.append(line)

    # 最後一段
    if current_action is not None:
        segments.append((current_action, current_path, current_lines))

    if not segments:
        return None

    # 逐段解析
    results: list[dict] = []
    for action, path, lines in segments:
        parsed = _parse_segment(action, path, lines)
        if parsed is None:
            # 段級降級：不影響其他段
            logger.warning("Segment parse failed: action=%s path=%s, degrading segment", action, path)
            path_hint = path[:80] if path else "unknown"
            results.append({
                "tool": "_degraded_user_message",
                "args": {
                    "content": f"[File {path_hint} was modified — patch segment could not be converted]",
                },
            })
        else:
            results.append(parsed)

    return results


def _parse_segment(action: str, path: str, lines: list[str]) -> dict | None:
    """解析單個文件段為工具調用參數。"""

    # 所有工具都要求 filePath 非空
    if not path or not path.strip():
        logger.warning("Segment has empty filePath: action=%s", action)
        return None

    if action == "add":
        plus_lines = [ln[1:] for ln in lines if ln.startswith("+")]
        content = "\n".join(plus_lines)
        # write_to_file 要求 content 非空（空文件無意義，且可能與 replace 語義混淆）
        if not content and not plus_lines:
            logger.warning("Add File has no '+' lines (empty content): %s", path)
            return None
        return {
            "tool": "write_to_file",
            "args": {"filePath": path, "content": content},
        }

    if action == "delete":
        return {
            "tool": "delete_file",
            "args": {"filePath": path},
        }

    if action == "update":
        # 提取 Move to 目標路徑
        dest_path: str | None = None
        content_lines: list[str] = []
        for ln in lines:
            move_match = _MOVE_TO_RE.match(ln)
            if move_match:
                dest_path = move_match.group(1).strip()
                continue
            content_lines.append(ln)

        minus_lines = [ln[1:] for ln in content_lines if ln.startswith("-")]
        plus_lines = [ln[1:] for ln in content_lines if ln.startswith("+")]
        context_lines = [ln[1:] for ln in content_lines if ln.startswith(" ")]

        # 無 diff 行 + 無 Move to → 空段
        if not minus_lines and not plus_lines and dest_path is None:
            logger.warning("Update File segment has no '-'/'+' lines and no Move to: %s", path)
            return None

        # Move to only（無內容修改）
        if not minus_lines and not plus_lines and dest_path is not None:
            # 安全：佔位 old_str/new_str 為單個空格（minLength=1）
            return {
                "tool": "replace_in_file",
                "args": {
                    "filePath": path,
                    "old_str": " ",
                    "new_str": " ",
                    "destinationPath": dest_path,
                },
            }

        # 有 diff 行但無 '-' 行 → 只有 '+' 行的 Update File
        old_str = "\n".join(minus_lines)
        if not old_str:
            # 有 context 行 → 用 context 行作為錨點構造 replace_in_file
            # 這對齊 Claude Code 的行為：用已有內容作為 old_str，追加新內容
            if context_lines:
                old_str = "\n".join(context_lines)
                new_str = "\n".join(context_lines + plus_lines)
                args = {
                    "filePath": path,
                    "old_str": old_str,
                    "new_str": new_str,
                }
                if dest_path is not None:
                    args["destinationPath"] = dest_path
                return {"tool": "replace_in_file", "args": args}

            # 無 context 行也無 '-' 行 → 只有 '+' 行的追加操作
            if plus_lines:
                # 有 Move to → 必須用 replace_in_file（append_to_file 不支持 destinationPath）
                if dest_path is not None:
                    logger.debug("Update File has only '+' lines with Move to, degrading to user message: %s", path)
                    return {
                        "tool": "_degraded_user_message",
                        "args": {
                            "content": f"[File {path} was modified (rename to {dest_path}, append) — appended content]\n" + "\n".join(plus_lines),
                        },
                    }
                logger.debug("Update File has only '+' lines and no context (append), converting to append_to_file: %s", path)
                return {
                    "tool": "append_to_file",
                    "args": {
                        "filePath": path,
                        "content": "\n".join(plus_lines),
                    },
                }
            # 無 '-' 也無 '+' 行（只有 @@ 行）→ 無效段
            logger.warning("Update File has no '-' or '+' lines: %s", path)
            return None

        args = {
            "filePath": path,
            "old_str": old_str,
            "new_str": "\n".join(plus_lines),
        }
        if dest_path is not None:
            args["destinationPath"] = dest_path
        return {"tool": "replace_in_file", "args": args}

    return None


# ---------------------------------------------------------------------------
# 反向轉換：結構化參數 → apply_patch DSL
# ---------------------------------------------------------------------------

def write_to_apply_patch(file_path: str, content: str) -> str:
    """將 Add File 操作轉為 apply_patch Add File 格式。"""
    lines = content.split("\n")
    body = "\n".join(f"+{ln}" for ln in lines)
    return "*** Begin Patch\n*** Add File: {}\n{}\n*** End Patch".format(file_path, body)


def delete_to_apply_patch(file_path: str) -> str:
    """將 Delete File 操作轉為 apply_patch Delete File 格式。"""
    return "*** Begin Patch\n*** Delete File: {}\n*** End Patch".format(file_path)


def append_to_apply_patch(file_path: str, content: str) -> str:
    """將 Append File 操作轉為 apply_patch Update File 格式（只有 '+' 行）。

    Codex 的 apply_patch 執行器支持只有 '+' 行的 Update File，等同於追加到文件末尾。
    """
    lines = content.split("\n")
    body = "\n".join(f"+{ln}" for ln in lines)
    return "*** Begin Patch\n*** Update File: {}\n@@\n{}\n*** End Patch".format(file_path, body)


def replace_to_apply_patch(file_path: str, old_str: str, new_str: str, dest_path: str | None = None) -> str:
    """將 Update File 操作轉為 apply_patch Update File 格式。

    對 old_str / new_str 做逐行 diff：提取公共前綴/後綴作為 context 行（空格前綴），
    差異部分作為 -/+ 行。模型如果在 old_str / new_str 中都帶 context，
    patch 對 Codex 的 apply_patch 解析器就是有效的。

    當 dest_path 非空時，在 Update File 行後插入 *** Move to: 行。
    """
    old_lines = old_str.split("\n")
    new_lines = new_str.split("\n")

    # 公共前綴：從頭匹配到第一個差異行
    prefix_len = 0
    min_len = min(len(old_lines), len(new_lines))
    while prefix_len < min_len and old_lines[prefix_len] == new_lines[prefix_len]:
        prefix_len += 1

    # 公共後綴：從尾向前匹配（不重疊前綴部分）
    suffix_len = 0
    while (
        suffix_len < len(old_lines) - prefix_len
        and suffix_len < len(new_lines) - prefix_len
        and old_lines[-(suffix_len + 1)] == new_lines[-(suffix_len + 1)]
    ):
        suffix_len += 1

    context_prefix = old_lines[:prefix_len]
    removed = old_lines[prefix_len : len(old_lines) - suffix_len]
    added = new_lines[prefix_len : len(new_lines) - suffix_len]
    context_suffix = old_lines[len(old_lines) - suffix_len :]

    parts = ["*** Begin Patch", "*** Update File: {}".format(file_path)]
    if dest_path:
        parts.append("*** Move to: {}".format(dest_path))
    parts.append("@@")
    for ln in context_prefix:
        parts.append(" {}".format(ln))
    for ln in removed:
        parts.append("-{}".format(ln))
    for ln in added:
        parts.append("+{}".format(ln))
    for ln in context_suffix:
        parts.append(" {}".format(ln))
    parts.append("*** End Patch")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 反向轉換異常
# ---------------------------------------------------------------------------

class ReverseConversionError(ValueError):
    """反向轉換失敗，攜帶錯誤信息供上層構造錯誤 tool result。"""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        msg = f"reverse conversion failed: {reason}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


def _single_op_to_apply_patch(args: dict) -> str:
    """Convert a single structured operation to apply_patch DSL (full patch)."""
    action = args.get("action", "")

    if action == "add_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        content = _extract_arg(args, _CONTENT_KEYS)
        if file_path is None or content is None:
            raise ReverseConversionError(
                "filePath and content are required for add_file",
                "Provide both filePath and content parameters.",
            )
        if not content:
            raise ReverseConversionError(
                "content must not be empty for add_file",
                "Provide actual file content, or use append_to_file to build the file incrementally.",
            )
        return write_to_apply_patch(file_path, content)

    if action == "update_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        old_str = _extract_arg(args, _OLD_STR_KEYS)
        new_str = _extract_arg(args, _NEW_STR_KEYS)
        dest_path = _extract_arg(args, _DEST_PATH_KEYS)
        if file_path is None or old_str is None or new_str is None:
            raise ReverseConversionError(
                "filePath, old_str, and new_str are required for update_file",
                "update_file requires filePath, old_str, and new_str parameters.",
            )
        if not old_str:
            raise ReverseConversionError(
                "old_str must not be empty for update_file",
                "When appending to a file, use append_to_file action instead. "
                "For update_file, include the last few lines of the file as old_str "
                "and add new content in new_str.",
            )
        return replace_to_apply_patch(file_path, old_str, new_str, dest_path=dest_path)

    if action == "delete_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        if file_path is None:
            raise ReverseConversionError(
                "filePath is required for delete_file",
                "delete_file requires a filePath parameter.",
            )
        return delete_to_apply_patch(file_path)

    if action == "append_to_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        content = _extract_arg(args, _CONTENT_KEYS)
        if file_path is None or content is None:
            raise ReverseConversionError(
                "filePath and content are required for append_to_file",
                "append_to_file requires both filePath and content parameters.",
            )
        return append_to_apply_patch(file_path, content)

    raise ReverseConversionError(
        f"unknown action: {action!r}",
        "Expected one of: add_file, update_file, append_to_file, delete_file, batch.",
    )


def reverse_tool_args_to_apply_patch(args: dict) -> str:
    """將上游 apply_patch 工具調用的結構化 arguments 轉為 apply_patch DSL 文本。

    支持兩種模式：
    1. 單操作（向後兼容）：{"action": "add_file", "filePath": "...", "content": "..."}
    2. 批量操作：{"action": "batch", "operations": [{"action": ..., ...}, ...]}

    Returns:
        apply_patch 格式的 DSL 字符串（可能含多個文件段）

    Raises:
        ReverseConversionError: 參數驗證失敗，攜帶原因和修正建議
    """
    if not isinstance(args, dict):
        raise ReverseConversionError(
            "args must be a dict",
            f"Got {type(args).__name__} instead.",
        )

    action = args.get("action", "")

    if action == "batch":
        operations = args.get("operations", [])
        if not operations or not isinstance(operations, list):
            raise ReverseConversionError(
                "operations must be a non-empty list for batch action",
                "Provide operations array with at least one entry, "
                "each containing action, filePath, and action-specific parameters.",
            )
        segments: list[str] = []
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                raise ReverseConversionError(
                    f"operation[{i}] must be a dict",
                    f"Got {type(op).__name__} instead.",
                )
            if "action" not in op:
                raise ReverseConversionError(
                    f"operation[{i}] missing required 'action' field",
                    "Each operation in the batch must have an action field.",
                )
            segments.append(_single_op_to_apply_patch(op))
        return _combine_dsl_segments(segments)

    return _single_op_to_apply_patch(args)


def _combine_dsl_segments(segments: list[str]) -> str:
    """Combine multiple full-patch DSL segments into one unified patch.

    Each segment is a self-contained patch with *** Begin Patch / *** End Patch.
    Strips inner markers so the result is a single valid patch.
    """
    if not segments:
        raise ReverseConversionError("no operations", "Provide at least one operation.")
    if len(segments) == 1:
        return segments[0]

    parts: list[str] = []
    for i, seg in enumerate(segments):
        lines = seg.splitlines()
        # Strip leading *** Begin Patch from non-first segments
        if i > 0 and lines and _BEGIN_RE.match(lines[0]):
            lines = lines[1:]
        # Strip trailing *** End Patch from non-last segments
        if i < len(segments) - 1 and lines and _END_RE.match(lines[-1]):
            lines = lines[:-1]
        parts.extend(lines)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# DSL 修復（repair_apply_patch_dsl）
# ═══════════════════════════════════════════════════════════════════════
#
# 當上游模型以自定義工具透傳模式調用 apply_patch 時，返回的 input 參數字串
# 可能不完整（缺少 *** Begin Patch、*** End Patch，或夾雜語音文本）。
# repair_apply_patch_dsl 負責修復這些格式問題，保證 Codex 能正確解析。
#
# 注意：apply_patch 與 git/diff patch 的關鍵差異：
#   - Context 行既是定位錨點也是輸出文本，寫錯會靜默覆蓋文件內容（不只是匹配失敗）。
#   - Hunk 按順序應用。同一函數的相鄰改動必須合併到一個 hunk，拆分會導致後序 hunk 匹配失敗。
#   - Context 建議 3-5 行，過長增加 whitespace 不匹配概率。
# ═══════════════════════════════════════════════════════════════════════

# DSL marker 正則
# 必須行首匹配，避免在文件內容行中間誤匹配（如 +// TODO: *** End Patch）
# 兼容各種不標準寫法：***, ****, ** *, * * *

_REPAIR_BEGIN_RE = re.compile(
    r"^[ \t]*(?:\*{3,4}|\*\*\s\*|\*\s\*\s\*)\s*Begin\s+Patch",
    re.IGNORECASE | re.MULTILINE,
)
_REPAIR_END_RE = re.compile(
    r"^[ \t]*(?:\*{3,4}|\*\*\s\*|\*\s\*\s\*)\s*End\s+Patch",
    re.IGNORECASE | re.MULTILINE,
)
# 文件操作關鍵字大小寫歸一化
# apply_patch 嚴格校驗 Add File / Update File / Delete File / Move to 的大小寫，
# 模型常見錯誤：update File、Update file、ADD FILE 等。需糾正為標準 Title Case。
_HEADER_NORMALIZE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\*{3}\s*Add\s+File\s*:\s*(.+)", re.IGNORECASE), r"*** Add File: \1"),
    (re.compile(r"\*{3}\s*Update\s+File\s*:\s*(.+)", re.IGNORECASE), r"*** Update File: \1"),
    (re.compile(r"\*{3}\s*Delete\s+File\s*:\s*(.+)", re.IGNORECASE), r"*** Delete File: \1"),
    (re.compile(r"\*{3}\s*Move\s+to\s*:\s*(.+)", re.IGNORECASE), r"*** Move to: \1"),
]
# @@ hunk header 歸一化
# 模型有時把 @@ 當成 unified-diff 的 hunk header（@@ -19,5 +19,6 @@），
# 或當成 anchor（@@ def some_function:）。apply_patch 期望 @@ 單獨成行。
# 匹配以 @@ 起始行、後跟空白 + 至少一個非空白字符的行，截斷為裸 @@。
_HUNK_HEADER_REPAIR_RE = re.compile(r"^@@[ \t]+\S[^\n]*$", re.MULTILINE)

# 歸一化 regex：捕獲前導空白 + 非標準前綴 + Begin/End Patch
# [\s*#] 允許字符之間有空格，所以能匹配 ** * / * * *
_NORMALIZE_BEGIN = re.compile(
    r"^([ \t]*)(?:[\*#][\s*#]*){2,}[ \t_]*begin[ \t_]*(?:of[ \t_]*)?patch[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NORMALIZE_END = re.compile(
    r"^([ \t]*)(?:[\*#][\s*#]*){2,}[ \t_]*end[ \t_]*(?:of[ \t_]*)?patch[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class RepairResult:
    """DSL 修復結果。

    Attributes:
        dsl: 修復後的完整 DSL。Codex harness 強校驗首末兩行（首行必須為
             ``*** Begin Patch``、末行必須為 ``*** End Patch``），所以
             修復結果保持 DSL 字符串本身的封閉性，不注入任何註記。
        repairs: 修復項描述列表（人類可讀短句）。空列表表示無修復。
    """
    dsl: str
    repairs: list[str] = field(default_factory=list)

    @property
    def was_repaired(self) -> bool:
        return bool(self.repairs)


def repair_apply_patch_dsl(dsl: str) -> RepairResult:
    """修復 apply_patch DSL 格式問題，返回 RepairResult。

    上游模型有時輸出不完整的 DSL：缺少 *** Begin Patch、
    缺少 *** End Patch，或在補丁前後夾雜語音文本。

    修復策略（按優先級）：
    - 兩者都有 → 截斷到 Begin..End 之間的純淨 DSL
    - 有 Begin 無 End → 補 End
    - 無 Begin 有 End → 在 End 之前的第一個文件頭前插入 Begin
    - 無 Begin 無 End → 在第一個文件頭前插入 Begin，末尾補 End
    - 無文件頭 → 原樣返回
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

    # Step 1: 歸一化 marker（** * Begin → *** Begin 等）
    before = text
    new_text = _NORMALIZE_BEGIN.sub(r"\1*** Begin Patch", text)
    new_text = _NORMALIZE_END.sub(r"\1*** End Patch", new_text)
    if new_text != before:
        # 區分 Begin / End 歸一化
        end_only = _NORMALIZE_END.sub(r"\1*** End Patch", text)
        begin_only = _NORMALIZE_BEGIN.sub(r"\1*** Begin Patch", text)
        if begin_only != text:
            repairs.append("normalized Begin marker")
        if end_only != text:
            repairs.append("normalized End marker")
        text = new_text

    # Step 2: 歸一化 @@ hunk header（@@ -19,5 +19,6 @@ → @@）
    text, hh_count = _normalize_hunk_headers(text)
    if hh_count:
        if hh_count == 1:
            repairs.append("normalized @@ hunk header")
        else:
            repairs.append(f"normalized {hh_count} @@ hunk headers")

    # Step 3: 歸一化文件操作關鍵字大小寫（update File → Update File 等）
    header_repairs = _normalize_file_headers(text)
    if header_repairs:
        text = header_repairs["text"]
        repairs.extend(header_repairs["repairs"])

    # Step 4: 找 Begin/End 位置（基於歸一化後的文本）
    begin_matches = list(_REPAIR_BEGIN_RE.finditer(text))
    end_matches = list(_REPAIR_END_RE.finditer(text))
    begin_match = begin_matches[0] if begin_matches else None
    end_match = end_matches[0] if end_matches else None

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
            # End 在行首/文件頭之前：降級到無 Begin 無 End 路徑
            first_header = _find_first_header(text)
            if first_header:
                text = "*** Begin Patch\n" + text[first_header.start():]
                text = text.rstrip() + "\n*** End Patch"
                repairs.append("wrapped with Begin/End markers")
            else:
                return RepairResult(dsl=dsl, repairs=[])  # 無法定位插入位置
    elif not begin_match and not end_match:
        first_header = _find_first_header(text)
        if first_header:
            text = "*** Begin Patch\n" + text[first_header.start():]
            text = text.rstrip() + "\n*** End Patch"
            repairs.append("wrapped with Begin/End markers")
        else:
            return RepairResult(dsl=dsl, repairs=[])  # 無文件頭，無法修復

    if repairs:
        _log_repair_diff(original, text, repairs)
        logger.info("repair_apply_patch_dsl: applied %d repair(s): %s",
                    len(repairs), "; ".join(repairs))

    return RepairResult(dsl=text, repairs=repairs)


def _normalize_file_headers(text: str) -> dict | None:
    """糾正文件操作關鍵字的大小寫。

    apply_patch 嚴格校驗 ``*** Add File:`` / ``*** Update File:`` /
    ``*** Delete File:`` / ``*** Move to:`` 的大小寫。
    模型常見錯誤：update File、Update file、ADD FILE 等。

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
            kw = replacement.split(":")[0].replace("*** ", "")
            repairs.append(f"normalized *** {kw}: casing")

    if not repairs:
        return None
    return {"text": new_text, "repairs": repairs}


def _log_repair_diff(original: str, repaired: str, repairs: list[str]) -> None:
    """WARNING 級別打 before/after，用於真機排查 `***` 丟失等。"""

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
    """掃描範圍內的第一個文件頭位置。"""
    first = None
    end = end if end >= 0 else len(text)
    for pat in (_ADD_FILE_RE, _UPDATE_FILE_RE, _DELETE_FILE_RE):
        m = pat.search(text, start, end)
        if m and (first is None or m.start() < first.start()):
            first = m
    return first


def _normalize_hunk_headers(text: str) -> tuple[str, int]:
    """截斷 @@ hunk header 行的尾部內容。

    apply_patch 期望 @@ 單獨成行（可選前導/尾隨空白）。
    模型誤把 @@ 當 unified-diff 頭時常見三種：
      - "@@ -19,5 +19,6 @@"（unified diff 元數據）
      - "@@ def some_function:"（function anchor）
      - "@@ # comment"（任意文本）

    Returns:
        (new_text, count) — 歸一化後的文本和被歸一化的行數。
    """
    count = 0

    def _strip(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "@@"

    new_text = _HUNK_HEADER_REPAIR_RE.sub(_strip, text)
    return new_text, count
