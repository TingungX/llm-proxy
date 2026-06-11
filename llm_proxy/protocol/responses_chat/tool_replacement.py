"""apply_patch DSL 与标准文件工具（write_to_file / replace_in_file / delete_file）之间的转换。

正向：解析 apply_patch 的 patch 文本，提取每个文件段的路径和内容/替换信息。
反向：将标准工具调用参数生成 apply_patch 格式的 patch 文本。

apply_patch DSL 完整语法：
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 标准文件工具定义（发给上游模型）
# ---------------------------------------------------------------------------

WRITE_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "write_to_file",
        "description": (
            "Creates a new file with the specified content. "
            "If the file already exists, it will be completely overwritten. "
            "For modifying existing files, prefer replace_in_file instead. "
            "For long content, prefer append_to_file to add content in smaller chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Absolute path of the target file.",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file.",
                },
            },
            "required": ["filePath", "content"],
        },
    },
}

REPLACE_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "replace_in_file",
        "description": (
            "Performs exact string replacements in an existing file. "
            "The old_str must match the file content exactly and must be unique within the file. "
            "Include enough surrounding context lines to ensure uniqueness. "
            "NEVER use old_str that could match multiple locations. "
            "NEVER use empty old_str. When appending to a file, include the last few lines "
            "of the file as old_str and add new content in new_str. "
            "To rename or move the file, provide destinationPath with the new path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Absolute path of the target file.",
                },
                "old_str": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "The exact text to be replaced. Must not be empty. "
                        "Include surrounding context lines to disambiguate."
                    ),
                },
                "new_str": {
                    "type": "string",
                    "description": (
                        "The replacement text. Include the same surrounding "
                        "context lines that were in old_str."
                    ),
                },
                "destinationPath": {
                    "type": "string",
                    "description": (
                        "Optional new absolute path for the file. "
                        "When provided, the file is moved/renamed to this path after the replacement. "
                        "Use this when the operation includes a rename or move."
                    ),
                },
            },
            "required": ["filePath", "old_str", "new_str"],
        },
    },
}

DELETE_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": (
            "Deletes an existing file permanently. "
            "This is a destructive operation. Only use when the user explicitly requests deletion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Absolute path of the file to delete.",
                },
            },
            "required": ["filePath"],
        },
    },
}

APPEND_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "append_to_file",
        "description": (
            "Appends content to the end of an existing file. "
            "Use this when adding new content to the end of a file without modifying existing content. "
            "Do NOT use replace_in_file with empty old_str for appending — use append_to_file instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Absolute path of the target file.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to append to the end of the file.",
                },
            },
            "required": ["filePath", "content"],
        },
    },
}

# Tool name aliases for lenient argument extraction
_PATH_KEYS = ("filePath", "file_path", "path", "filepath")
_CONTENT_KEYS = ("content", "contents", "text")
_OLD_STR_KEYS = ("old_str", "oldStr", "old_string", "old")
_NEW_STR_KEYS = ("new_str", "newStr", "new_string", "new")
_DEST_PATH_KEYS = ("destinationPath", "destination_path", "dest_path", "destPath", "new_path")


def _extract_arg(args: dict, keys: tuple[str, ...]) -> str | None:
    """从参数字典中按优先级取第一个存在的 key 值。"""
    for k in keys:
        v = args.get(k)
        if v is not None and isinstance(v, str):
            return v
    return None


# ---------------------------------------------------------------------------
# 正向转换：apply_patch DSL → 标准工具调用参数
# ---------------------------------------------------------------------------

_BEGIN_RE = re.compile(r"\*{3}\s*Begin\s+Patch", re.IGNORECASE)
_ADD_FILE_RE = re.compile(r"\*{3}\s*Add\s+File\s*:\s*(.+)", re.IGNORECASE)
_UPDATE_FILE_RE = re.compile(r"\*{3}\s*Update\s+File\s*:\s*(.+)", re.IGNORECASE)
_DELETE_FILE_RE = re.compile(r"\*{3}\s*Delete\s+File\s*:\s*(.+)", re.IGNORECASE)
_MOVE_TO_RE = re.compile(r"\*{3}\s*Move\s+to\s*:\s*(.+)", re.IGNORECASE)
_END_OF_FILE_RE = re.compile(r"\*{3}\s*End\s+of\s+File", re.IGNORECASE)
_END_RE = re.compile(r"\*{3}\s*End\s+Patch", re.IGNORECASE)


def parse_apply_patch_to_simple(input_text: str) -> list[dict] | None:
    """尝试将 apply_patch 的 patch 文本解析为标准工具调用参数列表。

    返回列表中每个元素格式：
        {"tool": "write_to_file", "args": {"filePath": ..., "content": ...}}
        {"tool": "replace_in_file", "args": {"filePath": ..., "old_str": ..., "new_str": ..., destinationPath?}}
        {"tool": "delete_file", "args": {"filePath": ...}}

    返回 None — 解析失败（格式错误或段内解析失败）
    """
    if not input_text or not isinstance(input_text, str):
        return None

    # 提取 Begin Patch 和 End Patch 之间的内容
    # 兼容无 Begin Patch 前缀的输入（Codex 有时不带前缀）
    begin_match = _BEGIN_RE.search(input_text)
    if begin_match:
        body_start = begin_match.end()
    else:
        # 无 Begin Patch：尝试从第一个文件头开始
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

    # 按文件头分割为段
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
            # End of File 标记段结束
            if current_action is not None:
                segments.append((current_action, current_path, current_lines))
            current_action = None
            current_path = None
            current_lines = []
        else:
            # diff 内容行：@@ / +/-/空格前缀 / *** Move to:
            stripped = line.strip()
            move_match = _MOVE_TO_RE.match(line)
            if move_match or stripped.startswith("@@") or line.startswith("+") or line.startswith("-") or line.startswith(" "):
                current_lines.append(line)

    # 最后一段
    if current_action is not None:
        segments.append((current_action, current_path, current_lines))

    if not segments:
        return None

    # 逐段解析
    results: list[dict] = []
    for action, path, lines in segments:
        parsed = _parse_segment(action, path, lines)
        if parsed is None:
            # 段级降级：不影响其他段
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
    """解析单个文件段为工具调用参数。"""

    # 所有工具都要求 filePath 非空
    if not path or not path.strip():
        logger.warning("Segment has empty filePath: action=%s", action)
        return None

    if action == "add":
        plus_lines = [ln[1:] for ln in lines if ln.startswith("+")]
        content = "\n".join(plus_lines)
        # write_to_file 要求 content 非空（空文件无意义，且可能与 replace 语义混淆）
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
        # 提取 Move to 目标路径
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

        # 无 diff 行 + 无 Move to → 空段
        if not minus_lines and not plus_lines and dest_path is None:
            logger.warning("Update File segment has no '-'/'+' lines and no Move to: %s", path)
            return None

        # Move to only（无内容修改）
        if not minus_lines and not plus_lines and dest_path is not None:
            # 安全：占位 old_str/new_str 为单个空格（minLength=1）
            return {
                "tool": "replace_in_file",
                "args": {
                    "filePath": path,
                    "old_str": " ",
                    "new_str": " ",
                    "destinationPath": dest_path,
                },
            }

        # 有 diff 行但无 '-' 行 → 只有 '+' 行的 Update File
        old_str = "\n".join(minus_lines)
        if not old_str:
            # 有 context 行 → 用 context 行作为锚点构造 replace_in_file
            # 这对齐 Claude Code 的行为：用已有内容作为 old_str，追加新内容
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

            # 无 context 行也无 '-' 行 → 只有 '+' 行的追加操作
            if plus_lines:
                # 有 Move to → 必须用 replace_in_file（append_to_file 不支持 destinationPath）
                if dest_path is not None:
                    logger.info("Update File has only '+' lines with Move to, degrading to user message: %s", path)
                    return {
                        "tool": "_degraded_user_message",
                        "args": {
                            "content": f"[File {path} was modified (rename to {dest_path}, append) — appended content]\n" + "\n".join(plus_lines),
                        },
                    }
                logger.info("Update File has only '+' lines and no context (append), converting to append_to_file: %s", path)
                return {
                    "tool": "append_to_file",
                    "args": {
                        "filePath": path,
                        "content": "\n".join(plus_lines),
                    },
                }
            # 无 '-' 也无 '+' 行（只有 @@ 行）→ 无效段
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
# 反向转换：简单函数调用参数 → apply_patch DSL
# ---------------------------------------------------------------------------

def write_to_apply_patch(file_path: str, content: str) -> str:
    """将 write_to_file 调用转为 apply_patch Add File 格式。"""
    lines = content.split("\n")
    body = "\n".join(f"+{ln}" for ln in lines)
    return "*** Begin Patch\n*** Add File: {}\n{}\n*** End Patch".format(file_path, body)


def delete_to_apply_patch(file_path: str) -> str:
    """将 delete_file 调用转为 apply_patch Delete File 格式。"""
    return "*** Begin Patch\n*** Delete File: {}\n*** End Patch".format(file_path)


def append_to_apply_patch(file_path: str, content: str) -> str:
    """将 append_to_file 调用转为 apply_patch Update File 格式（只有 '+' 行）。

    Codex 的 apply_patch 执行器支持只有 '+' 行的 Update File，等同于追加到文件末尾。
    """
    lines = content.split("\n")
    body = "\n".join(f"+{ln}" for ln in lines)
    return "*** Begin Patch\n*** Update File: {}\n@@\n{}\n*** End Patch".format(file_path, body)


def replace_to_apply_patch(file_path: str, old_str: str, new_str: str, dest_path: str | None = None) -> str:
    """将 replace_in_file 调用转为 apply_patch Update File 格式。

    对 old_str / new_str 做逐行 diff：提取公共前缀/后缀作为 context 行（空格前缀），
    差异部分作为 -/+ 行。模型如果在 old_str / new_str 中都带 context，
    patch 对 Codex 的 apply_patch 解析器就是有效的。

    当 dest_path 非空时，在 Update File 行后插入 *** Move to: 行。
    """
    old_lines = old_str.split("\n")
    new_lines = new_str.split("\n")

    # 公共前缀：从头匹配到第一个差异行
    prefix_len = 0
    min_len = min(len(old_lines), len(new_lines))
    while prefix_len < min_len and old_lines[prefix_len] == new_lines[prefix_len]:
        prefix_len += 1

    # 公共后缀：从尾向前匹配（不重叠前缀部分）
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
# 反向映射表
# ---------------------------------------------------------------------------

def build_reverse_tool_map() -> dict[str, str]:
    """返回上游 tool name -> 下游 custom tool name 的映射。"""
    return {
        "write_to_file": "apply_patch",
        "replace_in_file": "apply_patch",
        "delete_file": "apply_patch",
        "append_to_file": "apply_patch",
    }


# ---------------------------------------------------------------------------
# 从反向工具的函数参数中提取文件路径和输入文本（用于 StreamState 消费）
# ---------------------------------------------------------------------------

class ReverseConversionError(ValueError):
    """反向转换失败，携带错误信息供上层构造错误 tool result。"""

    def __init__(self, tool_name: str, reason: str, detail: str = ""):
        self.tool_name = tool_name
        self.reason = reason
        self.detail = detail
        msg = f"reverse conversion failed for {tool_name}: {reason}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


def reverse_tool_args_to_apply_patch(tool_name: str, args: dict) -> str:
    """将上游工具调用的 arguments 转为 apply_patch 的 input 字符串。

    Args:
        tool_name: 上游 tool name（"write_to_file" / "replace_in_file" / "delete_file"）
        args: 解析后的 arguments dict

    Returns:
        apply_patch 格式的 patch 字符串

    Raises:
        ReverseConversionError: 参数验证失败，携带原因和修正建议
    """
    if tool_name == "write_to_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        content = _extract_arg(args, _CONTENT_KEYS)
        if file_path is None or content is None:
            raise ReverseConversionError(
                tool_name,
                "filePath and content are required",
                "write_to_file requires both filePath and content parameters. "
                "For long content, consider using append_to_file to add content in smaller chunks.",
            )
        if not content:
            raise ReverseConversionError(
                tool_name,
                "content must not be empty",
                "Provide actual file content, or use append_to_file to build the file incrementally.",
            )
        return write_to_apply_patch(file_path, content)

    if tool_name == "replace_in_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        old_str = _extract_arg(args, _OLD_STR_KEYS)
        new_str = _extract_arg(args, _NEW_STR_KEYS)
        dest_path = _extract_arg(args, _DEST_PATH_KEYS)
        if file_path is None or old_str is None or new_str is None:
            raise ReverseConversionError(
                tool_name,
                "filePath, old_str, and new_str are required",
                "replace_in_file requires filePath, old_str, and new_str parameters.",
            )
        if not old_str:
            raise ReverseConversionError(
                tool_name,
                "old_str must not be empty",
                "When appending to a file, use append_to_file instead. "
                "For replace_in_file, include the last few lines of the file as old_str "
                "and add new content in new_str.",
            )
        return replace_to_apply_patch(file_path, old_str, new_str, dest_path=dest_path)

    if tool_name == "delete_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        if file_path is None:
            raise ReverseConversionError(
                tool_name,
                "filePath is required",
                "delete_file requires a filePath parameter.",
            )
        return delete_to_apply_patch(file_path)

    if tool_name == "append_to_file":
        file_path = _extract_arg(args, _PATH_KEYS)
        content = _extract_arg(args, _CONTENT_KEYS)
        if file_path is None or content is None:
            raise ReverseConversionError(
                tool_name,
                "filePath and content are required",
                "append_to_file requires both filePath and content parameters.",
            )
        return append_to_apply_patch(file_path, content)

    raise ReverseConversionError(
        tool_name,
        "unknown tool name",
        f"Expected one of: write_to_file, replace_in_file, append_to_file, delete_file. Got: {tool_name}",
    )
