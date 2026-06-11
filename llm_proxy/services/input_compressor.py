"""输入压缩服务 — 借鉴 RTK 思路压缩发给 LLM 的输入内容以节省 input token。

主要压缩目标：
- tool_result 中的 CLI 输出（git status、cargo build、npm install 等）
- 长代码块
- 冗余空白行和绝对路径

压缩策略（MVP）：
- drop_progress: 丢弃进度条/百分比行等噪声
- truncate: 截断超长代码块，保留首尾
- collapse: 折叠多余空行
- shorten_paths: 绝对路径 → 相对路径
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── 配置 ────────────────────────────────────────────────────────────

# tool_result 文本长度低于此值不做压缩
_MIN_TOOL_RESULT_LENGTH = 10

# 通用文本（system prompt、user/assistant 消息）长度低于此值不做压缩
_MIN_TEXT_LENGTH = 10


@dataclass
class CompressionConfig:
    """输入压缩配置"""
    enabled: bool = False
    strategies: list[str] = field(default_factory=list)
    truncate_max_lines: int = 200
    truncate_indicator: str = "... [truncated {n} lines]"
    collapse_max_blank_lines: int = 2
    shorten_paths_enabled: bool = True

    def has_strategy(self, name: str) -> bool:
        return name in self.strategies


# ─── 统计 ────────────────────────────────────────────────────────────

@dataclass
class CompressionStats:
    """压缩统计（用于日志/debug）"""
    original_chars: int = 0
    compressed_chars: int = 0
    items_compressed: int = 0

    def record(self, original: str, compressed: str) -> None:
        self.original_chars += len(original)
        self.compressed_chars += len(compressed)
        self.items_compressed += 1


# ─── 策略：预编译正则 ────────────────────────────────────────────────

# 丢弃进度条、编译/下载等噪声行（不包含空行，空行由 collapse 处理）
_DROP_PROGRESS_PATTERNS: list[re.Pattern] = [
    re.compile(r'^\s*\d+%\s'),                       # 进度百分比 " 42% "
    re.compile(r'^\s*\d+[./]\d+[kKMG]?%'),           # "1.2K%" / "42.3%"
    re.compile(r'^\s*(Compiling|Downloading|Building|Installing)\s', re.IGNORECASE),
    re.compile(r'^\s*Step\s+\d+/\d+'),               # Docker build steps
    re.compile(r'^\s*(added|removed|changed)\s+\d+\s+package', re.IGNORECASE),  # npm
    re.compile(r'^\s*Fresh\s'),                      # cargo fresh
    re.compile(r'^\s*Finished\s', re.IGNORECASE),    # cargo/rustc finished
    re.compile(r'^\s*(Running|Downloading)\s+.*\.\.\.$'),  # cargo running...
    re.compile(r'^\s*[=|#*\-]{5,}\s*$'),             # 进度条 [=====>    ]
]

# 代码块围栏检测
_CODE_FENCE_START = re.compile(r'^```')

# 项目根路径候选（用于 shorten_paths）
# 匹配 /Users/{name}/{1-2 dirs}/ — 项目根目录层级
_ROOT_CANDIDATES = re.compile(
    r'(/(?:Users|home)/\w+/(?:\w+/){1,2})'
)


# ─── 核心压缩类 ──────────────────────────────────────────────────────

class InputCompressor:
    """输入压缩器 — 对请求体中的冗余文本做压缩"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.stats = CompressionStats()
        self._detected_roots: set[str] = set()  # shorten_paths 缓存

    # ─── 公共接口 ────────────────────────────────────────────────

    def compress_tool_result(self, text: str | None) -> str | None:
        """压缩 tool_result 文本 — 应用所有策略"""
        if not text or len(text) < _MIN_TOOL_RESULT_LENGTH:
            return text

        original = text

        if self.config.has_strategy("drop_progress"):
            text = self._drop_progress(text)

        if self.config.has_strategy("shorten_paths"):
            text = self._shorten_paths(text)

        if self.config.has_strategy("truncate"):
            text = self._truncate(text)

        if self.config.has_strategy("collapse"):
            text = self._collapse(text)

        if text != original:
            self.stats.record(original, text)

        return text

    def compress_text(self, text: str | None) -> str | None:
        """压缩通用文本（非 tool_result）— 仅应用安全策略（collapse）"""
        if not text or len(text) < _MIN_TEXT_LENGTH:
            return text

        original = text

        if self.config.has_strategy("collapse"):
            text = self._collapse(text)

        if text != original:
            self.stats.record(original, text)

        return text

    # ─── Anthropic 格式 ─────────────────────────────────────────

    def compress_anthropic_body(self, body: dict) -> None:
        """压缩 Anthropic Messages 格式请求体（原地修改）"""
        # 压缩 tool_result content blocks
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        self._compress_anthropic_tool_result(block)
            # 纯文本 user/assistant 消息中也可能有长内容，只做 collapse
            elif isinstance(content, str):
                msg["content"] = self.compress_text(content)

        # system prompt 只做 collapse
        system = body.get("system")
        if isinstance(system, str):
            body["system"] = self.compress_text(system)
        elif isinstance(system, list):
            for block in system:
                text = block.get("text", "")
                if text:
                    block["text"] = self.compress_text(text)

    def _compress_anthropic_tool_result(self, block: dict) -> None:
        """压缩单个 Anthropic tool_result block"""
        content = block.get("content")
        if isinstance(content, str):
            block["content"] = self.compress_tool_result(content)
        elif isinstance(content, list):
            for sub in content:
                if sub.get("type") == "text":
                    sub["text"] = self.compress_tool_result(sub.get("text", ""))

    # ─── Chat 格式 ──────────────────────────────────────────────

    def compress_chat_body(self, body: dict) -> None:
        """压缩 OpenAI Chat Completions 格式请求体（原地修改）"""
        for msg in body.get("messages", []):
            role = msg.get("role")
            if role == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = self.compress_tool_result(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = self.compress_tool_result(part.get("text", ""))
            elif role in ("user", "assistant"):
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = self.compress_text(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                part["text"] = self.compress_text(text)

    # ─── Responses 格式 ─────────────────────────────────────────

    def compress_responses_body(self, body: dict) -> None:
        """压缩 OpenAI Responses 格式请求体（原地修改）"""
        for item in body.get("input", []):
            item_type = item.get("type", "")

            if item_type == "function_call_output":
                output = item.get("output", "")
                if isinstance(output, str):
                    item["output"] = self.compress_tool_result(output)

            elif item_type in ("message",):
                # Responses 中的 message 项
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text_key = "text" if "text" in part else None
                            if text_key and part.get(text_key):
                                part[text_key] = self.compress_text(part[text_key])

        # instructions（system prompt 等价）只做 collapse
        instructions = body.get("instructions", "")
        if isinstance(instructions, str):
            body["instructions"] = self.compress_text(instructions)

    # ─── 策略实现 ────────────────────────────────────────────────

    def _drop_progress(self, text: str) -> str:
        """丢弃进度条、编译下载等噪声行"""
        lines = text.split('\n')
        kept = []
        for line in lines:
            stripped = line.strip()
            # 空行保留（collapse 策略负责处理多余空行）
            if not stripped:
                kept.append(line)
                continue
            # 检查是否匹配噪声模式
            if any(p.match(stripped) for p in _DROP_PROGRESS_PATTERNS):
                continue
            kept.append(line)
        return '\n'.join(kept)

    def _truncate(self, text: str) -> str:
        """截断超长代码块，保留首尾行 + 截断指示器"""
        max_lines = self.config.truncate_max_lines
        indicator = self.config.truncate_indicator

        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text

        # 检测是否包含代码块围栏
        in_code = False
        code_block_ranges: list[tuple[int, int]] = []
        block_start = -1

        for i, line in enumerate(lines):
            if _CODE_FENCE_START.match(line.strip()):
                if in_code:
                    code_block_ranges.append((block_start, i))
                    in_code = False
                else:
                    block_start = i
                    in_code = True

        # 如果有未闭合的代码块，忽略最后一个
        if in_code:
            code_block_ranges.append((block_start, len(lines) - 1))

        # 如果没有代码块，不对纯文本做 truncate
        if not code_block_ranges:
            return text

        # 只截断超过阈值的代码块
        result_lines = list(lines)
        # 从后往前截断，避免索引偏移
        for start, end in reversed(code_block_ranges):
            block_line_count = end - start - 1  # 减去围栏行
            if block_line_count <= max_lines:
                continue

            head_count = max_lines // 2
            tail_count = min(10, block_line_count)
            truncated_count = block_line_count - head_count - tail_count

            # 替换中间部分为指示器
            head_lines = result_lines[start + 1:start + 1 + head_count]
            tail_lines = result_lines[end - tail_count:end]
            indicator_line = indicator.format(n=truncated_count)

            result_lines[start + 1:end] = head_lines + [indicator_line] + tail_lines

        return '\n'.join(result_lines)

    def _collapse(self, text: str) -> str:
        """折叠连续空行为最多 N 个

        collapse_max_blank_lines=2 表示最多 2 个连续空行（即最多 3 个换行符相连）。
        超过的空行被折叠为 max_blank_lines 个。
        """
        max_blank = self.config.collapse_max_blank_lines
        if max_blank < 1:
            return text

        # 将连续空行（可能含空格/制表符）折叠
        # 匹配: 超过 max_blank 个连续空行
        # 一个"空行"= 换行符 + 可选空白 + 换行符
        # 简化：匹配 3+ 连续换行符（含中间可能有空白），替换为 max_blank+1 个换行符
        max_consecutive_newlines = max_blank + 1
        # 匹配超过允许数量的连续换行（中间可能有空白）
        pattern = r'(?:\n\s*){' + str(max_consecutive_newlines + 1) + r',}'
        replacement = '\n' * max_consecutive_newlines
        return re.sub(pattern, replacement, text)

    def _shorten_paths(self, text: str) -> str:
        """将绝对路径替换为相对路径"""
        if not self.config.shorten_paths_enabled:
            return text

        # 检测文本中的项目根路径
        self._detect_roots(text)

        if not self._detected_roots:
            return text

        # 按路径长度降序排列，先替换最长的（更精确的匹配）
        result = text
        for root in sorted(self._detected_roots, key=len, reverse=True):
            escaped_root = re.escape(root)
            result = re.sub(escaped_root, './', result)

        return result

    def _detect_roots(self, text: str) -> None:
        """从文本中检测项目根路径"""
        matches = _ROOT_CANDIDATES.findall(text)
        for match in matches:
            self._detected_roots.add(match)
