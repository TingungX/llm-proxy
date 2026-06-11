"""Think tag (<think>...</think>) detection and extraction for chat completion responses.

MiniMax M3 and similar models embed reasoning content inside <think> tags
in the content field, which needs to be extracted and separated from visible text.
This module provides the state machine for streaming extraction and a
convenience function for non-streaming (complete text) extraction.
"""

import logging

logger = logging.getLogger(__name__)

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


def _is_strict_prefix(full: str, s: str) -> bool:
    return len(s) > 0 and len(s) < len(full) and full.startswith(s)


def _suffix_could_be_prefix(s: str, tag: str) -> int:
    best = 0
    for length in range(1, min(len(s), len(tag))):
        suffix = s[-length:]
        if tag.startswith(suffix):
            best = length
    return best


class ThinkTagStateMachine:
    def __init__(self):
        self.state: str = "none"
        self.can_start: bool = True
        self.leading_ws: str = ""
        self.buf: str = ""
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._reasoning_drain_buf: str = ""

    def reset(self):
        self.state = "none"
        self.can_start = True
        self.leading_ws = ""
        self.buf = ""
        self._reasoning_parts = []
        self._content_parts = []
        self._reasoning_drain_buf = ""

    def feed(self, chunk: str) -> tuple[list[str], list[str]]:
        self._reasoning_parts = []
        self._content_parts = []

        text = self.buf + chunk
        self.buf = ""

        if not text:
            return self._reasoning_parts, self._content_parts

        if self.state == "done":
            self._emit_content(text)
            return self._reasoning_parts, self._content_parts

        if self.state == "inside":
            self._process_inside(text)
            return self._reasoning_parts, self._content_parts

        self._process_none(text)
        return self._reasoning_parts, self._content_parts

    def drain(self) -> tuple[str, bool]:
        if self.state == "inside":
            remaining = self._reasoning_drain_buf
            self._reasoning_drain_buf = ""
            if self.buf:
                if remaining:
                    remaining = remaining + self.buf
                else:
                    remaining = self.buf
                self.buf = ""
            return remaining, remaining != ""

        remaining = self.leading_ws + self.buf
        self.leading_ws = ""
        self.buf = ""
        return remaining, False

    def _emit_reasoning(self, text: str):
        if text:
            self._reasoning_parts.append(text)

    def _emit_content(self, text: str):
        if text:
            self._content_parts.append(text)

    def _flush_leading_ws(self):
        if self.leading_ws:
            self._content_parts.append(self.leading_ws)
            self.leading_ws = ""

    def _process_none(self, text: str):
        i = 0
        while i < len(text):
            if not self.can_start:
                self._flush_leading_ws()
                self._emit_content(text[i:])
                return

            ch = text[i]

            if ch in (" ", "\t", "\r", "\n"):
                self.leading_ws += ch
                i += 1
                continue

            if ch == "<":
                tag_match, consumed = self._try_match_tag(text, i, OPEN_TAG)
                if tag_match:
                    self.leading_ws = ""
                    self.state = "inside"
                    self.can_start = False
                    after = text[i + consumed:]
                    if after:
                        self._process_inside(after)
                    return

                partial_len = self._check_partial_tag(text, i, OPEN_TAG)
                if partial_len > 0:
                    self.buf = text[i:]
                    return

                self.can_start = False
                self._flush_leading_ws()
                self._emit_content(text[i:])
                return

            self.can_start = False
            self._flush_leading_ws()
            self._emit_content(text[i:])
            return

    def _process_inside(self, text: str):
        i = 0
        while i < len(text):
            if text[i] == "<":
                tag_match, consumed = self._try_match_tag(text, i, CLOSE_TAG)
                if tag_match:
                    if i > 0:
                        self._emit_reasoning(text[:i])
                    self.state = "done"
                    after = text[i + consumed:]
                    if after:
                        self._emit_content(after)
                    return

                partial_len = self._check_partial_tag(text, i, CLOSE_TAG)
                if partial_len > 0:
                    if i > 0:
                        self._emit_reasoning(text[:i])
                    self.buf = text[i:]
                    return

            i += 1

        if text:
            self._emit_reasoning(text)
            self._reasoning_drain_buf = text

    def _try_match_tag(self, text: str, pos: int, tag: str) -> tuple[bool, int]:
        end = pos + len(tag)
        if end <= len(text) and text[pos:end] == tag:
            return True, len(tag)
        return False, 0

    def _check_partial_tag(self, text: str, pos: int, tag: str) -> int:
        remaining = len(text) - pos
        if remaining <= 0 or remaining >= len(tag):
            return 0
        fragment = text[pos:]
        if tag.startswith(fragment):
            return remaining
        overlap = _suffix_could_be_prefix(fragment, tag)
        if overlap > 0:
            return overlap
        return 0


def strip_think_tags(text: str) -> tuple[str, str]:
    """Strip think tags from a complete text string.

    Returns:
        (reasoning, content) — extracted reasoning text and remaining content.
        If no think tag is found, reasoning is empty and content is the original text.
    """
    if not text:
        return "", ""

    m = ThinkTagStateMachine()
    reasoning_parts, content_parts = m.feed(text)
    drain_result, to_reasoning = m.drain()

    reasoning = "".join(reasoning_parts)
    if to_reasoning and drain_result:
        reasoning = (reasoning + drain_result) if reasoning else drain_result

    content = "".join(content_parts)

    return reasoning, content
