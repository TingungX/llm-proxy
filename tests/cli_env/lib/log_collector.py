"""服务端日志捕获与过滤

从 uvicorn 进程的 stderr 中实时捕获日志，
支持按 request_id 过滤 access log，通过 /api/logs 查询。
"""

import re
import logging
from collections import defaultdict
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# 日志行匹配: "2026-06-19 12:34:56 INFO [abc12345] llm_proxy.xxx: message"
_LOG_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(\w+)\s+"
    r"\[([a-f0-9]{8}|-)\]\s+"
    r"([\w.]+):\s+(.*)"
)


class LogCollector:
    """日志收集器 — 累积 stderr 日志，提供按级别/logger/request_id 过滤"""

    def __init__(self):
        self._raw: list[str] = []
        self._parsed: list[dict] = []
        self._by_request_id: dict[str, list[dict]] = defaultdict(list)

    def ingest(self, text: str):
        """摄入原始日志文本（追加模式）"""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self._raw.append(line)
            parsed = self._parse_line(line)
            if parsed:
                self._parsed.append(parsed)
                rid = parsed["request_id"]
                if rid and rid != "-":
                    self._by_request_id[rid].append(parsed)

    def get_all(self) -> list[dict]:
        return list(self._parsed)

    def get_by_request_id(self, request_id: str) -> list[dict]:
        return list(self._by_request_id.get(request_id, []))

    def get_by_level(self, level: str) -> list[dict]:
        return [e for e in self._parsed if e["level"] == level]

    def get_by_logger(self, name: str) -> list[dict]:
        return [e for e in self._parsed if e["logger"] == name]

    def get_errors(self) -> list[dict]:
        return [e for e in self._parsed if e["level"] in ("ERROR", "CRITICAL")]

    def get_access_logs(self) -> list[dict]:
        return self.get_by_logger("llm_proxy.access")

    def grep(self, pattern: str) -> list[str]:
        """在原始日志中搜索匹配的行"""
        return [l for l in self._raw if re.search(pattern, l)]

    def clear(self):
        self._raw.clear()
        self._parsed.clear()
        self._by_request_id.clear()

    def _parse_line(self, line: str) -> dict | None:
        m = _LOG_PATTERN.match(line)
        if not m:
            return None
        return {
            "timestamp": m.group(1),
            "level": m.group(2),
            "request_id": m.group(3),
            "logger": m.group(4),
            "message": m.group(5),
            "raw": line,
        }


async def query_server_logs(
    base_url: str,
    request_id: str | None = None,
    limit: int = 50,
    timeout: float = 10,
) -> list[dict]:
    """通过 /api/logs 查询服务器记录的使用日志"""
    params = {"limit": str(limit)}
    if request_id:
        params["request_id"] = request_id

    url = f"{base_url.rstrip('/')}/api/logs/list"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        r = await client.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("records", data.get("logs", []))
        return []
