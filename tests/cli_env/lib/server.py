"""llm-proxy 实例管理 — 启动/停止/健康检查

管理两个进程：
1. llm-proxy（待测）：运行 uvicorn llm_proxy.main:app
2. mock upstream（模拟上游）：模拟 LLM API
"""

import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
UVICORN_BIN = str(VENV_DIR / "bin" / "uvicorn")
PYTHON_BIN = str(VENV_DIR / "bin" / "python3.14")


class ServerManager:
    """管理 llm-proxy 子进程"""

    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._port: int = 0
        self._config_path: str | None = None
        self._db_path: str | None = None
        self._env: dict[str, str] = {}

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    async def start(
        self,
        port: int = 0,
        config_path: str | None = None,
        db_path: str | None = None,
        log_level: str = "DEBUG",
        extra_env: dict[str, str] | None = None,
    ) -> int:
        """启动 llm-proxy 服务器

        Args:
            port: 端口，0 表示随机分配
            config_path: 自定义 config.json 路径
            db_path: 自定义 usage.db 路径
            log_level: 日志级别
            extra_env: 额外环境变量

        Returns:
            实际端口号
        """
        self._port = port or 0
        self._config_path = config_path
        self._db_path = db_path

        # 构建环境变量
        self._env = {
            "LLM_PROXY_DEV": "true",
            "LLM_PROXY_LOG_LEVEL": log_level,
            "PYTHONUNBUFFERED": "1",
            **os.environ,
        }
        if config_path:
            self._env["LLM_PROXY_CONFIG_PATH"] = config_path
        if db_path:
            self._env["LLM_PROXY_DB_PATH"] = db_path
        if extra_env:
            self._env.update(extra_env)

        # 构建 uvicorn 命令
        cmd = [
            PYTHON_BIN, "-m", "uvicorn",
            "llm_proxy.main:app",
            "--host", "127.0.0.1",
            "--port", str(port) if port > 0 else "0",
        ]

        logger.info("Starting llm-proxy: %s (config=%s, db=%s)", " ".join(cmd), config_path, db_path)

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_DIR),
            env=self._env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 如果 port=0，需要从 stderr 获取实际端口
        if port == 0:
            self._port = await self._discover_port()
        else:
            self._port = port

        # 等待服务器就绪
        await self._wait_ready(timeout=15)

        logger.info("llm-proxy ready on 127.0.0.1:%s", self._port)
        return self._port

    async def stop(self):
        """停止 llm-proxy 服务器"""
        if self._process:
            logger.info("Stopping llm-proxy (pid=%s)...", self._process.pid)
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("llm-proxy did not terminate, sending SIGKILL")
                self._process.kill()
                await self._process.wait()
            self._process = None
            logger.info("llm-proxy stopped")

    async def read_stderr(self) -> str:
        """读取服务器 stderr（包含日志）"""
        if self._process and self._process.stderr:
            try:
                data = await asyncio.wait_for(self._process.stderr.read(), timeout=2)
                return data.decode("utf-8", errors="replace")
            except (asyncio.TimeoutError, Exception):
                return ""
        return ""

    async def read_stdout(self) -> str:
        """读取服务器 stdout"""
        if self._process and self._process.stdout:
            try:
                data = await asyncio.wait_for(self._process.stdout.read(), timeout=2)
                return data.decode("utf-8", errors="replace")
            except (asyncio.TimeoutError, Exception):
                return ""
        return ""

    async def _discover_port(self) -> int:
        """从 stderr 输出中解析端口号（uvicorn 将启动信息输出到 stderr）"""
        if not self._process or not self._process.stderr:
            return 0

        start = time.monotonic()
        while time.monotonic() - start < 10:
            line = await asyncio.wait_for(
                self._process.stderr.readline(), timeout=10
            )
            text = line.decode("utf-8", errors="replace")
            # uvicorn 输出 "Uvicorn running on http://127.0.0.1:XXXXX"
            if "Uvicorn running on" in text:
                import re
                m = re.search(r"http://127\.0\.0\.1:(\d+)", text)
                if m:
                    port = int(m.group(1))
                    logger.info("Discovered port: %s", port)
                    return port
        raise RuntimeError("Could not discover llm-proxy port from logs")

    async def _wait_ready(self, timeout: int = 15) -> None:
        """等待服务器 /health 返回 200"""
        start = time.monotonic()
        last_err = ""
        async with httpx.AsyncClient() as client:
            while time.monotonic() - start < timeout:
                try:
                    r = await client.get(
                        f"http://127.0.0.1:{self._port}/health",
                        timeout=2,
                    )
                    if r.status_code == 200:
                        return
                except Exception as e:
                    last_err = str(e)
                await asyncio.sleep(0.3)
        raise RuntimeError(
            f"llm-proxy did not become ready in {timeout}s (last error: {last_err})"
        )


async def managed_server(
    port: int = 0,
    config_path: str | None = None,
    db_path: str | None = None,
    log_level: str = "DEBUG",
) -> AsyncGenerator[ServerManager, None]:
    """Context manager: 启动 → yield → 停止"""
    mgr = ServerManager()
    try:
        await mgr.start(
            port=port,
            config_path=config_path,
            db_path=db_path,
            log_level=log_level,
        )
        yield mgr
    finally:
        await mgr.stop()
