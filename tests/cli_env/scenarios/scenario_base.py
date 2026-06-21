"""Scenario 基类 — setup → run → verify → teardown → report"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tests.cli_env.lib.server import ServerManager
from tests.cli_env.lib.log_collector import LogCollector

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """单个检查项结果"""
    name: str
    expected: str
    actual: str
    passed: bool
    detail: str = ""


@dataclass
class ScenarioReport:
    """场景运行报告"""
    scenario: str
    status: str = ""              # "PASS" | "FAIL" | "ERROR"
    checks: list[CheckResult] = field(default_factory=list)
    server_log: str = ""
    response: Any = None
    duration_ms: float = 0.0
    error: str = ""


class Scenario(ABC):
    """场景基类

    子类需实现:
        name: 场景名称
        setup(): 启动 server + mock upstream
        run(): 发送请求，返回响应
        verify(response): 验证结果
        teardown(): 清理
    """

    def __init__(self):
        self.server = ServerManager()
        self.log_collector = LogCollector()
        self.run_id = uuid.uuid4().hex[:8]

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    async def setup(self) -> dict[str, Any]:
        """子类重写此方法做场景专属初始化

        Returns:
            包含环境信息的 dict（如 mock_upstream_url, config_path, db_path）
        """
        return {}

    async def run(self, env: dict[str, Any]) -> Any:
        """子类重写此方法以发送测试请求

        Args:
            env: setup() 返回的环境信息

        Returns:
            测试响应（任意格式，传给 verify）
        """
        raise NotImplementedError

    async def verify(self, response: Any) -> list[CheckResult]:
        """子类重写此方法以验证结果

        Args:
            response: run() 返回的响应

        Returns:
            检查项列表
        """
        raise NotImplementedError

    async def teardown(self):
        """子类重写此方法做清理"""
        pass

    async def run_all(self) -> ScenarioReport:
        """完整运行流程: setup → run → verify → teardown → report"""
        report = ScenarioReport(scenario=self.name)
        start = time.perf_counter()

        try:
            # 1. Setup
            env = await self.setup()

            # 2. Ingest any initial logs
            log_text = await self.server.read_stderr()
            self.log_collector.ingest(log_text)

            # 3. Run
            response = await self.run(env)
            report.response = response

            # 4. Capture server logs
            log_text = await self.server.read_stderr()
            self.log_collector.ingest(log_text)
            errors = self.log_collector.get_errors()
            if errors:
                report.server_log = "\n".join(e["raw"] for e in errors[:10])

            # 5. Verify
            checks = await self.verify(response)
            report.checks = checks
            report.status = "PASS" if all(c.passed for c in checks) else "FAIL"

        except Exception as e:
            logger.exception("Scenario %s failed with exception", self.name)
            report.status = "ERROR"
            report.error = f"{type(e).__name__}: {e}"
            # Try to capture logs anyway
            try:
                log_text = await self.server.read_stderr()
                if log_text:
                    self.log_collector.ingest(log_text)
                    errors = self.log_collector.get_errors()
                    if errors:
                        report.server_log = "\n".join(e["raw"] for e in errors[:10])
            except Exception:
                pass
        finally:
            report.duration_ms = (time.perf_counter() - start) * 1000
            # 4. Teardown
            try:
                await self.server.stop()
                await self.teardown()
            except Exception as e:
                logger.warning("Teardown error for %s: %s", self.name, e)

        return report

    def check(self, name: str, passed: bool, expected: str, actual: str,
              detail: str = "") -> CheckResult:
        return CheckResult(
            name=name,
            expected=expected,
            actual=actual,
            passed=passed,
            detail=detail,
        )
