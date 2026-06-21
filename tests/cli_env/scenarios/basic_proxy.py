"""场景 1: Basic Proxy — 验证 Codex CLI wire 路径

Responses API → llm-proxy → OpenAI Chat 上游
"""

import logging
from typing import Any

from tests.cli_env.lib import config_loader as cfg
from tests.cli_env.lib.client import send_responses, configure_endpoint
from tests.cli_env.lib.mock_upstream import MockUpstream
from tests.cli_env.scenarios.scenario_base import Scenario, ScenarioReport

logger = logging.getLogger(__name__)


class BasicProxyScenario(Scenario):
    """基本代理场景：Responses API → proxy → OpenAI upstream"""

    name = "basic_proxy"

    async def setup(self) -> dict[str, Any]:
        # 1. Start mock upstream
        self.mock = MockUpstream()
        await self.mock.start()
        mock_url = self.mock.get_url()
        mock_port = self.mock.port

        # Configure mock upstream response
        chat_resp = self.mock.make_chat_response(text="Hello from mock!")
        self.mock.set_default_response(status=200, body=chat_resp)

        # 2. Create config pointing mock upstream
        model = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model",
            api_key="mock-key",
            display_name="Test Model",
        )
        config = cfg.create_scenario_config(models={"test-model": model})
        config_path = cfg.write_temp_config(config, prefix="basic-proxy-")
        db_path = cfg.scenario_db_path("basic_proxy", self.run_id)

        # 3. Start llm-proxy
        proxy_port = await self.server.start(
            config_path=config_path,
            db_path=db_path,
        )

        # 4. Create endpoint for test
        ep = cfg.build_endpoint_payload(
            name="test-ep",
            api_key="test-key",
            models=[],
        )
        await configure_endpoint(self.server.url, ep)

        return {
            "mock": self.mock,
            "mock_port": mock_port,
            "config_path": config_path,
            "db_path": db_path,
        }

    async def run(self, env: dict[str, Any]) -> Any:
        # 发送 Responses API 请求（Codex CLI 线缆格式）
        resp = await send_responses(
            base_url=self.server.url,
            body={
                "model": "test-model",
                "input": "Hello, what can you do?",
            },
            api_key="test-key",
        )
        return resp

    async def verify(self, response: Any) -> list:
        checks = []
        status, headers, body, elapsed = response

        # Check 1: Status code
        checks.append(self.check(
            "status_code_200",
            passed=status == 200,
            expected="200",
            actual=str(status),
            detail=f"Response status: {status}",
        ))

        # Check 2: Response structure has output
        has_output = isinstance(body, dict) and "output" in body
        checks.append(self.check(
            "response_has_output",
            passed=has_output,
            expected="output field present",
            actual="output found" if has_output else "no output field",
        ))

        # Check 3: Output contains message with text
        # Responses API format: output[] -> {type: "message", content: [{type: "output_text", text: "..."}]}
        has_text = False
        if has_output:
            output = body["output"]
            if isinstance(output, list) and len(output) > 0:
                for msg in output:
                    if isinstance(msg, dict):
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            has_text = any(
                                block.get("type") == "output_text" and block.get("text")
                                for block in content
                                if isinstance(block, dict)
                            )
                            if has_text:
                                break
        checks.append(self.check(
            "response_has_text",
            passed=has_text,
            expected="output_text in content",
            actual="text found" if has_text else "no text content",
        ))

        # Check 4: Response has x-request-id header
        rid = headers.get("x-request-id", headers.get("x-request-id", ""))
        checks.append(self.check(
            "has_request_id",
            passed=bool(rid),
            expected="x-request-id header present",
            actual=f"x-request-id: {rid}" if rid else "missing",
        ))

        # Check 5: Timing is reasonable
        checks.append(self.check(
            "request_time",
            passed=elapsed < 10000,
            expected="< 10000ms",
            actual=f"{elapsed:.1f}ms",
        ))

        # Check 6: Mock upstream received the request
        call_log = self.mock.call_log
        upstream_received = len(call_log) > 0
        checks.append(self.check(
            "upstream_received",
            passed=upstream_received,
            expected="upstream received request",
            actual=f"received {len(call_log)} requests" if upstream_received else "no requests received",
        ))

        return checks

    async def teardown(self):
        if hasattr(self, 'mock'):
            await self.mock.stop()
