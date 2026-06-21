"""场景 2: Protocol Conversion — 验证 Claude Code CLI 路径

Anthropic Messages API → llm-proxy → OpenAI Chat 上游
"""

import logging
from typing import Any

from tests.cli_env.lib import config_loader as cfg
from tests.cli_env.lib.client import send_messages, configure_endpoint
from tests.cli_env.lib.mock_upstream import MockUpstream
from tests.cli_env.scenarios.scenario_base import Scenario, ScenarioReport

logger = logging.getLogger(__name__)


class ProtocolConversionScenario(Scenario):
    """协议转换场景：Messages → proxy → OpenAI upstream"""

    name = "protocol_conversion"

    async def setup(self) -> dict[str, Any]:
        # 1. Start mock upstream
        self.mock = MockUpstream()
        await self.mock.start()
        mock_url = self.mock.get_url()
        mock_port = self.mock.port

        # Configure mock to respond in OpenAI Chat format
        # (llm-proxy converts Anthropic Messages → OpenAI Chat internally,
        #  then the upstream responds in Chat format,
        #  and llm-proxy converts back to Anthropic Messages)
        chat_resp = self.mock.make_chat_response(
            text="Hello from protocol conversion!",
            model="test-model",
        )
        self.mock.set_default_response(status=200, body=chat_resp)

        # 2. Create config — upstream only speaks OpenAI (proxy must do Anthropic→OpenAI→OpenAI→Anthropic)
        model = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model",
            api_key="mock-key",
            display_name="Test OpenAI-Upstream Model",
        )
        config = cfg.create_scenario_config(models={"test-model": model})
        config_path = cfg.write_temp_config(config, prefix="proto-conv-")
        db_path = cfg.scenario_db_path("protocol_conversion", self.run_id)

        # 3. Start llm-proxy
        proxy_port = await self.server.start(
            config_path=config_path,
            db_path=db_path,
        )

        # 4. Create endpoint
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
        # 发送 Anthropic Messages API 请求（Claude Code CLI 线缆格式）
        resp = await send_messages(
            base_url=self.server.url,
            body={
                "model": "test-model",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello! What can you do?"}
                ],
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
        ))

        # Check 2: Response is Anthropic Messages format
        is_messages_format = (
            isinstance(body, dict)
            and body.get("type") == "message"
            and "content" in body
        )
        checks.append(self.check(
            "anthropic_messages_format",
            passed=is_messages_format,
            expected="type=message, content array",
            actual=f"type={body.get('type', 'missing')}" if isinstance(body, dict) else "not a dict",
        ))

        # Check 3: Content has text
        has_text = False
        if isinstance(body, dict) and "content" in body:
            content = body["content"]
            if isinstance(content, list):
                has_text = any(
                    isinstance(block, dict) and block.get("type") == "text"
                    for block in content
                )
        checks.append(self.check(
            "content_has_text",
            passed=has_text,
            expected="text block in content",
            actual="text found" if has_text else "no text block",
        ))

        # Check 4: Has stop_reason
        has_stop = isinstance(body, dict) and body.get("stop_reason") is not None
        checks.append(self.check(
            "has_stop_reason",
            passed=has_stop,
            expected="stop_reason present",
            actual=f"stop_reason={body.get('stop_reason', 'missing')}" if isinstance(body, dict) else "N/A",
        ))

        # Check 5: No protocol conversion errors in logs
        errors = self.log_collector.get_errors()
        conv_errors = [e for e in errors if "conversion" in e["message"].lower()
                       or "protocol" in e["message"].lower()]
        checks.append(self.check(
            "no_conversion_errors",
            passed=len(conv_errors) == 0,
            expected="no conversion errors",
            actual=f"{len(conv_errors)} conversion errors" if conv_errors else "clean",
        ))

        # Check 6: Mock upstream received the request
        call_log = self.mock.call_log
        upstream_received = len(call_log) > 0
        checks.append(self.check(
            "upstream_received",
            passed=upstream_received,
            expected="upstream received request",
            actual=f"received {len(call_log)} requests" if upstream_received else "no requests",
        ))

        return checks

    async def teardown(self):
        if hasattr(self, 'mock'):
            await self.mock.stop()
