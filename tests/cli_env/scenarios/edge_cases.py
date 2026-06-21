"""场景 5: Edge Cases — 错误处理、压缩、failover
"""

import json
import logging
from typing import Any

from tests.cli_env.lib import config_loader as cfg
from tests.cli_env.lib.client import send_responses, send_messages, configure_endpoint
from tests.cli_env.lib.mock_upstream import MockUpstream
from tests.cli_env.scenarios.scenario_base import Scenario, ScenarioReport

logger = logging.getLogger(__name__)


class EdgeCasesScenario(Scenario):
    """边缘场景"""

    name = "edge_cases"

    async def setup(self) -> dict[str, Any]:
        # 1. Start mock upstream
        self.mock = MockUpstream()
        await self.mock.start()
        mock_url = self.mock.get_url()
        mock_port = self.mock.port

        chat_resp_ok = self.mock.make_chat_response(
            text="OK response", model="test-model"
        )

        # Default response: 200 OK
        self.mock.set_default_response(status=200, body=chat_resp_ok)

        # 2. Config with 2 models (for failover test) + compression enabled
        model_primary = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model-primary",
            api_key="mock-key",
            display_name="Primary Model",
        )
        # This second model won't actually be used for failover (failover needs same family_routing)
        # But we include it for completeness
        model_secondary = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model-secondary",
            api_key="mock-key",
            display_name="Secondary Model",
        )

        config = cfg.create_scenario_config(
            models={
                "test-model": model_primary,
                "test-model-2": model_secondary,
            },
            error_handling={"failover_enabled": True, "no_retry_enabled": True},
            compression={"enabled": True, "max_input_tokens": 80000},
        )
        config_path = cfg.write_temp_config(config, prefix="edge-")
        db_path = cfg.scenario_db_path("edge_cases", self.run_id)

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
            "config_path": config_path,
            "db_path": db_path,
        }

    async def run(self, env: dict[str, Any]) -> dict[str, Any]:
        results = {}

        # Test 1: Malformed body → 400
        logger.info("Test 1: Malformed body")
        import httpx
        try:
            r = await httpx.AsyncClient().post(
                f"{self.server.url}/v1/responses",
                content=b"this is not json",
                headers={"Content-Type": "application/json", "x-api-key": "test-key"},
            )
            results["malformed_body"] = (r.status_code, dict(r.headers), r.text, 0)
        except Exception as e:
            results["malformed_body"] = (0, {}, str(e), 0)

        # Test 2: Request with valid body (normal test)
        logger.info("Test 2: Valid request")
        results["valid_request"] = await send_responses(
            base_url=self.server.url,
            body={"model": "test-model", "input": "Hello!"},
            api_key="test-key",
        )

        # Test 3: Request with large prompt (compression test)
        logger.info("Test 3: Large prompt (compression)")
        large_prompt = "Hello world! " * 5000  # ~75k chars
        results["large_prompt"] = await send_responses(
            base_url=self.server.url,
            body={"model": "test-model", "input": large_prompt},
            api_key="test-key",
        )

        # Test 4: Upstream 5xx — set mock to return 503 then reset
        logger.info("Test 4: Upstream 5xx")
        error_body = {"error": {"message": "Service Unavailable", "type": "server_error"}}
        self.mock.set_default_response(status=503, body=error_body)
        results["upstream_5xx"] = await send_responses(
            base_url=self.server.url,
            body={"model": "test-model", "input": "Hello!"},
            api_key="test-key",
        )
        # Reset mock
        ok_body = self.mock.make_chat_response(text="OK after failover", model="test-model")
        self.mock.set_default_response(status=200, body=ok_body)

        # Test 5: max_tokens = 1 (minimal output)
        logger.info("Test 5: Minimal max_tokens")
        results["minimal_tokens"] = await send_responses(
            base_url=self.server.url,
            body={"model": "test-model", "input": "Write a long story.", "max_output_tokens": 1},
            api_key="test-key",
        )

        return results

    async def verify(self, response: dict[str, Any]) -> list:
        checks = []

        # Check 1: Malformed body → 400
        malformed = response.get("malformed_body", (0, {}, "", 0))
        checks.append(self.check(
            "malformed_body_returns_400",
            passed=malformed[0] == 400,
            expected="400",
            actual=str(malformed[0]),
        ))

        # Check 2: Valid request → 200
        valid = response.get("valid_request", (0, {}, {}, 0))
        checks.append(self.check(
            "valid_request_returns_200",
            passed=valid[0] == 200,
            expected="200",
            actual=str(valid[0]),
        ))

        # Check 3: Large prompt → 200 (compression may succeed)
        large = response.get("large_prompt", (0, {}, {}, 0))
        checks.append(self.check(
            "large_prompt_returns_200",
            passed=large[0] == 200,
            expected="200",
            actual=str(large[0]),
        ))

        # Check 4: Upstream 5xx → proxy returns error (499 or 503)
        upstream_err = response.get("upstream_5xx", (0, {}, {}, 0))
        status = upstream_err[0]
        # llm-proxy may return 499 (x-should-retry: false) or pass through 503
        checks.append(self.check(
            "upstream_5xx_handled",
            passed=status in (499, 502, 503),
            expected="499/502/503",
            actual=str(status),
        ))

        # Check 5: Minimal tokens → still completes (truncation)
        minimal = response.get("minimal_tokens", (0, {}, {}, 0))
        checks.append(self.check(
            "minimal_tokens_completes",
            passed=minimal[0] == 200,
            expected="200",
            actual=str(minimal[0]),
        ))

        # Check 6: Mock upstream received all expected requests
        call_log = self.mock.call_log
        total_received = len(call_log)
        # At minimum: valid_request + large_prompt should have reached upstream
        # (5xx may also, malformed doesn't)
        checks.append(self.check(
            "upstream_received_requests",
            passed=total_received >= 3,
            expected=">= 3 requests",
            actual=f"{total_received} requests",
        ))

        return checks

    async def teardown(self):
        if hasattr(self, 'mock'):
            await self.mock.stop()
