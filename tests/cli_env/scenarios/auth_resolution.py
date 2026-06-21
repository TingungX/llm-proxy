"""场景 4: Auth + Model Resolution — 验证鉴权和模型解析全路径

测试各种鉴权场景：无 Key、错误 Key、正确 Key、default 端点、family_routing
"""

import logging
from typing import Any

from tests.cli_env.lib import config_loader as cfg
from tests.cli_env.lib.client import (
    send_responses, send_messages, configure_endpoint, get_endpoints
)
from tests.cli_env.lib.mock_upstream import MockUpstream
from tests.cli_env.scenarios.scenario_base import Scenario, ScenarioReport

logger = logging.getLogger(__name__)


class AuthResolutionScenario(Scenario):
    """鉴权+模型解析场景"""

    name = "auth_resolution"

    async def setup(self) -> dict[str, Any]:
        # 1. Start mock upstream
        self.mock = MockUpstream()
        await self.mock.start()
        mock_url = self.mock.get_url()
        mock_port = self.mock.port

        chat_resp = self.mock.make_chat_response(text="Auth test OK", model="test-model")
        self.mock.set_default_response(status=200, body=chat_resp)

        # 2. Config with 2 models
        model_a = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model-a",
            api_key="mock-key",
            display_name="Model A",
        )
        model_b = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model-b",
            api_key="mock-key",
            display_name="Model B",
        )
        config = cfg.create_scenario_config(
            models={
                "model-a": model_a,
                "model-b": model_b,
            },
            error_handling={"failover_enabled": True, "no_retry_enabled": True},
        )
        config_path = cfg.write_temp_config(config, prefix="auth-")
        db_path = cfg.scenario_db_path("auth_resolution", self.run_id)

        # 3. Start llm-proxy
        proxy_port = await self.server.start(
            config_path=config_path,
            db_path=db_path,
        )

        # 4. Create endpoints:
        #    - Endpoint with specific key "valid-key"
        #    - Default endpoint is auto-created
        ep_specific = cfg.build_endpoint_payload(
            name="valid-ep",
            api_key="valid-key",
            models=[],
        )
        await configure_endpoint(self.server.url, ep_specific)

        return {
            "mock": self.mock,
            "config_path": config_path,
            "db_path": db_path,
        }

    async def run(self, env: dict[str, Any]) -> dict[str, Any]:
        results = {}

        # Test 1: No API Key
        logger.info("Test 1: No API Key")
        results["no_key"] = await send_responses(
            base_url=self.server.url,
            body={"model": "model-a", "input": "hi"},
            api_key=None,
        )

        # Test 2: Wrong API Key
        logger.info("Test 2: Wrong API Key")
        results["wrong_key"] = await send_responses(
            base_url=self.server.url,
            body={"model": "model-a", "input": "hi"},
            api_key="wrong-key",
        )

        # Test 3: Valid Key + unknown model
        logger.info("Test 3: Valid Key + Unknown model")
        results["unknown_model"] = await send_responses(
            base_url=self.server.url,
            body={"model": "nonexistent-model", "input": "hi"},
            api_key="valid-key",
        )

        # Test 4: Valid Key + existing model
        logger.info("Test 4: Valid Key + Existing model")
        results["valid_request"] = await send_responses(
            base_url=self.server.url,
            body={"model": "model-a", "input": "Hello!"},
            api_key="valid-key",
        )

        # Test 5: Default endpoint (empty API Key → "default")
        logger.info("Test 5: Default endpoint")
        results["default_endpoint"] = await send_responses(
            base_url=self.server.url,
            body={"model": "model-b", "input": "Hello default!"},
            api_key="default",
        )

        # Test 6: Family routing test — create with family_routing
        logger.info("Test 6: Family routing")
        ep_with_routing = cfg.build_endpoint_payload(
            name="routing-ep",
            api_key="routing-key",
            models=[],
            family_routing={
                "test-family": {"target": "model-a"},
            },
        )
        await configure_endpoint(self.server.url, ep_with_routing)
        results["family_routing"] = await send_responses(
            base_url=self.server.url,
            body={"model": "test-family", "input": "Routing test"},
            api_key="routing-key",
        )

        return results

    async def verify(self, response: dict[str, Any]) -> list:
        checks = []

        # Check 1: No Key → 401
        no_key = response.get("no_key", (0, {}, {}, 0))
        status = no_key[0]
        checks.append(self.check(
            "no_key_returns_401",
            passed=status == 401,
            expected="401",
            actual=str(status),
        ))

        # Check 2: Wrong Key → 401
        wrong_key = response.get("wrong_key", (0, {}, {}, 0))
        status = wrong_key[0]
        checks.append(self.check(
            "wrong_key_falls_to_default",
            passed=status == 200,
            expected="200 (default endpoint)",
            actual=str(status),
        ))

        # Check 3: Valid Key + Unknown Model → 400 (model not found)
        unknown = response.get("unknown_model", (0, {}, {}, 0))
        status = unknown[0]
        body = unknown[2] if len(unknown) > 2 else {}
        checks.append(self.check(
            "unknown_model_returns_400",
            passed=status == 400,
            expected="400",
            actual=str(status),
            detail=f"body: {body}",
        ))

        # Check 4: Valid Key + Valid Model → 200
        valid = response.get("valid_request", (0, {}, {}, 0))
        status = valid[0]
        checks.append(self.check(
            "valid_request_returns_200",
            passed=status == 200,
            expected="200",
            actual=str(status),
        ))

        # Check 5: Default endpoint → 200
        default = response.get("default_endpoint", (0, {}, {}, 0))
        status = default[0]
        checks.append(self.check(
            "default_endpoint_returns_200",
            passed=status == 200,
            expected="200",
            actual=str(status),
        ))

        # Check 6: Family routing → 200
        routing = response.get("family_routing", (0, {}, {}, 0))
        status = routing[0]
        checks.append(self.check(
            "family_routing_returns_200",
            passed=status == 200,
            expected="200",
            actual=str(status),
        ))

        return checks

    async def teardown(self):
        if hasattr(self, 'mock'):
            await self.mock.stop()
