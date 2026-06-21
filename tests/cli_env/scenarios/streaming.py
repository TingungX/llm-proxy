"""场景 3: Streaming — 验证流式响应全链路

Responses API 流式 + Messages API 流式
"""

import json
import logging
from typing import Any

from tests.cli_env.lib import config_loader as cfg
from tests.cli_env.lib.client import send_responses, send_messages, configure_endpoint
from tests.cli_env.lib.mock_upstream import MockUpstream
from tests.cli_env.scenarios.scenario_base import Scenario, ScenarioReport

logger = logging.getLogger(__name__)


class StreamingScenario(Scenario):
    """流式场景：验证 SSE 流式响应全链路"""

    name = "streaming"

    async def setup(self) -> dict[str, Any]:
        # 1. Start mock upstream
        self.mock = MockUpstream()
        await self.mock.start()
        mock_url = self.mock.get_url()
        mock_port = self.mock.port

        chat_resp = self.mock.make_chat_response(
            text="This is a streaming response! " * 5,
            model="test-model",
        )
        self.mock.set_default_response(status=200, body=chat_resp)

        # 2. OpenAI-only upstream (proxy handles both Responses→Chat and Messages→Chat conversions)
        model = cfg.make_openai_model_config(
            api_base=f"http://127.0.0.1:{mock_port}",
            upstream_model="test-model",
            api_key="mock-key",
            display_name="Test Model",
        )
        config = cfg.create_scenario_config(models={"test-model": model})
        config_path = cfg.write_temp_config(config, prefix="streaming-")
        db_path = cfg.scenario_db_path("streaming", self.run_id)

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

    async def run(self, env: dict[str, Any]) -> Any:
        # Test 1: Responses API streaming
        logger.info("Testing Responses API streaming...")
        responses_stream = await send_responses(
            base_url=self.server.url,
            body={
                "model": "test-model",
                "input": "Tell me a short story.",
            },
            api_key="test-key",
            stream=True,
        )

        # Test 2: Messages API streaming
        logger.info("Testing Messages API streaming...")
        messages_stream = await send_messages(
            base_url=self.server.url,
            body={
                "model": "test-model",
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "Tell me a short story."}
                ],
            },
            api_key="test-key",
            stream=True,
        )

        return {
            "responses_stream": responses_stream,
            "messages_stream": messages_stream,
        }

    async def verify(self, response: Any) -> list:
        checks = []
        responses_data = response.get("responses_stream", (0, {}, [], 0))
        messages_data = response.get("messages_stream", (0, {}, [], 0))

        r_status, r_headers, r_events, r_elapsed = responses_data
        m_status, m_headers, m_events, m_elapsed = messages_data

        # === Responses API Streaming ===

        # Check R1: Status code
        checks.append(self.check(
            "responses_stream_status",
            passed=r_status == 200,
            expected="200",
            actual=str(r_status),
            detail="Responses API streaming endpoint",
        ))

        # Check R2: Has SSE events
        checks.append(self.check(
            "responses_stream_has_events",
            passed=len(r_events) > 0,
            expected="> 0 events",
            actual=f"{len(r_events)} events",
            detail="Responses API SSE events count",
        ))

        # Check R3: Has multiple events (> 10 for streaming content)
        checks.append(self.check(
            "responses_stream_multiple_events",
            passed=len(r_events) > 1,
            expected="> 1 event",
            actual=f"{len(r_events)} events",
        ))

        # Check R4: Has done event
        has_done = any(
            (evt == "[DONE]" if isinstance(evt, str) else
             (isinstance(evt, dict) and evt.get("type") == "done"))
            for evt in (r_events if isinstance(r_events, list) else [])
        )
        checks.append(self.check(
            "responses_stream_done_event",
            passed=has_done,
            expected="[DONE] or done event present",
            actual="done found" if has_done else "no done event",
        ))

        # === Messages API Streaming ===

        # Check M1: Status code
        checks.append(self.check(
            "messages_stream_status",
            passed=m_status == 200,
            expected="200",
            actual=str(m_status),
            detail="Messages API streaming endpoint",
        ))

        # Check M2: Has SSE events
        checks.append(self.check(
            "messages_stream_has_events",
            passed=len(m_events) > 0,
            expected="> 0 events",
            actual=f"{len(m_events)} events",
            detail="Messages API SSE events count",
        ))

        # Check M3: Has message_stop event
        has_stop = any(
            isinstance(evt, dict) and evt.get("event") == "message_stop"
            for evt in (m_events if isinstance(m_events, list) else [])
        )
        checks.append(self.check(
            "messages_stream_stop_event",
            passed=has_stop,
            expected="message_stop event present",
            actual="stop found" if has_stop else "no message_stop event",
        ))

        # Check M4: Has message_start event
        has_start = any(
            isinstance(evt, dict) and evt.get("event") == "message_start"
            for evt in (m_events if isinstance(m_events, list) else [])
        )
        checks.append(self.check(
            "messages_stream_start_event",
            passed=has_start,
            expected="message_start event present",
            actual="start found" if has_start else "no message_start event",
        ))

        # Check M5: No stream errors
        has_error = any(
            isinstance(evt, dict) and "error" in evt.get("event", "").lower()
            for evt in (m_events if isinstance(m_events, list) else [])
        ) or any(
            isinstance(evt, str) and "error" in evt.lower()
            for evt in (r_events if isinstance(r_events, list) else [])
        )
        checks.append(self.check(
            "no_stream_errors",
            passed=not has_error,
            expected="no stream errors",
            actual="error found" if has_error else "clean",
        ))

        return checks

    async def teardown(self):
        if hasattr(self, 'mock'):
            await self.mock.stop()
