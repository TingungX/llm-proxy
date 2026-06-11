"""模型解析步骤 — 解析模型名，校验端点权限"""

import logging

from llm_proxy.handlers.base import PipelineContext, HandlerStep, PipelineStop
from llm_proxy.state import get_state, resolve_model_for_endpoint
from llm_proxy.protocol.errors import make_anthropic_error, make_openai_error

logger = logging.getLogger(__name__)


def _make_error(ctx: PipelineContext, message: str, error_type: str, status_code: int):
    if ctx.error_protocol == "anthropic":
        return make_anthropic_error(message, error_type, status_code)
    return make_openai_error(message, error_type, status_code)


class ModelResolveStep(HandlerStep):
    """模型解析步骤：解析模型名 → 校验 API Key → 校验端点权限"""

    async def execute(self, ctx: PipelineContext) -> None:
        raw_model = ctx.body.get("model", "")
        logger.info(f"Received request for model: {raw_model}")

        s = get_state()
        endpoint = ctx.endpoint
        endpoint_family_routing = endpoint.get("family_routing")

        resolved = resolve_model_for_endpoint(
            raw_model, s.config, s.model_map, endpoint_family_routing
        )
        if not resolved:
            logger.error(f"Unknown model: {raw_model}")
            raise PipelineStop(_make_error(ctx, f"Unknown model: {raw_model}", "invalid_request_error", 400))

        api_base, api_key, actual_model, model_id, upstream_protocol, failover_family = resolved
        logger.info(f"Resolved: raw={raw_model} -> actual={actual_model} config_key={model_id} protocol={upstream_protocol}")

        if not api_key:
            logger.error(f"Model {model_id} has no API key configured")
            raise PipelineStop(_make_error(
                ctx,
                f"Model {model_id} has no API key configured. Set it in the model settings.",
                "invalid_request_error",
                400,
            ))

        allowed = [m.lower() for m in endpoint.get("models", [])]
        if allowed and model_id.lower() not in allowed:
            logger.warning(f"Endpoint {endpoint['endpoint_id']} not allowed: {model_id}")
            raise PipelineStop(_make_error(
                ctx,
                f"Model {raw_model} not available for this endpoint",
                "invalid_request_error",
                403,
            ))

        ctx.resolved = resolved

