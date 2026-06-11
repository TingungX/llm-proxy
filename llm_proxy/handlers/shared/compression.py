"""输入压缩 Pipeline 步骤 — 在转发前压缩请求体中的冗余内容以节省 input token。

借鉴 RTK（Rust Token Killer）的思路，压缩 tool_result 中的 CLI 输出噪声、
截断长代码块、折叠空行、缩短绝对路径。

插入位置：VisionFallbackStep 之后、ProxyStep 之前（或 ResponsesConvertStep 之前）。
"""

import logging

from llm_proxy.handlers.base import HandlerStep, PipelineContext
from llm_proxy.services.input_compressor import CompressionConfig, InputCompressor
from llm_proxy.state import get_state

logger = logging.getLogger(__name__)

# 默认启用时的策略列表
_DEFAULT_STRATEGIES = ["drop_progress", "truncate", "collapse", "shorten_paths"]


class CompressionStep(HandlerStep):
    """输入压缩步骤 — 压缩请求体中的冗余内容以节省 input token"""

    async def execute(self, ctx: PipelineContext) -> None:
        config = self._build_config(ctx)
        if not config.enabled:
            return

        compressor = InputCompressor(config)

        if ctx.error_protocol == "anthropic":
            compressor.compress_anthropic_body(ctx.body)
        elif "input" in ctx.body and isinstance(ctx.body.get("input"), list):
            # Responses 格式（有 input 数组字段）
            compressor.compress_responses_body(ctx.body)
        else:
            # Chat 格式
            compressor.compress_chat_body(ctx.body)

        if compressor.stats.items_compressed > 0:
            logger.info(
                "Input compressed: %d items, %d→%d chars",
                compressor.stats.items_compressed,
                compressor.stats.original_chars,
                compressor.stats.compressed_chars,
            )

    def _build_config(self, ctx: PipelineContext) -> CompressionConfig:
        """从全局配置 + 端点设置合并压缩配置"""
        state = get_state()
        global_cfg = getattr(state, 'compression_config', {}) or {}

        # 端点级覆盖
        endpoint = ctx.endpoint or {}
        endpoint_settings = endpoint.get("settings", {})
        if isinstance(endpoint_settings, str):
            # settings 可能是 JSON 字符串
            import json
            try:
                endpoint_settings = json.loads(endpoint_settings)
            except (json.JSONDecodeError, TypeError):
                endpoint_settings = {}
        endpoint_cfg = endpoint_settings.get("compression", {}) if isinstance(endpoint_settings, dict) else {}

        # 合并：端点覆盖全局
        merged = {**global_cfg, **endpoint_cfg}

        # 处理 strategies：如果未指定但 enabled=True，使用默认列表
        if merged.get("enabled") and "strategies" not in merged:
            merged["strategies"] = _DEFAULT_STRATEGIES

        return CompressionConfig(**merged)
