"""LLM Proxy 入口 — 向后兼容，实际逻辑已迁移到 llm_proxy/"""

from llm_proxy.main import app  # noqa: F401

# 所有路由已迁移到 llm_proxy/routes/，需在此导入以注册
import llm_proxy.routes.misc       # noqa: F401
import llm_proxy.routes.messages   # noqa: F401
import llm_proxy.routes.config     # noqa: F401
import llm_proxy.routes.usage      # noqa: F401
import llm_proxy.routes.endpoints  # noqa: F401
import llm_proxy.routes.latency    # noqa: F401
