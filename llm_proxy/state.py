"""全局状态 — State 单例 + get_state() + init_state()"""

import asyncio
import logging
import hashlib
import re
from typing import Optional

from llm_proxy.config_loader import load_config
from llm_proxy.infra.http_client import get_client

logger = logging.getLogger(__name__)
lifecycle = logging.getLogger("llm_proxy.lifecycle")


def _get_enabled_protocols(model_cfg: dict) -> list[str]:
    """从模型配置提取启用的协议列表（字符串形式）。

    upstream_protocols = [{protocol: "anthropic", enabled: true, ...}, ...]
    """
    protocols = model_cfg.get("upstream_protocols", [])
    return [e["protocol"] for e in protocols if e.get("enabled", False)]


def _build_paths_from_entries(model_cfg: dict) -> dict[str, str]:
    """从 upstream_protocols 的 entry path 字段构建路径字典。

    {model_id(lower): {"anthropic/messages": "/v1/messages", ...}}
    只含 enabled 协议，无 enabled 的也包含在内（前端需要保留路径信息）。
    """
    result: dict[str, str] = {}
    for e in model_cfg.get("upstream_protocols", []):
        proto = e.get("protocol", "")
        path = (e.get("path") or "").strip()
        if proto == "anthropic":
            result["anthropic/messages"] = path or "/v1/messages"
        elif proto == "openai/chat-completions":
            result["openai/chat-completions"] = path or "/v1/chat/completions"
        elif proto == "openai/responses":
            result["openai/responses"] = path or "/v1/responses"
    return result


# ─── Resolver pure functions (from resolver.py) ────────────────────

def normalize_claude_model(model: str) -> Optional[str]:
    """归一化 Claude 模型名，保留主版本号以便精确路由
    
    返回值：
    - haiku/sonnet: 不带版本号（Anthropic 只有一个活跃版本）
    - opus-N-M: 带版本号（多版本共存）
    """
    m = model.lower().strip()
    m = re.sub(r"\[[\d.k]+\]", "", m).strip()
    if m.startswith("claude-haiku"):
        return "haiku"
    if m.startswith("claude-sonnet"):
        return "sonnet"
    if m.startswith("claude-opus"):
        ver = re.search(r"claude-opus-(\d+)-(\d+)", m)
        if ver:
            return f"opus-{ver.group(1)}-{ver.group(2)}"
        return "opus"
    return None


def _strip_date_suffix(model: str) -> str:
    """移除 Claude 模型名的日期后缀（如 -20251001），保留 claude-{family}-{ver} 格式
    
    claude-haiku-4-5-20251001 → claude-haiku-4-5
    claude-opus-4-8-20250528 → claude-opus-4-8
    claude-sonnet-4-6-20250514 → claude-sonnet-4-6
    """
    return re.sub(r"-\d{8,}$", "", model)


def _get_endpoint_id(api_key: str) -> str:
    """生成端点 ID（API Key 的 SHA256 前 16 位）"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def build_model_map(config: dict) -> dict:
    """从 config 构建 {模型名(小写): (api_base, api_key, upstream_model, upstream_protocol)}。

    从 `upstream_protocols` 取第一个启用的协议作为"代表协议"用于日志展示。
    """
    return {
        k.lower(): (
            v["api_base"],
            v["api_key"],
            v.get("upstream_model"),
            (_get_enabled_protocols(v) or [None])[0],
        )
        for k, v in config["models"].items()
    }


def _resolve_from_routing(
    family: str,
    routing: dict,
    model_map: dict,
) -> Optional[tuple[str, str, str, str, Optional[str], Optional[str]]]:
    """从 routing 配置解析目标模型"""
    entry = routing.get(family)
    if not entry:
        return None
    
    if isinstance(entry, str):
        target = entry
        failover = None
    else:
        target = entry.get("target")
        failover = entry.get("failover")
    
    if not target:
        return None
    
    target_lower = target.lower()
    if target_lower not in model_map:
        return None
    
    api_base, api_key, upstream, protocol = model_map[target_lower]
    return (api_base, api_key, upstream or target_lower, target, protocol, failover)


def resolve_model_for_endpoint(
    model: str,
    config: dict,
    model_map: dict,
    endpoint_family_routing: dict | None,
) -> Optional[tuple[str, str, str, str, Optional[str], Optional[str]]]:
    """解析模型名，使用端点级映射
    
    Args:
        model: 客户端发来的模型名（如 "claude-opus-4-7"）
        config: 完整配置字典
        model_map: build_model_map(config) 的结果
        endpoint_family_routing: 端点的 family_routing 配置（可为 None）
    
    Returns:
        六元组或 None:
        - api_base: 上游 API 地址
        - api_key: 上游 API 密钥
        - upstream_model: 实际发给上游的模型名
        - config_key: config.json 中的模型 ID
        - upstream_protocol: 上游协议
        - failover_family: 下一个 failover family（可为 None）
    """
    m = model.lower().strip()
    m = re.sub(r"\[[\d.k]+\]", "", m).strip()
    
    routing = endpoint_family_routing or {}
    
    if routing:
        if m in routing:
            return _resolve_from_routing(m, routing, model_map)
        stripped = _strip_date_suffix(m)
        if stripped and stripped in routing:
            return _resolve_from_routing(stripped, routing, model_map)
        if not m.startswith("claude-") and (m.startswith("opus") or m.startswith("sonnet") or m.startswith("haiku")):
            prefixed = f"claude-{m}"
            if prefixed in routing:
                return _resolve_from_routing(prefixed, routing, model_map)
            prefixed_stripped = _strip_date_suffix(prefixed)
            if prefixed_stripped and prefixed_stripped in routing:
                return _resolve_from_routing(prefixed_stripped, routing, model_map)
        family = normalize_claude_model(m)
        if family:
            if family in routing:
                return _resolve_from_routing(family, routing, model_map)
            generic = family.split("-")[0]
            if generic != family and generic in routing:
                return _resolve_from_routing(generic, routing, model_map)
    
    if m in model_map:
        api_base, api_key, upstream, protocol = model_map[m]
        return (api_base, api_key, upstream or m, m, protocol, None)
    
    return None


def resolve_model(
    model: str,
    config: dict,
    model_map: dict,
) -> Optional[tuple[str, str, str, str, Optional[str]]]:
    """解析模型名 → (api_base, api_key, upstream_model, config_key, upstream_protocol) 或 None
    
    保留此函数用于向后兼容，但不再使用全局 family_routing。
    """
    result = resolve_model_for_endpoint(model, config, model_map, None)
    if result:
        return result[:5]
    return None


# ─── State class ──────────────────────────────────────────────────

class State:
    """应用运行时状态 — 单例，通过 get_state() 获取"""

    def __init__(self, config: dict):
        self.config = config
        self.model_map = build_model_map(config)
        self.vision_map = self._build_vision_map(config)
        self.protocols_map = self._build_protocols_map(config)
        self.paths_map = self._build_paths_map(config)
        self.allow_proxy_map = self._build_allow_proxy_map(config)
        self.compression_config = config.get("compression", {})

    def resolve_model(self, model: str, endpoint_family_routing: dict | None = None):
        return resolve_model_for_endpoint(
            model, self.config, self.model_map, endpoint_family_routing
        )

    async def reload(self):
        self.config = load_config()
        self.model_map = build_model_map(self.config)
        self.vision_map = self._build_vision_map(self.config)
        self.protocols_map = self._build_protocols_map(self.config)
        self.paths_map = self._build_paths_map(self.config)
        self.allow_proxy_map = self._build_allow_proxy_map(self.config)
        self.compression_config = self.config.get("compression", {})

    @staticmethod
    def _build_vision_map(cfg: dict) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for k, v in cfg.get("models", {}).items():
            result[k.lower()] = v.get("vision_support", False)
        return result

    @staticmethod
    def _build_protocols_map(cfg: dict) -> dict[str, set[str]]:
        """构建 {model_id(lower): {启用的协议集合}}。"""
        result: dict[str, set[str]] = {}
        for key, val in cfg.get("models", {}).items():
            enabled = _get_enabled_protocols(val)
            if enabled:
                result[key.lower()] = set(enabled)
        return result

    @staticmethod
    def _build_paths_map(cfg: dict) -> dict[str, dict[str, str]]:
        """从 config 构建 {model_id(lower): {协议路径字典}}。

        从 upstream_protocols 的 entry.path 读取。
        """
        result: dict[str, dict[str, str]] = {}
        for key, val in cfg.get("models", {}).items():
            paths = _build_paths_from_entries(val)
            if paths:
                result[key.lower()] = paths
        return result

    @staticmethod
    def _build_allow_proxy_map(cfg: dict) -> dict[str, bool]:
        return {
            k.lower(): v.get("allow_proxy", False)
            for k, v in cfg.get("models", {}).items()
        }


# ─── Singleton access ─────────────────────────────────────────────

_instance: State | None = None


def init_state(config: dict | None = None) -> State:
    global _instance
    cfg = config or load_config()
    models_count = len(cfg.get("models", {}))
    compression_enabled = cfg.get("compression", {}).get("enabled", False)
    error_handling = cfg.get("error_handling", {})
    lifecycle.info(
        "Config loaded: %d models, compression=%s, failover=%s",
        models_count,
        compression_enabled,
        error_handling.get("failover_enabled", False),
    )
    _instance = State(cfg)
    return _instance


def get_state() -> State:
    global _instance
    if _instance is None:
        cfg = load_config()
        models_count = len(cfg.get("models", {}))
        lifecycle.info("Lazy-init state: %d models from config", models_count)
        _instance = State(cfg)
    return _instance


# ─── Omlx health check ───────────────────────────────────────────

async def _check_omlx_health():
    try:
        client = get_client()
        resp = await client.get("http://localhost:8000/v1/models", timeout=5.0)
        if resp.status_code < 500:
            logger.info("Omlx vision service is available")
        else:
            logger.warning("Omlx vision service returned %s", resp.status_code)
    except Exception as e:
        logger.warning("Omlx vision service is not available: %s", e)
        logger.warning(
            "Models with vision_support=false will degrade gracefully, "
            "but image-to-text fallback won't work until Omlx is running."
        )

try:
    asyncio.get_running_loop()
except RuntimeError:
    pass
else:
    asyncio.create_task(_check_omlx_health())
