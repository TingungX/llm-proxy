"""协议可达性表 + 上游选择算法

集中表达"哪些 (客户端协议, 上游协议) 组合是可达的"，避免在 handler 里散落
if/elif。同协议透传是隐含可达的，不写在这里。

新加转换器时只需：
  1. 在 protocol/<转换模块>/ 下实现 request/response/stream
  2. 在下方的 IMPLEMENTED_CONVERSIONS 里加一行
"""

from __future__ import annotations

# 协议别名规范化
# - "openai" 是历史别名，等价于 "openai/chat-completions"
# - 在新模型上推荐直接写 "openai/chat-completions" 或 "openai/responses"
_PROTOCOL_ALIASES: dict[str, str] = {
    "openai": "openai/chat-completions",
}

# 显式实现的跨协议转换对 (source_client_protocol, target_upstream_protocol)
# 顺序就是"多可达时的优先级"：第一个匹配的胜出
# 同协议透传是隐含的，不列在这里
IMPLEMENTED_CONVERSIONS: tuple[tuple[str, str], ...] = (
    ("anthropic", "openai/chat-completions"),
    ("openai/responses", "openai/chat-completions"),
    ("openai/chat-completions", "openai/responses"),
)


def normalize(protocol: str) -> str:
    """把协议别名规范化为标准形式（"openai" → "openai/chat-completions"）"""
    return _PROTOCOL_ALIASES.get(protocol, protocol)


def normalize_set(protocols) -> set[str]:
    """规范化集合中的所有协议"""
    return {normalize(p) for p in protocols}


def is_reachable(client: str, upstream: str) -> bool:
    """判断 (client → upstream) 是否可达（同协议或显式转换对）"""
    client_n = normalize(client)
    upstream_n = normalize(upstream)
    if client_n == upstream_n:
        return True
    return (client_n, upstream_n) in IMPLEMENTED_CONVERSIONS


class NoReachableProtocol(Exception):
    """当前模型对指定客户端协议没有可达的上游协议。

    携带完整诊断信息：admin 应能一眼看出哪里配错了。
    """

    def __init__(self, client: str, available: set[str]):
        self.client = normalize(client)
        self.available = normalize_set(available)
        # 列出所有已实现的跨协议对（含 client 出发的）
        all_known = {src for src, _ in IMPLEMENTED_CONVERSIONS}
        all_known.add("anthropic")
        all_known.add("openai/chat-completions")
        all_known.add("openai/responses")
        if self.client in self.available:
            self.available.discard(self.client)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [
            f"Model has no reachable upstream protocol for client={self.client!r}.",
            f"  Configured upstream_protocols: {sorted(self.available) or '(empty)'}",
            f"  Implemented conversions from {self.client!r}: "
            + (
                ", ".join(
                    f"{self.client} → {tgt}"
                    for src, tgt in IMPLEMENTED_CONVERSIONS
                    if src == self.client
                )
                or "(none)"
            ),
            f"  All implemented conversions: {list(IMPLEMENTED_CONVERSIONS)}",
            f"  Fix: add a protocol the model supports to upstream_protocols, "
            f"or extend IMPLEMENTED_CONVERSIONS to add a new converter.",
        ]
        return "\n".join(lines)


def select_upstream(client: str, available) -> str:
    """根据客户端协议和可用上游集合，选出最优上游协议。

    算法：
      1. 同协议优先（client ∈ available）：选 client（同协议透传）
      2. 否则按 IMPLEMENTED_CONVERSIONS 顺序，找第一个可达的
      3. 都不可达：抛 NoReachableProtocol

    Args:
        client: 客户端协议（"anthropic" | "openai/chat-completions" | "openai/responses"）
        available: 该模型支持的上游协议集合（set/iterable of str）

    Returns:
        选中的上游协议字符串
    """
    client_n = normalize(client)
    available_n = normalize_set(available) if not isinstance(available, set) else {
        normalize(p) for p in available
    }

    if not available_n:
        raise NoReachableProtocol(client_n, set())

    # 1. 同协议透传
    if client_n in available_n:
        return client_n

    # 2. 按已实现转换表顺序找第一个可达
    for src, tgt in IMPLEMENTED_CONVERSIONS:
        if src == client_n and tgt in available_n:
            return tgt

    # 3. 不可达
    raise NoReachableProtocol(client_n, available_n)

