import logging
import os
from contextvars import ContextVar

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

# Dev 环境标识：通过环境变量 LLM_PROXY_DEV=true 或端口非 4000 时自动判断
IS_DEV = os.environ.get("LLM_PROXY_DEV", "").lower() in ("1", "true", "yes")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")
        return True


def _resolve_level() -> int:
    """解析日志级别：环境变量 > 默认值（dev=DEBUG, prod=WARNING）"""
    env = os.environ.get("LLM_PROXY_LOG_LEVEL", "").upper()
    if env:
        return getattr(logging, env, logging.INFO)
    return logging.DEBUG if IS_DEV else logging.WARNING


def setup_logging(level: int | None = None) -> None:
    """配置日志。

    - level=None 时，根据 LLM_PROXY_LOG_LEVEL 环境变量或 IS_DEV 自动选择
    - dev 环境 (LLM_PROXY_DEV=true)：默认 DEBUG，保留全量日志
    - prod 环境 (Docker/main)：默认 WARNING，只输出警告和错误
    - 关键生命周期日志（启动/关闭/定时任务）始终通过 llm_proxy.lifecycle logger 输出
    """
    actual_level = level if level is not None else _resolve_level()

    fmt = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(actual_level)
    root.handlers.clear()
    root.addHandler(handler)

    # 生命周期日志始终 INFO 级别（启动/关闭/定时任务）
    logging.getLogger("llm_proxy.lifecycle").setLevel(logging.INFO)

    # 抑制第三方库噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
