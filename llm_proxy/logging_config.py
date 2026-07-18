import logging
import os
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

# Dev 环境标识：通过环境变量 LLM_PROXY_DEV=true 或端口非 4000 时自动判断
IS_DEV = os.environ.get("LLM_PROXY_DEV", "").lower() in ("1", "true", "yes")

# 日志文件唯一真相源：所有运行方式（dev.sh / start.sh / docker）统一写此路径。
# 可通过 LLM_PROXY_LOG_FILE 环境变量覆盖（Docker 中挂载到宿主机）。
DEFAULT_LOG_FILE = "logs/llm-proxy.log"
LOG_FILE = os.environ.get("LLM_PROXY_LOG_FILE", DEFAULT_LOG_FILE)

# 单文件上限 10MB，保留 5 份轮转，总计约 60MB
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


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


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_stream_handler() -> logging.Handler:
    """stdout/stderr handler——保留终端可见性（screen / docker logs）"""
    handler = logging.StreamHandler()
    handler.setFormatter(_build_formatter())
    handler.addFilter(RequestIdFilter())
    return handler


def _make_file_handler() -> logging.Handler | None:
    """文件 handler——日志唯一真相源。

    失败时不阻断启动（只 warn 到 stderr），因为日志是辅助设施，
    不应让一个 IO 问题把整个服务拉死。失败原因通常是权限/磁盘问题，需人工介入。
    """
    try:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(_build_formatter())
        handler.addFilter(RequestIdFilter())
        return handler
    except Exception as exc:
        # 不用 logging 报告——此时 logging 可能尚未完全就绪
        import sys
        print(f"[logging_config] WARNING: file handler init failed for {LOG_FILE!r}: {exc!r}", file=sys.stderr)
        return None


def setup_logging(level: int | None = None) -> None:
    """配置日志。

    - level=None 时，根据 LLM_PROXY_LOG_LEVEL 环境变量或 IS_DEV 自动选择
    - dev 环境 (LLM_PROXY_DEV=true)：默认 DEBUG，保留全量日志
    - prod 环境 (Docker/main)：默认 WARNING，只输出警告和错误
    - 关键生命周期日志（启动/关闭/定时任务）始终通过 llm_proxy.lifecycle logger 输出
    - 文件落盘：所有运行方式统一写 LLM_PROXY_LOG_FILE（默认 logs/llm-proxy.log）
    """
    actual_level = level if level is not None else _resolve_level()

    root = logging.getLogger()
    root.setLevel(actual_level)
    root.handlers.clear()

    root.addHandler(_make_stream_handler())

    file_handler = _make_file_handler()
    if file_handler is not None:
        root.addHandler(file_handler)

    # 生命周期日志始终 INFO 级别（启动/关闭/定时任务）
    logging.getLogger("llm_proxy.lifecycle").setLevel(logging.INFO)

    # 抑制第三方库噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger("llm_proxy.lifecycle").info(
        "logging initialized: level=%s, file=%s", logging.getLevelName(actual_level), LOG_FILE
    )
