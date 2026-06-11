import logging
from contextvars import ContextVar

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")
        return True


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
