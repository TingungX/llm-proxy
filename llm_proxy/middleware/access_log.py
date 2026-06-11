import time
import logging

from starlette.requests import Request
from starlette.responses import Response

from llm_proxy.handlers.base import PipelineStop

logger = logging.getLogger("llm_proxy.access")


async def access_log_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except PipelineStop as ps:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %s %.1fms",
            request.method,
            request.url.path,
            ps.response.status_code,
            duration_ms,
        )
        raise
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s 500 %.1fms",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response
