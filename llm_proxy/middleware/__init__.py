"""全局异常兜底中间件"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def catch_all_exceptions(request: Request, call_next):
    """捕获所有未处理异常，返回结构化错误而非裸 500"""
    from llm_proxy.handlers.base import PipelineStop

    try:
        return await call_next(request)
    except PipelineStop as ps:
        return ps.response
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "proxy_error",
                    "message": "Internal proxy error",
                }
            },
        )
