import uuid

from starlette.requests import Request
from starlette.responses import Response

from llm_proxy.logging_config import REQUEST_ID_CTX


async def request_id_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    REQUEST_ID_CTX.set(request_id)
    try:
        response: Response = await call_next(request)
    finally:
        REQUEST_ID_CTX.set("-")
    response.headers["x-request-id"] = request_id
    return response
