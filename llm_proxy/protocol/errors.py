"""共享错误响应辅助函数"""
from fastapi.responses import JSONResponse


def make_anthropic_error(
    message: str,
    error_type: str = "invalid_request_error",
    status: int = 400,
) -> JSONResponse:
    """构建 Anthropic 格式的错误响应"""
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status_code=status,
    )


def make_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    status: int = 400,
    code: str | None = None,
) -> JSONResponse:
    """构建 OpenAI 格式的错误响应"""
    error_obj: dict = {"message": message, "type": error_type}
    if code is not None:
        error_obj["code"] = code
    return JSONResponse({"error": error_obj}, status_code=status)
