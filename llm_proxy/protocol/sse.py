"""SSE 流式协议透传辅助"""
import json
import logging

logger = logging.getLogger(__name__)


async def stream_response(resp, on_chunk=None, on_event=None):
    """安全地转发上游 SSE 流
    
    on_event(event_type, data): 每个完整 SSE 事件解析后回调
    """
    current_event = ""
    try:
        async for line in resp.aiter_lines():
            chunk = (line + "\n").encode()
            if on_chunk:
                on_chunk(chunk)

            if on_event:
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and current_event:
                    try:
                        data = json.loads(line[6:])
                        on_event(current_event, data)
                    except json.JSONDecodeError:
                        pass
                    current_event = ""

            yield chunk
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        error_chunk = (
            b'event: error\n'
            b'data: {"error": {"type": "proxy_error", "message": "Stream interrupted"}}\n\n'
        )
        yield error_chunk
    finally:
        pass
