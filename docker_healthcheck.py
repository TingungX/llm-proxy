"""Docker health check script — 比内联 python -c 更详细的错误日志。

健康检查逻辑：
1. GET http://localhost:4000/api/config
2. 成功 → exit(0)
3. 失败 → 打印 error 到 stderr → exit(1)

stderr 输出的内容会出现在 `docker logs` 中。
"""

import sys
import json

try:
    import httpx
except ImportError:
    print("HEALTHCHECK FAILED: httpx not installed", file=sys.stderr)
    sys.exit(1)

try:
    resp = httpx.get("http://localhost:4000/api/config", timeout=5.0)
    resp.raise_for_status()
    print("HEALTHCHECK OK", file=sys.stderr)
    sys.exit(0)
except httpx.ConnectError:
    print("HEALTHCHECK FAILED: Cannot connect to http://localhost:4000", file=sys.stderr)
    sys.exit(1)
except httpx.TimeoutException:
    print("HEALTHCHECK FAILED: Request timed out after 5s", file=sys.stderr)
    sys.exit(1)
except httpx.HTTPStatusError as e:
    status = e.response.status_code
    try:
        detail = e.response.json()
        print(
            f"HEALTHCHECK FAILED: HTTP {status} — {json.dumps(detail)[:200]}",
            file=sys.stderr,
        )
    except Exception:
        body = e.response.text[:200]
        print(
            f"HEALTHCHECK FAILED: HTTP {status} — {body}",
            file=sys.stderr,
        )
    sys.exit(1)
except Exception as e:
    print(f"HEALTHCHECK FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

