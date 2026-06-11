"""冒烟测试 — 启动服务 → 发请求 → 验证 200 → 退出

用法: python3 tests/smoke_test.py
退出码: 0 = 通过, 1 = 失败
"""

import sys
import time
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
HOST = "http://127.0.0.1:4000"


def check_endpoint(url: str, label: str, headers: dict = None) -> bool:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status == 200:
            print(f"  ✓ {label} ({resp.status})")
            return True
        print(f"  ✗ {label} ({resp.status})")
        return False
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return False


def main():
    print("=== LLM Proxy Smoke Test ===\n")

    # 1. 启动服务
    print("Starting server...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "llm_proxy.main:app",
         "--host", "0.0.0.0", "--port", "4000"],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    try:
        # 2. 测试各端点
        passed = True

        passed &= check_endpoint(f"{HOST}/health", "/health")
        passed &= check_endpoint(f"{HOST}/api/config", "/api/config")
        passed &= check_endpoint(f"{HOST}/api/endpoints", "/api/endpoints")
        passed &= check_endpoint(f"{HOST}/api/usage/summary", "/api/usage/summary")
        # /v1/models 需要 API Key，测试 401 响应
        try:
            req = urllib.request.Request(f"{HOST}/v1/models")
            resp = urllib.request.urlopen(req, timeout=10)
            print(f"  ✗ /v1/models: expected 401, got {resp.status}")
            passed = False
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"  ✓ /v1/models ({e.code} - API Key required)")
            else:
                print(f"  ✗ /v1/models: expected 401, got {e.code}")
                passed = False
        passed &= check_endpoint(f"{HOST}/", "首页")

        # 测 /v1/messages 的 400 路径（未知模型）
        try:
            import json
            data = json.dumps({"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
            req = urllib.request.Request(
                f"{HOST}/v1/messages",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            body = json.loads(resp.read())
            if resp.status == 401 and "error" in body:
                print(f"  ✓ /v1/messages 错误处理 ({resp.status})")
            else:
                print(f"  ✗ /v1/messages 错误处理: unexpected {resp.status}")
                passed = False
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            if e.code == 401 and "error" in body:
                print(f"  ✓ /v1/messages 错误处理 ({e.code} - API Key required)")
            elif e.code == 400 and "error" in body:
                print(f"  ✓ /v1/messages 错误处理 ({e.code})")
            else:
                print(f"  ✗ /v1/messages 错误处理: unexpected {e.code}")
                passed = False

        print()
        if passed:
            print("PASS — 所有冒烟测试通过")
            return 0
        else:
            print("FAIL — 有测试未通过")
            return 1

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
