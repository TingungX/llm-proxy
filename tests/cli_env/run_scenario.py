#!/usr/bin/env python3
"""CLI 调试环境入口

用法:
    python tests/cli-env/run_scenario.py basic_proxy
    python tests/cli-env/run_scenario.py --all
    python tests/cli-env/run_scenario.py streaming -p 14010

可用场景:
    basic_proxy          — Responses API → proxy → OpenAI 上游
    protocol_conversion  — Messages API → proxy → OpenAI 上游
    streaming            — 流式响应全链路
    auth_resolution      — 鉴权 + 模型解析
    edge_cases           — 错误处理 / 压缩 / failover
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# 确保项目根在 sys.path 中
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


# 场景注册表
SCENARIOS = {}


def _register_scenarios():
    """延迟导入注册场景"""
    from tests.cli_env.scenarios.basic_proxy import BasicProxyScenario
    from tests.cli_env.scenarios.protocol_conversion import ProtocolConversionScenario
    from tests.cli_env.scenarios.streaming import StreamingScenario
    from tests.cli_env.scenarios.auth_resolution import AuthResolutionScenario
    from tests.cli_env.scenarios.edge_cases import EdgeCasesScenario

    SCENARIOS["basic_proxy"] = BasicProxyScenario
    SCENARIOS["protocol_conversion"] = ProtocolConversionScenario
    SCENARIOS["streaming"] = StreamingScenario
    SCENARIOS["auth_resolution"] = AuthResolutionScenario
    SCENARIOS["edge_cases"] = EdgeCasesScenario


def _format_report(report) -> str:
    """格式化输出报告"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Scenario: {report.scenario}")
    lines.append(f"Status:   {report.status}")
    lines.append(f"Duration: {report.duration_ms:.1f}ms")
    lines.append("-" * 60)

    if report.checks:
        lines.append(f"{'CHECK':<40} {'EXPECTED':<20} {'ACTUAL':<20} {'RESULT'}")
        lines.append("-" * 100)
        for c in report.checks:
            result = "✓" if c.passed else "✗"
            lines.append(f"{c.name:<40} {c.expected:<20} {c.actual:<20} {result}")
            if c.detail:
                lines.append(f"  └ {c.detail}")

    if report.server_log:
        lines.append("-" * 60)
        lines.append("Server Errors:")
        lines.append(report.server_log)

    if report.error:
        lines.append("-" * 60)
        lines.append(f"Error: {report.error}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _format_json_report(report) -> str:
    """JSON 格式报告"""
    return json.dumps({
        "scenario": report.scenario,
        "status": report.status,
        "checks": [
            {"name": c.name, "expected": c.expected, "actual": c.actual,
             "passed": c.passed, "detail": c.detail}
            for c in report.checks
        ],
        "server_log": report.server_log,
        "response": report.response,
        "duration_ms": round(report.duration_ms, 1),
        "error": report.error,
    }, indent=2, ensure_ascii=False, default=str)


async def run_scenario(name: str, port: int = 0) -> bool:
    """运行单个场景"""
    _register_scenarios()

    if name not in SCENARIOS:
        print(f"Unknown scenario: {name}")
        print(f"Available: {', '.join(sorted(SCENARIOS.keys()))}")
        return False

    scenario_class = SCENARIOS[name]
    if port > 0:
        scenario = scenario_class()
        scenario.server._port = port
    else:
        scenario = scenario_class()

    print(f"\nRunning scenario: {name}...")
    report = await scenario.run_all()
    print(_format_report(report))
    print()

    return report.status == "PASS"


async def run_all_scenarios(port: int = 0) -> dict[str, bool]:
    """运行所有场景"""
    _register_scenarios()
    results = {}

    for name in sorted(SCENARIOS.keys()):
        ok = await run_scenario(name, port=port)
        results[name] = ok

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in sorted(results.items()):
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")
    print(f"\n{passed}/{total} scenarios passed")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLI 调试环境 — llm-proxy 场景化集成测试",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help="场景名称（不指定则列出可用场景）",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="运行所有场景",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=0,
        help="llm-proxy 端口（0=随机分配）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 格式输出",
    )

    args = parser.parse_args()
    _register_scenarios()

    if not args.scenario and not args.all:
        print("可用场景:")
        for name in sorted(SCENARIOS.keys()):
            print(f"  {name}")
        print("\n用法: python tests/cli-env/run_scenario.py <scenario>")
        print("      python tests/cli-env/run_scenario.py --all")
        return 0

    if args.all:
        results = asyncio.run(run_all_scenarios(port=args.port))
        failed = [name for name, ok in results.items() if not ok]
        return 1 if failed else 0
    else:
        ok = asyncio.run(run_scenario(args.scenario, port=args.port))
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
