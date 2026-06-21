# CLI 调试环境

场景化集成测试框架，用于验证 llm-proxy 在处理真实 CLI 线缆格式请求时的完整行为。

## 设计思路

不模拟 CLI（不 spawn Codex CLI / Claude Code CLI 子进程），而是发送**与 CLI 完全相同的 HTTP 请求**（同协议、同路径、同格式），走完整的 llm-proxy Pipeline。

每个场景：
1. **setup**: 启动 mock upstream + llm-proxy（隔离端口，自定义配置）
2. **run**: 发送 wire-format 请求（Responses API / Messages API）
3. **verify**: 验证响应结构、状态码、日志
4. **teardown**: 清理进程和临时文件
5. **report**: 输出结构化结果

## 场景列表

| 场景 | 描述 | CLI 线路 |
|------|------|----------|
| `basic_proxy` | Responses API → proxy → OpenAI 上游 | Codex CLI |
| `protocol_conversion` | Messages API → proxy → OpenAI 上游 | Claude Code CLI |
| `streaming` | 流式响应（Responses + Messages 双线） | 两者 |
| `auth_resolution` | 鉴权 + 模型解析（6 种组合） | 通用 |
| `edge_cases` | 错误处理 / 压缩 / failover | 通用 |

## 使用方法

```bash
# 运行单个场景
python tests/cli-env/run_scenario.py basic_proxy

# 指定端口（避免冲突）
python tests/cli-env/run_scenario.py streaming -p 14010

# 运行所有场景
python tests/cli-env/run_scenario.py --all

# JSON 格式输出（供程序化处理）
python tests/cli-env/run_scenario.py basic_proxy --json
```

## 新增场景

1. 在 `scenarios/` 下创建 `.py` 文件
2. 继承 `Scenario` 基类，实现 `setup()` / `run()` / `verify()` / `teardown()`
3. 在 `run_scenario.py` 的 `_register_scenarios()` 中注册

## 架构

```
tests/cli-env/
  run_scenario.py            # 入口
  scenarios/
    scenario_base.py         # Scenario 基类
    basic_proxy.py           # 场景 1
    protocol_conversion.py   # 场景 2
    streaming.py             # 场景 3
    auth_resolution.py       # 场景 4
    edge_cases.py            # 场景 5
  lib/
    mock_upstream.py         # Mock LLM API 上游服务器
    config_loader.py         # 场景配置生成
    server.py                # llm-proxy 进程管理
    client.py                # Wire-format HTTP 客户端
    log_collector.py         # 日志捕获与过滤
```

## 环境依赖

- Python 3.10+
- httpx
- uvicorn（llm-proxy 自带）
- 项目 `.venv` 中的依赖

## 注意事项

- 每个场景使用**独立端口**和**独立临时配置**，互不干扰
- mock upstream 返回**可预测的固定响应**，测试结果确定性高
- server stderr 日志被实时捕获，可通过 `LogCollector` 按 request_id 过滤分析
- 临时配置文件存放在系统临时目录，测试结束后清理
