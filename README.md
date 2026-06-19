# LLM Proxy

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Preact-10.22-673AB8?logo=preact" alt="Preact">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License AGPL-3.0">
</p>

**LLM Proxy** — 一个基于 FastAPI 的 LLM API 聚合网关，支持多模型、多协议的**双向格式转换**。你可以用任一种协议格式请求，代理自动转换为目标上游所需的协议格式。

目前支持的协议转换：

| 请求格式 | 可转换的上游格式 | 请求路由 |
| Anthropic Messages | Anthropic Messages / OpenAI Chat Completions | `/v1/messages` |
| OpenAI Chat Completions | OpenAI Chat Completions / Anthropic Messages | `/v1/chat/completions` |
| OpenAI Responses | OpenAI Chat Completions | `/v1/responses` |

→ **Anthropic 、 OpenAI Chat Completions 和 OpenAI Responses 三种请求格式均可由指定路径路由到任意格式的上游模型。**

---

## 目录

- [协议互转矩阵](#协议互转矩阵)
- [功能特性](#功能特性)
- [架构](#架构)
- [快速开始](#快速开始)
- [部署方式](#部署方式)
- [配置说明](#配置说明)
- [API 文档](#api-文档)
- [管理面板](#管理面板)
- [测试](#测试)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 协议互转矩阵

| 客户端格式 | 上游支持格式 | 路由入口 | 当前转换路径 |
|------------|------------|----------|----------|
| Anthropic Messages | Anthropic / Chat | `/v1/messages` | 同协议透传 / `anthropic_openai/` 旧通道 |
| OpenAI Chat Completions | Chat / Responses | `/v1/chat/completions` | 同协议透传 / `responses_chat/` 旧通道 |
| OpenAI Responses API | Responses / Chat / Anthropic | `/v1/responses` | **IR 通道**（`protocol/ir/`）|

所有三种请求格式均可路由到任意目标格式；目前 `/v1/responses` 已直接走 IR 抽象层（`IRProxyStep`），
其他两个路由仍在逐步迁移到 IR 通道。任意两个协议间的单次请求仅经过一次 IR 转换（`client → IR → upstream`），
不产生级联转换损耗。

> 迁移目标：把 `/v1/messages` 和 `/v1/chat/completions` 也接入 IRProxyStep，全量切换后删除
> `anthropic_openai/` 和 `responses_chat/` 旧通道。

---

## 功能特性

### Codex Desktop 深度兼容

LLM Proxy 全面兼容 Codex Desktop 的 OpenAI Responses API 通信协议：

- **apply_patch 透传 + DSL 修复**：`apply_patch` 直接降级为单个 function tool，
  上游返回的 `arguments` 原样作为 `custom_tool_call.input` 给 Codex。
  返回路径经过 [`repair_apply_patch_dsl`](llm_proxy/protocol/responses_chat/tool_replacement.py)
  修复常见的 DSL 格式问题（缺 `*** Begin Patch` / `*** End Patch`、`@@` hunk header
  不规范、`Add File` / `Update File` / `Delete File` / `Move to` 关键字大小写错误等）
- **apply_patch 工具描述注入**：服务端把工具描述替换为 `APPLY_PATCH_TOOL_DESCRIPTION`，
  显式列出 marker `***` 前缀、`@@` 单独成行、行前缀 ` ` / `- ` / `+ ` 规则、
  context 字符级匹配要求等，让模型在写之前就规避常见错误
- **非标准工具降级**：`namespace` 递归展开（子工具命名用 `__` 分隔，避开部分上游的
  `^[a-zA-Z0-9_-]+$` 限制）、`web_search` 客户端本地执行、其他 `custom` 工具透传
- **think 标签自动提取**：上游返回的 `<think>` 标签内容自动提取为 reasoning / thinking blocks
- **SSE 事件 type 安全**：每个 SSE data payload 自动注入 `type` 字段（与 `event:` 头镜像），
  Codex 只解析 data JSON 的 `type`，不认 `event:` 头；缺失会导致 "stream closed before
  response.completed" 错误
- **协议转换**：无论上游是 Chat Completions 还是 Anthropic Messages，自动完成双向转换

参考 [`config.toml.example`](config.toml.example) 快速配置 Codex Desktop 接入。

### 其他 AI 客户端

| 工具 | 协议 | 接入路径 |
|------|------|----------|
| Claude Code | Anthropic Messages | `/v1/messages` |
| OpenCode / 任何 OpenAI 兼容客户端 | OpenAI Chat Completions | `/v1/chat/completions` |
| 任何 OpenAI Responses 兼容客户端 | OpenAI Responses API | `/v1/responses` |

将工具的 `api_base` 指向本代理即可。

### 核心功能

| 功能 | 说明 |
|------|------|
| **全协议互转** | Anthropic / Chat / Responses 三种协议任意互转；`/v1/responses` 已走 IR 抽象层，其他两个路由由旧通道提供迁移期兼容 |
| **统一 IR 抽象层** | `protocol/ir/` 零外部依赖的中间表示层，ProtocolConverter 注册表模式，新增协议只需实现一个子类 |
| **旧通道迁移期运行** | `anthropic_openai/` 服务 Anthropic 跨协议；`responses_chat/` 服务 Chat→Responses 转换与 apply_patch DSL 修复 |
| **多上游聚合** | 一个代理接入 DeepSeek、MiniMax、GLM、OpenCode 等多个模型提供商 |
| **RTK 输入压缩** | 内置输入压缩工具（Rust Token Killer），压缩 tool_result 中的 CLI 输出噪声 |
| **端点认证与隔离** | 基于 API Key 的端点隔离，每个端点独立配置可用模型 |
| **模型路由与 Fallback** | 模型 family failover 链，429/503 自动切换 |
| **请求跟踪** | 每请求唯一 Request ID，结构化日志，Web 管理面板筛选查询 |
| **管理面板** | Preact + Vite 构建的 Web 控制台，管理端点/模型/用量/日志 |

---

## 架构

```
请求 (Anthropic / Chat / Responses)
  │
  ▼
Handler Pipeline (Auth → ModelResolve → ProtocolSelect → ... → Proxy)
  │
  ├── 同协议 ──→ 透传到上游
  │
  └── 跨协议 ──→ 协议转换层
                    │
                    ├── IR 通道（IRProxyStep）★ 当前 /v1/responses 在用
                    │   └── client_body → IRRequest → upstream_body
                    │   └── upstream SSE → IRStreamEvent → client SSE
                    │
                    └── 旧通道（ProxyStep）★ /v1/messages 和 /v1/chat/completions 迁移期使用
                        ├── anthropic_openai：Anthropic ↔ Chat
                        └── responses_chat：Chat → Responses（含 apply_patch DSL 修复）
```

协议转换层的核心是 `protocol/ir/`：

```
                    ┌─────────────┐
   Anthropic ──────→│             │──────→ Anthropic
   Chat     ──────→│  IR 抽象层  │──────→ Chat
   Responses──────→│ (dataclass) │──────→ Responses
                    └─────────────┘
```

转换路径：**请求方向** `client_body → to_ir() → IRRequest → to_upstream() → upstream_body`；**响应方向** `upstream_body → response_to_ir() → IRResponse → response_from_ir() → client_body`。IR 类型用 dataclass 定义，零外部依赖，协议特有字段通过 `extensions` dict 透传。

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 复制配置模板并填入你的 API Key
cp config.example.json config.json
# vi config.json

# 启动
bash start.sh
# 或手动启动
python -m uvicorn llm_proxy.main:app --port 4000
```

启动后访问 http://localhost:4000/static/ 进入管理面板。

### 使用示例

**Anthropic 格式（跨协议到 OpenAI Chat）：**
```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

**OpenAI Chat 格式：**
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

**OpenAI Responses 格式（跨协议到 Chat）：**
```bash
curl http://localhost:4000/v1/responses \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-pro", "input": "Hello", "max_output_tokens": 100}'
```

---

## 部署方式

### Docker（推荐）

```bash
# 构建前端
cd static && npm ci && npm run build && cd ..

# 启动
docker-compose up -d
```

### macOS launchd

编辑 `~/Library/LaunchAgents/com.llmproxy.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llmproxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>llm_proxy.main:app</string>
        <string>--port</string>
        <string>4000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/llm-proxy</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/path/to/llm-proxy/proxy.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/llm-proxy/proxy.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.llmproxy.plist
```

### Linux systemd

编辑 `/etc/systemd/system/llm-proxy.service`：

```ini
[Unit]
Description=LLM Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/llm-proxy
ExecStart=/usr/local/bin/python3 -m uvicorn llm_proxy.main:app --port 4000
Restart=always
RestartSec=5
User=your-user

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now llm-proxy
```

---

## 配置说明

模型通过 `config.json` 配置，每个模型包含以下字段：

| 字段 | 说明 |
|------|------|
| `api_base` | 上游 API 地址（不含 `/v1/...`） |
| `api_key` | 上游 API Key |
| `upstream_model` | 实际发给上游的模型名 |
| `upstream_protocol` | 标量字段（兼容保留）；建议用 `upstream_protocols` 数组 |
| `upstream_protocols` | 数组，如 `["anthropic", "openai"]`，协议选择根据可达性表自动决定 |
| `upstream_paths` | 各协议对应的上游路径（可选） |
| `vision_support` | 是否支持图片输入 |

端点（API Key 认证、模型白名单、family routing）通过管理面板或 API 配置，存储在 SQLite 中，支持运行态热更新。

---

## API 文档

### 代理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | Anthropic Messages 格式（同协议透传 / 跨协议转换） |
| POST | `/v1/chat/completions` | OpenAI Chat Completions 格式（同协议透传 / 跨协议转换） |
| POST | `/v1/responses` | OpenAI Responses 格式（含 apply_patch/namespace 工具转换） |
| POST | `/v1/messages/count_tokens` | Token 计数 |
| GET | `/v1/models` | 模型列表 |

### 管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/PUT | `/api/config` | 读写配置 |
| GET/POST/PUT/DEL | `/api/endpoints` | 端点 CRUD |
| GET | `/api/usage[?days=&group_by=&granularity=]` | 用量查询 |
| GET | `/api/logs/list` | 日志查询 |
| POST | `/api/latency` | 延迟测试 |
| POST | `/api/detect-protocol` | 检测上游协议 |

---

## 管理面板

基于 Preact + TypeScript + Vite 构建的 Web 控制台，功能包括：

- 端点 CRUD 管理
- 模型用量概览与热力图
- 请求日志筛选与查看
- 延迟测试
- 配置导入导出

### 前端开发

```bash
cd static
npm install
npm run dev      # 热重载开发
npm run build    # 生产构建
npm run test     # 测试
```

---

## 测试

```bash
# 全量测试
python -m pytest tests/ -v

# 仅 IR 抽象层测试
python -m pytest tests/test_ir_conversions.py tests/test_ir_streaming.py -v

# 冒烟测试
python tests/smoke_test.py
```

---

## 项目结构

```
llm-proxy/
├── llm_proxy/
│   ├── main.py                    # FastAPI 入口
│   ├── state.py                   # 运行态管理
│   ├── config_loader.py           # 配置加载
│   ├── logging_config.py          # 统一日志格式
│   ├── routes/                    # HTTP 路由（薄层）
│   ├── handlers/                  # Pipeline 处理管道
│   │   ├── base.py                # PipelineContext + HandlerStep + Pipeline
│   │   ├── *_handler.py           # 各路由 Pipeline 组装
│   │   └── shared/                # 可复用步骤
│   │       ├── auth.py
│   │       ├── model_resolve.py
│   │       ├── protocol_select.py
│   │       ├── proxy.py           # 旧 ProxyStep（引用旧通道）
│   │       └── ir_proxy.py        # 新 IRProxyStep（引用 IR 层）
│   ├── protocol/                  # 协议转换层
│   │   ├── capabilities.py        # 协议可达性表 + 上游选择算法
│   │   ├── ir/                    # ★ IR 抽象层（新增）
│   │   │   ├── __init__.py        #   ProtocolConverter 基类 + REGISTRY
│   │   │   ├── types.py           #   IRRequest/IRResponse/IRMessage/IRContentBlock
│   │   │   ├── _common.py         #   共享工具函数
│   │   │   ├── _stream.py         #   流式工具函数
│   │   │   ├── anthropic.py       #   Anthropic ↔ IR
│   │   │   ├── chat.py            #   Chat ↔ IR
│   │   │   └── responses.py       #   Responses ↔ IR
│   │   ├── anthropic_openai/      # Anthropic ↔ Chat（旧通道，保留兼容）
│   │   ├── responses_chat/        # Responses ↔ Chat（旧通道，保留兼容）
│   │   ├── errors.py              # 错误格式化
│   │   ├── sse.py                 # SSE 透传
│   │   └── think_tag.py           # Think 标签检测
│   ├── services/                  # 业务服务
│   │   ├── tool_call_fix.py       # Tool call 修复
│   │   ├── vision_service.py      # 图像→文本降级
│   │   └── input_compressor.py    # RTK 输入压缩
│   ├── infra/                     # 基础设施层
│   │   ├── db.py                  # SQLite 操作
│   │   ├── http_client.py         # 全局 HTTP 客户端
│   │   └── archive.py             # 用量记录
│   └── middleware/                # 中间件
│       ├── request_id.py
│       ├── access_log.py
│       └── catch_all_exceptions.py
├── static/                        # 前端（Preact + TypeScript + Vite）
├── tests/                         # 测试
├── docs/                          # 文档
├── config.example.json            # 配置模板
├── Dockerfile / docker-compose.yml
└── start.sh                       # 启动入口
```

---

## 许可证

**AGPL-3.0**

本软件使用 [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html) 发布。

简而言之：你可以自由使用、修改、分发本软件，但如果你将其用于商业用途（包括但不限于作为商业服务的后端组件），你必须将完整的源代码（包括你的修改和与之交互的完整系统）以同样的许可证开放给所有用户。
