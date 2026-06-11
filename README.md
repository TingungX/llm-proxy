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
|----------|------------------|----------|
| Anthropic Messages | Anthropic Messages / OpenAI Chat Completions | `/v1/messages` |
| OpenAI Chat Completions | OpenAI Chat Completions / Anthropic Messages | `/v1/chat/completions` |
| OpenAI Responses | OpenAI Chat Completions | `/v1/responses` |

→ **Anthropic , OpenAI Chat Completions 和 OpenAI Responses 三种请求格式均可由指定路径路由到任意格式的上游模型。**

---

## 目录

- [LLM Proxy](#llm-proxy)
  - [目录](#目录)
  - [功能特性](#功能特性)
    - [基本全面支持 Codex Desktop](#基本全面支持-codex-desktop)
    - [其他 AI 客户端](#其他-ai-客户端)
    - [核心功能](#核心功能)
  - [快速开始](#快速开始)
    - [使用示例](#使用示例)
  - [部署方式](#部署方式)
    - [Docker（推荐）](#docker推荐)
    - [macOS launchd](#macos-launchd)
    - [Linux systemd](#linux-systemd)
  - [配置说明](#配置说明)
  - [API 文档](#api-文档)
    - [代理 API](#代理-api)
    - [管理 API](#管理-api)
  - [管理面板](#管理面板)
    - [前端开发](#前端开发)
  - [测试](#测试)
  - [项目结构](#项目结构)
  - [许可证](#许可证)

---

## 功能特性

### 基本全面支持 Codex Desktop

LLM Proxy 深度兼容 Codex Desktop 通信协议（OpenAI Responses API），并提供完整的工具转换支持：

Codex Desktop 使用 **OpenAI Responses API** 协议，LLM Proxy 在 `/v1/responses` 路径自动完成 Responses → Chat Completions 格式转换，并完整支持 apply_patch、namespace、web_search 等工具转换。

参考 [`config.toml.example`](config.toml.example) 快速配置 Codex Desktop 接入本代理。

### 其他 AI 客户端

以下 AI 编程工具同样兼容，直接接入即可使用：

| 工具 | 使用协议 | 接入路径 |
|------|----------|----------|
| **Claude Code** | Anthropic Messages API | `/v1/messages` |
| **OpenCode** | OpenAI Chat Completions | `/v1/chat/completions` |
| 任何 OpenAI 兼容客户端 | OpenAI Chat Completions | `/v1/chat/completions` |

将工具的 `api_base` 指向本代理即可。**一个代理同时满足所有工具的后端模型接入需求。**

### 核心功能

| 功能 | 说明 |
|------|------|
| **多协议双向转换** | Anthropic Messages ↔ OpenAI Chat Completions 全双向、OpenAI Responses → Chat Completions |
| **多上游聚合** | 一个代理接入 DeepSeek、MiniMax、GLM、OpenCode 等多个模型提供商 |
| **RTK（输入压缩 / Beta）** | 内置输入压缩工具（Rust Token Killer），压缩 tool_result 中的 CLI 输出噪声、截断长代码块、折叠空行以节省 input token；默认关闭，可在端点配置编辑中手动启用（启用后可能降低模型表现） |
| **端点认证与隔离** | 基于 API Key 的端点隔离，每个端点独立配置可用模型 |
| **模型路由与 Fallback** | 开启后，模型 family failover 链，429/503 自动切换 |
| **请求跟踪** | 每请求唯一 Request ID，结构化日志，Web 管理面板筛选查询 |
| **工具格式兼容** | namespace、apply_patch 等非标准工具类型自动转换 |
| **管理面板** | Preact + Vite 构建的 Web 控制台，管理端点/模型/用量/日志 |

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

**Anthropic 格式：**
```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

**OpenAI 格式：**
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
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

Docker 内置健康检查（每 30s 通过 `GET /api/config` 验证），开箱即用。

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
| `upstream_protocol` | `anthropic` 或 `openai`，为空时自动探测 |
| `upstream_paths` | 各协议对应的上游路径（可选） |

端点（API Key 认证、模型白名单、family routing）通过管理面板或 API 配置，存储在 SQLite 中，支持运行态热更新。

---

## API 文档

### 代理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | Anthropic Messages 格式 |
| POST | `/v1/chat/completions` | OpenAI Chat Completions 格式 |
| POST | `/v1/responses` | OpenAI Responses 格式（转为 Chat Completions） |

所有请求需携带 `x-api-key` 或 `Authorization: Bearer` 头进行认证。

### 管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/PUT | `/api/config` | 读写配置 |
| GET/POST/PUT/DEL | `/api/endpoints` | 端点 CRUD |
| GET | `/api/usage[?days=&group_by=&granularity=]` | 用量查询 |
| GET | `/api/logs` | 日志查询 |
| POST | `/api/latency` | 延迟测试 |

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
python -m pytest tests/ -v
python tests/smoke_test.py
```

---

## 项目结构

```
llm-proxy/
├── llm_proxy/           # Python 后端
│   ├── main.py          # FastAPI 入口
│   ├── state.py         # 运行态管理
│   ├── config_loader.py # 配置加载
│   ├── routes/          # HTTP 路由（薄层）
│   ├── handlers/        # Pipeline 处理管道
│   ├── protocol/        # 协议双向转换（Python 实现）
│   ├── services/        # 业务服务
│   ├── infra/           # 基础设施（SQLite/HTTP 客户端）
│   └── middleware/      # 中间件（request_id/access_log）
├── static/              # 前端（Preact + TypeScript + Vite）
├── tests/               # 测试
├── docs/                # 文档
├── config.example.json  # 配置模板
├── Dockerfile
├── docker-compose.yml
└── start.sh / proxy.py  # 启动入口
```

---

## 许可证

**AGPL-3.0**

本软件使用 [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html) 发布。

简而言之：你可以自由使用、修改、分发本软件，但**如果你将其用于商业用途（包括但不限于作为商业服务的后端组件），你必须将完整的源代码（包括你的修改和与之交互的完整系统）以同样的许可证开放给所有用户。**
