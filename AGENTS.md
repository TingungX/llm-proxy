# LLM Proxy AGENTS.md

面向任何接手此仓库的开发者。先读此文，再改代码。

## 项目是什么

LLM Proxy 是一个**基于 FastAPI 的 LLM API 聚合网关**。它接收 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses 三种客户端格式的请求，路由到任意格式的上游模型，并做双向协议转换。

## 一句话数据流

```
Client → FastAPI → Middlewares (request_id → access_log → catch_all_exceptions)
                   → Routes (薄委托层)
                   → Handlers (Pipeline 模式: Auth → ModelResolve → ... → Proxy)
                   → Upstream LLM API
```

## 目录结构

```
llm-proxy/
  llm_proxy/                    主 Python 包
    main.py                     FastAPI app 创建 + lifespan + 路由注册 + 静态文件挂载
    state.py                    State 单例 + 模型解析函数（get_state / init_state）
    config_loader.py            config.json 加载/保存/重载
    logging_config.py           统一日志格式 + REQUEST_ID_CTX

    routes/                     薄路由层（见 routes/AGENTS.md）
      messages.py               POST /v1/messages
      openai.py                 POST /v1/chat/completions
      responses.py              POST /v1/responses
      misc.py                   GET /, /health, /v1/models, /count_tokens
      config.py                 配置 CRUD
      endpoints.py              端点 CRUD
      usage.py                  用量查询
      logs.py                   日志查询
      latency.py                延迟测试

    handlers/                   Pipeline 模式处理（见 handlers/AGENTS.md）
      base.py                   PipelineContext + HandlerStep + Pipeline + PipelineStop
      messages_handler.py       Messages 路由 Pipeline
      openai_handler.py         OpenAI Chat 路由 Pipeline
      responses_handler.py      Responses 路由 Pipeline
      shared/                   可复用步骤（auth, model_resolve, proxy 等）

    protocol/                   协议转换层（见 protocol/AGENTS.md）
      ir/                       [新通道] 建议新代码走此路径
      anthropic_openai/         [旧通道] PendingDeprecationWarning
      responses_chat/           [旧通道] PendingDeprecationWarning
      capabilities.py           协议可达性表 + select_upstream 算法
      errors.py                 错误格式化
      ...

    services/                   纯业务逻辑
      tool_call_fix.py          孤立 tool_call 修复
      vision_service.py         图像处理
      input_compressor.py       输入压缩

    infra/                      基础设施
      db.py                     SQLite 操作（用量记录、端点管理）
      archive.py                JSONL 用量归档
      http_client.py            全局 HTTP 客户端（直接 / 代理）
      url_utils.py              URL 工具

    middleware/                 FastAPI 中间件
      request_id.py, access_log.py, __init__.py (catch_all_exceptions)

  static/                       前端（Preact + Vite）
    src/                        TSX 源码
    css/                        CSS（main.css, logs.css）
    dist/                       Vite 构建产物（被 FastAPI 作为静态文件挂载）
    package.json

  tests/                        测试（pytest + smoke_test.py）
  docs/                         技术文档
  dev.sh                        开发启动脚本（port 4010, screen + uvicorn --reload）
  start.sh                      本地启动脚本（port 4000）
  restart.sh                    重启脚本
  Dockerfile                    Docker 构建（port 4000）
  docker-compose.yml            Docker Compose 部署
  config.example.json           配置示例
  proxy.py                      向后兼容入口（import llm_proxy.main:app）
```

## 三个核心概念

### 1. Pipeline 模式（参见 handlers/AGENTS.md）

所有对外 API 请求走 Pipeline 步骤链：
1. **AuthStep** — 提取 API Key → 匹配端点 → 校验权限
2. **ModelResolveStep** — 解析模型名 → 端点权限校验 → 返回六元组
3. **VisionFallbackStep** — 不支持视觉的模型：图片→文本降级
4. **CompressionStep** — 输入压缩（节省 token）
5. **ResponsesConvertStep** — 仅在 Responses 路由中，Responses→Chat 格式转换
6. **ProxyStep** — 根据 upstream_protocol 转发（同协议透传 / 跨协议转换 / IR 路径）

步骤可抛出 PipelineStop(response) 中断管道，中间件识别并返回。

### 2. 协议转换架构（参见 protocol/AGENTS.md）

有两套并存的转换通道：

| 通道 | 状态 | 说明 |
|------|------|------|
| protocol/ir/（IR 层） | 新通道 | 建议新代码走此路径 |
| protocol/anthropic_openai/ | 旧通道 PendingDeprecationWarning | 保留不破坏 ProxyStep |
| protocol/responses_chat/ | 旧通道 PendingDeprecationWarning | 保留不破坏 ProxyStep |

IR 层用法：
```
from llm_proxy.protocol.ir import convert_request, convert_response
upstream_body = convert_request("anthropic", "openai/responses", body)
client_resp = convert_response("openai/responses", "anthropic", upstream_body)
```

协议选择走 select_upstream()（capabilities.py）：同协议优先 → 按 IMPLEMENTED_CONVERSIONS 顺序选。

### 3. 模型与端点

**模型三个概念（极易混淆）**：
- Config Key：config.json 的键、用量记录用
- Upstream Model：实际发给上游的模型名
- Display Name：仅前端展示

**端点认证**：
- 请求带 x-api-key 或 Authorization: Bearer
- 通过 API Key hash（SHA256 前 16 位）匹配端点
- 不匹配 → 尝试 default 端点 → 再不匹配 → 401
- 默认端点（api_key="default"）不可删除

**模型解析**：
- Claude 系列：Codex-opus-4-7 → opus-4-7 → 查 family_routing（端点优先）
- 回退：opus-4-7 无匹配 → opus（去版本号）
- 非 Claude：直接查 model_map（key 全小写）
- Codex slug（如 gpt-5.3-codex）：在端点的 family_routing 中直接匹配

## 请求数据流详解

### Anthropic 格式（POST /v1/messages）
```
routes/messages.py → MessagesHandler.handle()
  Auth → ModelResolve → VisionFallback → Compression → Proxy
    Proxy._handle_anthropic()
      同协议透传 或 anthropic_to_chat() → chat 上游 → chat_to_anthropic()
```

### OpenAI Chat 格式（POST /v1/chat/completions）
```
routes/openai.py → OpenAIHandler.handle()
  Auth → ModelResolve → OpenAIProtocol → ToolCallFix → VisionFallback → Compression → Proxy
    Proxy._proxy_to_chat()
      同协议透传 或 Chat→Responses 转换
```

### OpenAI Responses 格式（POST /v1/responses）
```
routes/responses.py → ResponsesHandler.handle()
  Auth → ModelResolve → ProtocolSelect → VisionFallback → Compression → ResponsesConvert → Proxy
    Proxy._proxy_to_chat()
      Responses→Chat 流式/非流式转换 或 同协议透传
```

## 关键 API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /v1/messages | Anthropic 格式 |
| POST | /v1/chat/completions | OpenAI Chat 格式 |
| POST | /v1/responses | OpenAI Responses API 格式 |
| POST | /v1/messages/count_tokens | Token 计数 |
| GET | /v1/models | 模型列表（支持 Codex ModelInfo 格式） |
| GET/PUT | /api/config | 全局配置（PUT 自动删除 family_routing） |
| CRUD | /api/endpoints | 端点管理 |
| GET | /api/usage | 用量查询 |
| GET | /api/logs/* | 日志查询 |

## Import 规则

| 要用的东西 | 正确写法 |
|------------|----------|
| 数据库查询 | from llm_proxy.infra import db |
| 用量归档 | from llm_proxy.infra.archive import archive_record |
| 错误格式化 | from llm_proxy.protocol.errors import ... |
| 全局状态 | from llm_proxy.state import get_state, init_state |
| 模型解析 | from llm_proxy.state import resolve_model_for_endpoint |
| API Key 提取 | from llm_proxy.handlers.shared.auth import _extract_api_key |
| IR 转换（推荐） | from llm_proxy.protocol.ir import convert_request, convert_response |
| 旧通道（不推荐） | from llm_proxy.protocol.anthropic_openai import ... |
| 旧通道（不推荐） | from llm_proxy.protocol.responses_chat import ... |

**禁止**：
- `from llm_proxy import state` + `state.xxx`（必须走 get_state()）
- 在 protocol/ 代码中引用 HTTP 请求/响应对象
- 在路由文件中写业务逻辑

## 配置结构

config.json（已 gitignore，参考 config.example.json）：

```json
{
  "models": {
    "模型ID": {
      "api_base": "...",
      "api_key": "...",
      "upstream_model": "实际发给上游的模型名",
      "upstream_protocols": ["anthropic", "openai"],
      "vision_support": false,
      "context_window": 1000000,
      "display_name": "前端展示用",
      "allow_proxy": false
    }
  },
  "error_handling": { "failover_enabled": true, "no_retry_enabled": true },
  "compression": { "enabled": true, "max_input_tokens": 80000 }
}
```

注意：
- family_routing 仅存在端点的 family_routing 字段（不在 config.json 里）
- api_base 不含 /v1/... 后缀
- upstream_protocol（标量）已被 upstream_protocols（数组）取代，但标量仍兼容

## 配置加载链

```
config.json → config_loader.load_config() → State.__init__()
                    → 构建 model_map, vision_map, protocols_map, paths_map, allow_proxy_map
```

端点配置存储在 usage.db 的 endpoints 表中。PUT /api/config 会将端点覆盖合入数据库。

## 错误处理规则

- 全局中间件 catch_all_exceptions 捕获所有异常，识别 PipelineStop 并返回其响应
- Pipeline 步骤内错误：raise PipelineStop(make_xxx_error(...))
- 未预期异常：中间件兜底 500
- failover 触发：499 / 503；链由 family_routing 配置的 failover 字段动态决定
- x-should-retry: false 阻止 SDK 重试

## 日志体系

- 统一格式：%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s
- Request ID：每请求 8 位 hex，REQUEST_ID_CTX ContextVar，响应头 x-request-id
- 中间件顺序（外→内）：request_id → access_log → catch_all_exceptions
- Access Log 走 llm_proxy.access logger
- API Key 脱敏：handlers/shared/auth.py 的 _extract_api_key() + 前 4 位
- 生命周期日志走 llm_proxy.lifecycle logger，不受全局级别影响
- 流式 chunk 日志为 DEBUG 级别（dev 环境可见）
- httpx/httpcore 已抑制为 WARNING

## IR 层新通道迁移路径

endpoint.settings 加 "ir_enabled": true → handler 根据 flag 选择 IRProxyStep 或 ProxyStep。
全量切换后删除 anthropic_openai/ 和 responses_chat/ 通道。

## 开发与部署

### 开发（dev server, port 4010, git worktree dev 分支）

```bash
cd ../llm-proxy-dev && ./dev.sh start
```

日志：cd ../llm-proxy-dev && ./dev.sh log
附加终端：screen -r llm-proxy-dev（Ctrl-A D 退出）

环境变量：自动加载 .dev-env（LLM_PROXY_DEV=true, LLM_PROXY_LOG_LEVEL=DEBUG）

**DB 同步（从 main 复制数据到 dev）**：
1. ./dev.sh stop
2. cp usage.db usage.db.bak.dev-$(date +%Y%m%d-%H%M%S)
3. cp /path/to/main/usage.db ../llm-proxy-dev/usage.db
4. 同样处理 config.json
5. ./dev.sh start

不要 git add/rm usage.db、*.bak.dev-*

### 生产（Docker, port 4000）

```bash
docker compose up -d --build
```

必须用 docker-compose，不能用裸 docker run。原因：config.json 和 usage.db 通过 volume 挂载，裸 docker run 不会挂载 volume，Docker 会自动创建空目录导致 FileNotFoundError。

### 测试

```bash
python -m pytest tests/ -v          # 单元测试
python tests/smoke_test.py          # 冒烟测试
```

## 日志分层

| 环境 | 默认级别 | 说明 |
|------|----------|------|
| dev（LLM_PROXY_DEV=true） | DEBUG | 全量日志 |
| prod（Docker/main） | WARNING | 仅警告和错误 |
| 手动 | LLM_PROXY_LOG_LEVEL=INFO | 可调高 dev 级别 |

## 前端规范

详见 static/ 目录。关键规则：
- CSS token 在 :root 中定义，禁止派生新 token
- 字号 7 档：--fs-xs(11) / --fs-sm(12) / --fs-base(13) / --fs-md(14) / --fs-lg(15) / --fs-xl(18) / --fs-2xl(24)
- 禁止内联 fontSize、color、background、border、borderRadius 等样式
- 三次重复的 modal/field/form 控件必须先抽组件
- class 命名：单词（模态框根）、父-子（元素）、形容词（状态）、短横线连接（工具）
- 禁止驼峰式、组件前缀、BEM

## 已知陷阱

1. **Grid stretch + Preact**：并排 card 必须写 height: 100%，不能依赖默认 stretch
2. **section-title 主次变体**：同级并排 card 标题统一用 .section-title，嵌套父级才用 .section-title.lg
3. **CSS @import**：所有 CSS 由 app.css 一个入口 import，禁止在 main.css 末尾再 import logs.css
4. **inline fontSize**：发现一处就清理一类，新加 inline 字号一律拒收
5. **重复 Modal/Field**：第三次复制前必须先抽组件
6. **apply_patch 的 old_str 不为空**：对齐 Claude Code 设计，保证多轮历史一致性
7. **子节点 Parallel Risk**：多个 subagent 不要同时修改同一文件
8. **config.json 是 gitignore 的**：用 config.example.json 做模板
9. **upstream_protocol 标量已弃用**：新加模型用 upstream_protocols 数组
10. **proxy.py 是兼容入口**：直接 import llm_proxy.main:app，不要迁内容过来
11. **llm_proxy/adapters/ 目录为空**：已废弃，不要往里加东西
12. **sidecar 配置已废弃**：config.json 中残留的 sidecar key 会被 config_loader.py 发出 DeprecationWarning 并忽略
13. **SSE data type 必须注入**：`sse_format()` 自动在 data JSON 顶层注入 `type` 字段（与 `event:` 头镜像）。
    Codex / Claude Code 只解析 data 的 `type` 字段，不认 `event:` 头；缺失则收不到任何事件，
    最终报错 "stream closed before response.completed"。这是流式响应的常见遗漏点。
14. **developer role → system 降级**：`protocol/ir/chat.py` 在 `_message_ir_to_chat()` 中自动将
    `role == "developer"` 降级为 `"system"`。OpenAI 引入 developer role 后部分上游（如 DeepSeek）
    不兼容此角色值，必须降级否则请求失败。
15. **tool_call item_id 须独立生成**：IR 通道和 responses_chat 旧通道中，所有 tool_call item 的 `id`
    统一用 `uuid4().hex[:24]` 独立生成，**禁止**引用上游 `tool_call.id`（可能含 Chat 格式的
    不可用前缀如 `call_`）或 block.id（与 item_id 语义不同）。`StreamState` 新增 `func_item_ids`
    字典持久化每轮 tool call 的 item_id，避免流式过程中重建 item_id 时不一致。

## 更多细节

本目录包含子目录 AGENTS.md：
- llm_proxy/handlers/AGENTS.md  — Pipeline 模式详解
- llm_proxy/protocol/AGENTS.md  — 协议转换架构详解
- llm_proxy/routes/AGENTS.md    — 路由表与委托规则
