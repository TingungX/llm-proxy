# AGENTS.md

LLM Proxy — 基于 FastAPI 的 LLM API 聚合网关，支持多模型、多协议的双向格式转换。Anthropic Messages、OpenAI Chat Completions、OpenAI Responses 三种请求格式均可路由到任意格式的上游模型。

## Dev 环境规范（强约束）

**所有代码修改和调试必须在 dev server 中进行，在 CLI 副本测试通过后才准合并到 main。**

### 架构

```
生产环境（不受影响）          开发环境
─────────────────          ─────────────────
llm-proxy (main)           llm-proxy-dev (dev 分支)
  └─ Docker :4000            └─ uvicorn --reload :4010
  └─ Codex → :4000           └─ Codex dev profile → :4010
  └─ Claude Code → :4000     └─ Claude Code (项目级配置) → :4010
```

### dev server（`../llm-proxy-dev/`）

- **位置**：`/Users/tingung/Projects/github/llm-proxy-dev`（git worktree，`dev` 分支）
- **端口**：4010
- **启动**：`cd ../llm-proxy-dev && ./dev.sh start`
- **停止**：`cd ../llm-proxy-dev && ./dev.sh stop`
- **日志**：`cd ../llm-proxy-dev && ./dev.sh log`（或 `tail -f ../llm-proxy-dev/dev-server.log`）
- **连接终端**：`screen -r llm-proxy-dev`（Ctrl-A D 退出）
- **环境变量**：自动加载 `../llm-proxy-dev/.dev-env`（`LLM_PROXY_DEV=true`，`LLM_PROXY_LOG_LEVEL=DEBUG`）
- **DB 同步**：从 main 复制 `usage.db` 和 `config.json`（见下方同步步骤）

### Codex CLI dev 副本

```bash
# 连接 dev server (port 4010)
codex -p dev

# 对比：连接生产 (port 4000) — 不变
codex
```

- **Profile 文件**：`~/.codex/dev.config.toml`
- 仅覆盖 `model_provider` 和 `base_url`，其他配置（插件、MCP、features）继承自 `~/.codex/config.toml`

### Claude Code CLI dev 副本

```bash
# 在 llm-proxy-dev 目录下启动 Claude Code
cd ../llm-proxy-dev
claude

# 项目级 .claude/settings.local.json 自动覆盖 ANTHROPIC_BASE_URL 为 :4010
```

- **配置文件**：`../llm-proxy-dev/.claude/settings.local.json`
- 本机 `~/.claude/settings.json` 不变（仍指向 :4000）

### DB 同步步骤

dev server 接入的是**稳定服务器**（main 分支 root）的数据：

1. 停止 dev server：`cd ../llm-proxy-dev && ./dev.sh stop`
2. 备份当前 dev DB：`cp usage.db usage.db.bak.dev-$(date +%Y%m%d-%H%M%S)`
3. 复制 main DB：`cp /Users/tingung/Projects/github/llm-proxy/usage.db ../llm-proxy-dev/usage.db`
4. 同样处理 `config.json`
5. 启动 dev server：`./dev.sh`
- **不要**直接 git add/rm `usage.db`、`*.bak.dev-*`

### 工作流

1. **修改代码**：在 `../llm-proxy-dev/` 中修改，dev server 自动热更新
2. **验证**：用 CLI 副本（`codex -p dev` 或 `cd ../llm-proxy-dev && claude`）测试
3. **合并**：CLI 副本测试通过后，将 dev 分支合并到 main
4. **部署**：main 分支通过 Docker 部署到生产（:4000）

### 日志分层

| 环境 | 默认级别 | 说明 |
|------|----------|------|
| dev (`LLM_PROXY_DEV=true`) | DEBUG | 全量日志：per-request 路由、协议选择、流处理、chunk 内容 |
| prod (Docker/main) | WARNING | 仅警告和错误；生命周期日志（启动/关闭/定时任务）始终 INFO |
| 手动覆盖 | `LLM_PROXY_LOG_LEVEL=INFO` | 可随时调高 dev 日志级别 |

- 生命周期日志使用 `logging.getLogger("llm_proxy.lifecycle")`，不受全局级别影响
- per-request 日志（路由、协议选择、流处理）在代码中为 `logger.debug`，dev 环境下可见，prod 下静默

## 验证（改代码后必跑）

```bash
python -m pytest tests/ -v && python tests/smoke_test.py
```

## 命令

```bash
# 开发（dev server, port 4010）
cd ../llm-proxy-dev && ./dev.sh

# 生产（Docker, port 4000，已部署）
# ⚠️ 必须用 docker-compose 启动，不能用裸 docker run
# 原因：config.json 和 usage.db 通过 volume 挂载，裸 docker run 不会挂载，
# Docker 会把不存在的 bind mount 源自动创建为空目录，导致 FileNotFoundError
docker compose up -d

# 重建镜像后重启
docker compose up -d --build

# 测试
python -m pytest tests/ -v                                       # 单元测试
python tests/smoke_test.py                                       # 冒烟测试
```

## 前端规范

### 样式组织

- **入口**：`static/src/app.css`（被 Vite 编译），import `main.css` 和 `logs.css`
- **设计 token**（CSS 变量，定义在 `:root`）：
  - 颜色：`--bg` / `--surface` / `--bg-elevated` / `--border` / `--text` / `--text-secondary` / `--text-muted` / `--accent` / `--accent-dim` / `--danger`
  - 字号：`--fs-xs`(11) / `--fs-sm`(12) / `--fs-base`(13) / `--fs-md`(14) / `--fs-lg`(15) / `--fs-xl`(18) / `--fs-2xl`(24)
  - 字体：`--font-mono` = `'SF Mono', Monaco, monospace`
- **480px 响应式**会重定义所有字号 token，无需单独处理
- **不要**派生新 token；先复用 7 档之一

### 字号语义（强约束）

| Token | 用途 | 禁用场景 |
|---|---|---|
| `--fs-xs` | 标签 uppercase、tag/badge、caption、field-label | 主体文本 |
| `--fs-sm` | table body、label、helper text、按钮、field-hint | 大标题 |
| `--fs-base` | 主体 input/select 文本、body、log meta | 标题 |
| `--fs-md` | day-arrow/day-label（强调） | 通用 |
| `--fs-lg` | modal-title、card heading、section-title.lg | 表格 |
| `--fs-xl` | h1、stat 用量值、usage-value | footer |
| `--fs-2xl` | stat-box 大数字、modal close | 普通标签 |

**禁止**：
- 内联 `fontSize:`（必须走 token 或 class）
- 硬编码 `10px`/`11px`/`12px` 等字面量
- 使用 `em` / `rem` / `%` 表达字号（用 `.text-xs`/`.text-sm`/`.text-base` 等替代）
- 在 `:root` 外定义字号

### 颜色使用

- 文本：`--text`（主）/ `--text-secondary`（副）/ `--text-muted`（提示）
- 背景：`--bg`（页面）/ `--surface`（卡片）/ `--bg-elevated`（弹层）
- 状态：`--accent`（成功/主色）/ `--danger`（错误）
- **禁止**内联颜色字面量；按钮上的 `#000` 是按钮对比色例外

### 内联 style 规则

- **允许**：`display` / `flex-direction` / `gap` / `grid-template-columns` / `align-items` / `justify-content` / `margin` / `padding` / `width: <具体值>` / `height: <具体值>` / `maxHeight` / `overflow` / `minWidth`
- **禁止**：`color` / `background` / `fontSize` / `fontWeight` / `border` / `borderRadius` / `textTransform` / `letterSpacing`（必须走 class）
- 例外：动态条件样式（chip 选中态、协议 badge 颜色）用 className 切换或专门的 class

### 组件复用

- **可复用**：`Modal`、`Field`、`ProtocolChip`、`ProtocolBadge`、`EmptyState`、`Toggle`、`Toast`、`ChartCanvas`
- **新组件**放 `static/src/components/`，命名 PascalCase
- Modal 必须用 `<Modal size="sm|md|lg" onClose>`，**禁止**复制 ModalOverlay 内联
- 表单字段必须用 `<Field label hint error required>`，**禁止**复制 Field 实现
- 协议 chip/徽章必须用 `<ProtocolChip>` / `<ProtocolBadge>`，**禁止**自写 button[style]
- 空状态用 `<EmptyState>` 或 `<EmptyState compact>`
- **Toggle**：`.toggle-row` 内置 `gap: 10px`（label 与 switch 间距），不要在外部再加 margin/gap

### class 命名

- 组件根：`.modal`、`.heatmap-wrapper`、`.chart-container`（单词）
- 元素：`.heatmap-cell`、`.modal-content`（父-子，单数）
- 状态：`.active`、`.on`、`.off`、`.toast.ok` / `.toast.err`（形容词）
- 工具：`.text-xs`、`.text-muted`、`.w-full`、`.font-mono`、`.modal-body-pad`（短横线连接）
- **禁止**：驼峰式、组件前缀（`.Field__label`）、BEM

### 调试规范

- 添加/修改样式后必须浏览器实查（不要只看 diff）
- 修改 `main.css` / `logs.css` 后重启 Vite dev server
- 修改 token 后必须全站扫一眼，**不允许**只在新组件里改
- 引入新 class 前先 grep `static/css/`，**禁止**和已有 class 重复

### 已知陷阱（重构过程中暴露的问题，写给未来的自己）

#### 1. Grid stretch 在 Preact 渲染下不总是生效

症状：并排的 `.card`（如 `用量概览` / `用量热力图`）底部不对齐，左 card 比右 card 矮 16px，正好是 padding-bottom。

根因：grid 默认 `align-items: stretch` 应该让 item 填满行高，但 Preact 异步渲染下，card 内部 `display: flex` 或 content 计算完成后才确定 card 高度，grid 想再 stretch 时已被 content-size 锁定。

修复：`.usage-overview-row` 显式 `align-items: stretch` + 两个 card 显式 `height: 100%` + `box-sizing: border-box`。

**规则**：任何"并排 + 等高"的 card 布局都必须在 CSS 里写 `height: 100%`，不能依赖默认 stretch。

#### 2. `section-title` 的"主次"变体只用于真正嵌套的层级

症状：用量页两个并列的 card（概览 + 热力图）一个用 `.section-title.lg`（15px 白色），一个用 `.section-title`（11px uppercase 灰色），视觉上突兀。

根因：误把"主-次"语义套在同级并排的 card 标题上。`.section-title.lg` 真正适用的是"父级 section 的标题"或"页面级 h3"，不是并排兄弟。

**规则**：
- 同级并排的 card 标题统一用 `.section-title`
- 嵌套的父级标题用 `.section-title.lg`
- 禁止**在并排两列里**用 `.section-title.lg` / `.section-title` 做"主次"对比

#### 3. CSS @import 不要在 main.css 里串到 logs.css

症状：原本 `app.css` 用 `@import` 链式加载 `main.css` → `logs.css`，多一层无意义依赖。

**规则**：所有 CSS 由 `app.css` 一个入口 `@import`，Vite 原生处理 `main.css` + `logs.css` 即可。**禁止**在 `main.css` 末尾再 import `logs.css`，会让加载顺序耦合到具体文件。

#### 4. inline `fontSize` / `em` / 硬编码 px 是反复出现的反模式

症状：发现 4 处 `fontSize: '10px'` / `'0.85em'` / `'0.9em'` 散落在 4 个组件，每次改字号都要全局搜。

**规则**：发现一处就清理一类。新加 inline 字号一律拒收，强制走 className 工具类或 token。

#### 5. 重复的 Modal/Field/ProtocolChip 是早期重构技术债

症状：`ModalOverlay` 复制 4 次，`Field` 复制 2 次，`ProtocolChip` 逻辑重复但 markup 不一致（一个 `button[style]` 一个 `button.protocol-chip`）。

修复：抽取 `components/Modal.tsx` / `Field.tsx` / `ProtocolChip.tsx`，4 处 ModalOverlay 全部替换。

**规则**：复制超过 1 次的 modal/field/form 控件必须先抽组件，**禁止**第三次复制。

## 架构速览

```
llm_proxy/main.py:app (FastAPI)
├─ routes/            薄路由层，接收请求 → 委托 Handler
├─ handlers/          Pipeline/Handler — 请求处理管道
│   ├─ base.py        PipelineContext + HandlerStep + Pipeline + PipelineStop
│   ├─ *_handler.py   各路由 Pipeline 组装
│   └─ shared/        可复用步骤（auth / model_resolve / protocol_select / vision_fallback / compression / paths / responses_convert / proxy）
├─ protocol/          协议转换层（独立，无路由感知）
│   ├─ capabilities.py  协议可达性表 + 上游选择算法
│   ├─ constants.py     Stop Reason 映射常量
│   ├─ detector.py     协议检测
│   ├─ sse.py          SSE 透传
│   ├─ errors.py       错误格式化（make_anthropic_error / make_openai_error）
│   ├─ think_tag.py    Think 标签检测与提取（MiniMax M3）
│   ├─ anthropic_openai/  Anthropic ↔ OpenAI 转换通道（Python 实现）
│   │   ├─ request.py     Anthropic→Chat 请求转换
│   │   ├─ response.py    Chat→Anthropic 响应转换
│   │   ├─ stream.py      Chat SSE→Anthropic SSE 流转换
│   │   └─ rectifier.py   请求修正（schema清理、tool映射等）
│   └─ responses_chat/    Responses API ↔ Chat Completions 转换通道
│       ├─ request.py     Responses→Chat 请求转换 + 流式事件转换
│       ├─ response.py    Chat→Responses 非流式 + 反向转换
│       ├─ stream.py      StreamState + ThinkTagStateMachine（已合并）
│       ├─ tool_replacement.py  apply_patch ↔ 标准文件工具
│       └─ usage.py       用量格式转换
├─ services/          纯业务逻辑（tool_call_fix / vision_service / input_compressor）
├─ infra/             基础设施层
│   ├─ archive.py     JSONL 用量归档（后台线程写入）
│   ├─ db.py          SQLite 操作
│   ├─ http_client.py 全局 HTTP 客户端
│   └─ url_utils.py   URL 工具
├─ middleware/        中间件（request_id / access_log / catch_all_exceptions）
├─ state.py           State 类 + get_state() + init_state()
├─ config_loader.py   config.json 读写（已 gitignore，用 config.example.json 做模板）
└─ logging_config.py   统一日志格式
```

## 开发规范

### Pipeline/Handler 模式

所有请求处理走 Pipeline：

```python
# handlers/responses_handler.py  (实际示例)
class ResponsesHandler:
    def __init__(self):
        self.pipeline = Pipeline([
            AuthStep(),
            ModelResolveStep(),
            ProtocolSelectStep(),    # 协议选择（含 tool 降级）
            VisionFallbackStep(),    # 图像→文本降级
            CompressionStep(),       # 输入压缩（节省 token）
            ResponsesConvertStep(),  # Responses→Chat 格式转换
            ProxyStep(),             # 代理转发
        ])

    async def handle(self, request: Request) -> JSONResponse | StreamingResponse:
        ctx = PipelineContext(request=request, body=await request.json(),
                              headers=dict(request.headers), error_protocol="openai")
        return await self.pipeline.execute(ctx)
```

- **新步骤**：继承 `HandlerStep`，实现 `async def execute(self, ctx) -> None`
- **中断管道**：`raise PipelineStop(response)` — 中间件会识别并返回其携带的响应
- **共享步骤**放 `handlers/shared/`，路由特有步骤放 `handlers/xxx_handler.py` 内
- **路由文件**只做委托，不包含业务逻辑

### 协议层规范

- `protocol/` 只做格式转换，**不感知 HTTP 请求/响应**
- 两个通道完全独立：`anthropic_openai/` 和 `responses_chat/`
- `__init__.py` 暴露公共 API，上层代码应从此处导入

### State 访问

- 所有运行时状态通过 `get_state()` 获取
- `init_state()` 在 `lifespan` 中显式调用
- 禁止 `from llm_proxy import state` + `state.xxx` 模式
- 测试中可用 `init_state(test_config)` 注入

### 错误处理

- `catch_all_exceptions` 中间件识别 `PipelineStop` 并返回其携带的响应
- 步骤内错误：`raise PipelineStop(make_xxx_error(...))`
- 未预期异常：中间件兜底返回 500

### Import 规范

| 模块 | 正确 import |
|------|------------|
| 数据库 | `from llm_proxy.infra import db` |
| 用量归档 | `from llm_proxy.infra.archive import archive_record` |
| URL 工具 | `from llm_proxy.infra.url_utils import ...` |
| 错误格式化 | `from llm_proxy.protocol.errors import ...` |
| 状态 | `from llm_proxy.state import get_state, init_state` |
| 模型解析 | `from llm_proxy.state import resolve_model_for_endpoint` |
| API Key 提取 | `from llm_proxy.handlers.shared.auth import _extract_api_key` |
| Responses↔Chat | `from llm_proxy.protocol.responses_chat import ...` |
| Anthropic↔Chat | `from llm_proxy.protocol.anthropic_openai import ...` |

## 模型三个概念（极易混淆）

| 概念 | 用途 | 示例 |
|------|------|------|
| **Config Key** | config.json 键、family_routing 值、权限与用量记录 | `"my-model"` |
| **Upstream Model** | 实际发给上游的模型名 | `"upstream-model-name"` |
| **Display Name** | 仅前端展示 | `"My Model (Provider)"` |

## resolve_model_for_endpoint() 返回六元组

`(api_base, api_key, upstream_model, config_key, upstream_protocol, failover_family)`
- 权限/用量用 `config_key`，请求体用 `upstream_model`
- 实际协议选择走 `protocols_map` + `select_upstream()`（`protocol/capabilities.py`），而非此处返回值

## 路由逻辑

1. Claude 系列：`Codex-opus-4-7` → `opus-4-7` → 查 `family_routing`（端点优先于全局）
2. 回退：`opus-4-7` 无匹配 → `opus`（去版本号）
3. 非 Claude：直接查 `MODEL_MAP`（key 全小写，不区分大小写）
4. Codex slug（如 `gpt-5.3-codex`）：在端点的 `family_routing` 中直接匹配

## 端点与认证

- 所有请求必须带 `x-api-key` 或 `Authorization: Bearer`
- 通过 API Key 匹配端点（hash 前 16 位），不匹配 → 尝试 default 端点 → 再不匹配 → 401
- 默认端点（`api_key="default"`）不可删除
- 端点设置优先于全局；`api_key` 明文存在 `endpoints` 表

## 关键 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | Anthropic 格式：同协议透传 / Python 跨协议转换 |
| POST | `/v1/chat/completions` | OpenAI 格式：同协议透传 / Chat→Responses 跨协议转换（含 tool_call_fix） |
| POST | `/v1/responses` | Responses API 格式：同协议透传 / Responses→Chat 格式转换（含 apply_patch 展开）|
| POST | `/v1/messages/count_tokens` | Token 计数（转发上游） |
| GET | `/v1/models` | 模型列表 |
| GET/PUT | `/api/config` | PUT 会**删除 family_routing 字段**（已迁移到端点表） |
| GET/POST/PUT/DEL | `/api/endpoints` | 端点 CRUD（含 `/:endpoint_id` 子路由） |
| GET | `/api/usage[?days=&group_by=&granularity=]` | 用量查询（明细7天，聚合90天） |
| GET | `/api/usage/summary` | 用量汇总统计 |
| POST | `/api/latency` | 延迟测试 |
| PUT | `/api/models/{model_id}` | 更新单个模型配置 |
| DELETE | `/api/models/{model_id}` | 删除单个模型 |
| POST | `/api/detect-protocol` | 检测上游协议 |
| POST | `/api/providers/{model_id}/detect` | 检测指定模型的上游协议 |
| GET | `/api/logs/list` | 日志列表查询（支持分页/筛选/时间范围） |
| GET | `/api/logs/summary` | 日志数据汇总统计 |
| GET | `/api/logs/filter-options` | 日志筛选选项（端点/模型/状态） |

## 配置结构

```json
{
  "models": { "模型ID": { "api_base": "...", "api_key": "...", "upstream_model": "...", "upstream_protocols": ["anthropic|openai"], "upstream_paths": {"anthropic/messages": "..."}, "vision_support": false, "context_window": 1000000, "display_name": "..." } },
  "error_handling": { "failover_enabled": true, "no_retry_enabled": true },
  "compression": { "enabled": true, "max_input_tokens": 80000 }
}
```

- `family_routing` 仅存端点表（`PUT /api/config` 自动删除）
- 模型 api_base 不含 `/v1/...` 后缀
- 协议检测顺序：端点上配置 → 模型配置（`upstream_protocols` 数组）→ 运行时自动探测（Anthropic 优先）
- 模型级 `upstream_protocol`（标量）已被 `upstream_protocols`（数组）取代，但标量仍兼容
- 模型级别可选字段：`upstream_paths`（per-protocol 路径映射）、`vision_support`、`context_window`、`allow_proxy`

## ~~Sidecar 进程~~（已弃用）

Sidecar 已由 `protocol/anthropic_openai/` Python 原生转换通道替代，`infra/sidecar.py` 已删除。config.json 中残留的 `sidecar` key 会被 `config_loader.py` 发出 `DeprecationWarning` 并忽略。

## 错误处理

- 全局中间件 `catch_all_exceptions` 兜底所有异常，**识别 `PipelineStop` 并返回其响应**
- Pipeline 步骤用 `raise PipelineStop(make_xxx_error(...))` 中断并返回错误
- failover 触发：429 / 503（`_is_rate_error` 检测）；链由 `family_routing` 配置的 `failover` 字段动态决定
- `x-should-retry: false` 阻止 SDK 重试

## 日志体系

- **统一格式**：`%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s`
- **Request ID**：每请求生成 8 位 hex，通过 `REQUEST_ID_CTX` ContextVar 透传，响应头 `x-request-id` 返回
- **中间件顺序**（外→内）：`request_id_middleware` → `access_log_middleware` → `catch_all_exceptions`
- **Access Log**：`llm_proxy.access` logger 记录 `METHOD /path STATUS X.Xms`
- **API Key 脱敏**：日志中仅打印前 4 位 + `***`，统一使用 `_extract_api_key`（`handlers/shared/auth.py`）
- **流式 chunk**：`request.py` 的 Raw line / Received chunk 日志为 DEBUG 级别
- **httpx/httpcore**：已抑制为 WARNING，不再出现在 INFO
- **error 日志**：捕获异常的 `logger.error` 统一使用 `exc_info=True`

## Anthropic ↔ OpenAI 跨协议转换（Python 实现）

`protocol/anthropic_openai/` 替代了原来的 `anthropic-proxy-rs` sidecar，实现 Anthropic Messages ↔ OpenAI Chat Completions 双向转换。

- `request.py`：Anthropic→Chat 请求转换（reasoning_effort 映射、o-series max_completion_tokens、billing header 剥离、tool_choice 映射、clean_schema、cache_control 保留）
- `response.py`：Chat→Anthropic 响应转换（reasoning_content→thinking、tool_calls→tool_use、refusal→text、finish_reason 映射、cache token 映射）
- `stream.py`：Chat SSE→Anthropic SSE 流转换（状态机管理 thinking/text/tool_use block、tool call delayed start）
- `rectifier.py`：请求修正（schema 清理、工具映射等）

## Responses → Chat Completions 转换规则

`protocol/responses_chat/request.py` 负责将 Responses API 请求转为 Chat Completions 请求，
并将上游 Chat Completions SSE 流转为 Responses SSE 事件。

### 请求转换 (`convert_input_to_messages`)

| Responses 类型 | Chat 消息 |
|------|------|
| `instructions` | `{"role": "system", "content": "..."}` |
| `type: "reasoning"` | 合并到下一个 `assistant` 消息的 `reasoning_content`，多个用换行拼接 |
| `type: "message"` | 直接映射（`content` 为数组时过滤 `input_text`/`output_text`/`input_image`） |
| `type: "function_call"` | `assistant` 消息 + `tool_calls` 数组 |
| `type: "function_call_output"` | `tool` 消息 + `tool_call_id` + content=output |

### 参数映射

- `reasoning.effort` → `reasoning_effort`（映射：none/auto/minimal→low/low/medium/high/xhigh）
- `max_output_tokens` → `chat_body["max_tokens"]`（仅当存在且非 None 时）
- `temperature` / `top_p` → 同名透传
- 流式请求自动添加 `stream_options.include_usage`

### 流式事件转换 (`convert_chunk_to_events`)

- `reasoning_content` delta → `response.reasoning_summary_text.delta`（DeepSeek-V4/o1）
- `content` delta → `response.output_text.delta`（先过 `.HORIZONTAL ELLIPSIS` 状态机分流到 reasoning）
- `tool_calls` delta → `response.function_call.started` / `response.function_call_arguments.delta`
- `finish_reason="stop"` → `response.output_text.done` + `response.output_item.done`
- `[DONE]` → `response.completed`（含完整 `output` 数组 + `usage`）

### Think 标签自动检测（在 `stream.py` 中）

- **CanStart 保护**：仅响应第一个 content chunk 允许触发 `.HORIZONTAL ELLIPSIS` 检测；一旦非 `.HORIZONTAL ELLIPSIS` 正文通过，永久禁用以防误判
- 跨 chunk 边界的前缀匹配（如 `<thi` 缓存到下一 chunk）
- 前导空白：匹配到 `.HORIZONTAL ELLIPSIS` 时丢弃，否则作为正文输出

### Keepalive

- `stream_chat_to_responses` 用 `asyncio.create_task` + `Queue` + `wait_for(timeout=15s)` 实现
- 15 秒无新数据则发送 SSE 注释行 `: keepalive\n\n`

## Tool Call 修复 (`services/tool_call_fix.py`)

处理 Codex 历史中断产生的 tool_call/tool response 不匹配问题（DeepSeek 等上游严格校验）。

`fix_orphaned_tool_calls(messages)`：
1. 每个 `assistant(tool_calls)` 紧后收集同属 `tool` 消息，按 `tool_call_id` 匹配
2. 缺失 response → 插入占位符 `{"role": "tool", "content": "[Tool call was interrupted]"}`
3. **孤立/错位 tool response → 降级为 user 消息保留上下文**，而非直接删除
4. 无前置 `assistant(tool_calls)` 的纯 `tool` 消息同样降级

降级格式：`"Function call output (<call_id>): <content>"`

## apply_patch → 标准文件工具 转换管道 (`protocol/responses_chat/tool_replacement.py`)

将 Codex 专用的 `apply_patch`（单 string 参数 + 内嵌 DSL）替换为标准文件工具发给上游；上游返回的工具调用还原为 `apply_patch` 的 `custom_tool_call` 事件返回 Codex。

### 核心设计原则（对齐 Claude Code）

**`old_str` 永远不允许为空**。这确保 `replace_in_file` 始终有明确的锚点，保证多轮对话的历史一致性——反向放行空 old_str 会导致正向回传时无法还原，上下文断裂。

### 四个标准工具

| 工具 | 用途 | 对应 apply_patch 语法 |
|------|------|----------------------|
| `write_to_file` | 创建/覆盖文件 | `*** Add File` |
| `replace_in_file` | 替换已有内容（old_str 必须非空） | `*** Update File` + diff（有 `-` 行） |
| `append_to_file` | 追加到文件末尾 | `*** Update File` + 只有 `+` 行 |
| `delete_file` | 删除文件 | `*** Delete File` |

### 完整数据流

```
Codex (Responses API)
  │ tools=[{type:"custom", name:"apply_patch"}]
  ▼
handlers/responses_handler.py ← Pipeline: Auth → ModelResolve → ProtocolSelect → ...
  │
  ▼
convert_tools_to_chat()       ← apply_patch → 四个工具定义
  │                             reverse_tool_map: {四个工具 → "apply_patch"}
  ▼
上游模型 (write_to_file / replace_in_file / append_to_file / delete_file 调用)
  │
  ▼
stream_chat_to_responses()
  │ StreamState(reverse_tool_map=...) → 检测上游 tool name
  │   ├─ 在 reverse_tool_map 中 → reverse_tool_args_to_apply_patch() 转回 DSL
  │   │   └─ ReverseConversionError → 生成包含错误信息的 custom_tool_call
  │   └─ 不在 → 正常 function_call 事件
  ▼
Codex (收到 custom_tool_call 事件 → apply_patch 执行)
```

### 正向转换（历史消息中的 apply_patch → 标准工具）

`convert_input_to_messages` 遇到 `type:"custom"` + `name:"apply_patch"`：
1. `parse_apply_patch_to_simple(input_text)` 解析 DSL，返回 `list[dict]`（支持多文件）
2. 单文件 → 保持原始 `call_id`
3. 多文件 → 每个操作派生独立 `call_id`（`{base_id}_0`, `{base_id}_1`, ...），tool response 复制到所有派生 ID
4. 逐段降级：某段解析失败不影响其他段（生成 `_degraded_user_message`）

#### 只有 `+` 行的 Update File 处理

| 情况 | 处理方式 | 原因 |
|------|----------|------|
| 有 context 行 | `replace_in_file`：context → `old_str`，context + plus → `new_str` | 用已有内容作为锚点（对齐 Claude Code） |
| 无 context 行，无 Move to | `append_to_file`：不需要 old_str | 追加操作，闭环兼容 |
| 无 context 行，有 Move to | `_degraded_user_message` | append_to_file 不支持 destinationPath |

### 反向转换（上游 tool_call → custom_tool_call 事件）

`StreamState` + `reverse_tool_map` 在三个位置触发：
- `handle_tool_call_id` — 检测上游 tool name，输出 `custom_tool_call` 或 `function_call` item
- `close_func_blocks` — 闭合时调用 `reverse_tool_args_to_apply_patch()` 转回 DSL
- `_build_output_items` — 非流式完成的 output 数组同样走逆向

`to_responses_response`（非流式响应）同理。

#### 参数验证（`ReverseConversionError`）

`reverse_tool_args_to_apply_patch` 在参数无效时抛出 `ReverseConversionError`，携带错误原因和修正建议：

| 场景 | 错误原因 | 修正建议 |
|------|----------|----------|
| `write_to_file` content 为空 | `content must not be empty` | 提供实际文件内容 |
| `replace_in_file` old_str 为空 | `old_str must not be empty` | 追加时用 `append_to_file` 而非空 old_str |
| 缺少必填参数 | `missing required args` | 补全参数 |
| 未知工具名 | `unknown tool name` | 使用标准工具名 |

上层捕获 `ReverseConversionError` 后，生成包含错误信息的 `custom_tool_call` 传给 Codex。Codex 执行 apply_patch 时解析失败，返回错误 tool result，上游模型收到错误后可修正重试。

## 非 Function 工具转换（v1.1.1）

Codex 发送的 `namespace`、`web_search` 等非 `function` 类型工具不再被丢弃，统一降级为标准 `function` 类型发给上游。

### 处理规则

| 原始类型 | 转换方式 | reverse_tool_map |
|----------|----------|-----------------|
| `custom` (name=apply_patch) | 展开为 4 个文件操作工具 | 写入（逆向到 custom_tool_call） |
| `custom` (其他) | 透传为 function，保留原生 schema | 写入（逆向到 custom_tool_call） |
| `namespace` | 递归展开子 `function` 工具 | **不写入**（返回 `function_call` 事件） |
| `web_search` | 降级为带 `query` 参数的 function | **不写入** |
| 其他非 function | 降级为 function，尽量保留 name/params | **不写入** |

### 降级触发

`routes/responses.py` 检测到 `tools` 中包含 `{custom, namespace, web_search, tool_search, image_generation}` 类型之一时，标记 `needs_conversion=True`，强制走 chat-completions 路径。

### 代码位置

- `responses_to_chat.py` — `_normalize_params()` 统一参数补齐，`_make_chat_function_tool()` 创建 function 定义
- `routes/responses.py` — `needs_conversion` 检测代替 `has_custom_tools`
- `stream_state.py` — 非 apply_patch 的 custom 工具参数直接 JSON 序列化

### 闭环验证

四个文件工具均实现双向闭环：上游工具 → apply_patch DSL → 正向解析 → 还原为同一工具。

| 工具 | 反向生成的 apply_patch | 正向还原 |
|------|----------------------|---------|
| `write_to_file` | `*** Add File` + 全部 `+` 行 | `write_to_file` ✅ |
| `replace_in_file` | `*** Update File` + context/-/+ 行 | `replace_in_file` ✅ |
| `append_to_file` | `*** Update File` + 只有 `+` 行 | `append_to_file` ✅ |
| `delete_file` | `*** Delete File` | `delete_file` ✅ |
