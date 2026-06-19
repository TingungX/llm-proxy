# LLM Proxy

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Preact-10.22-673AB8?logo=preact" alt="Preact">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License AGPL-3.0">
</p>

**LLM Proxy** — A FastAPI-based LLM API aggregation gateway with **bidirectional format conversion** across multiple models and protocols. Send requests in your preferred protocol format; the proxy automatically converts them to whatever the target upstream requires.

Supported protocol conversions:

| Request Format | Convertible Upstream Formats | Route |
|----------------|------------------------------|-------|
| Anthropic Messages | Anthropic Messages / OpenAI Chat Completions | `/v1/messages` |
| OpenAI Chat Completions | OpenAI Chat Completions / Anthropic Messages | `/v1/chat/completions` |
| OpenAI Responses | OpenAI Chat Completions | `/v1/responses` |

→ **Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses — all three request formats can route to any upstream model through their respective paths.**

## Protocol Matrix

| Client Format | Supported Upstream Formats | Route | Current Conversion Path |
|---------------|----------------------------|-------|--------------------------|
| Anthropic Messages | Anthropic / Chat | `/v1/messages` | Same-protocol passthrough / `anthropic_openai/` legacy channel |
| OpenAI Chat Completions | Chat / Responses | `/v1/chat/completions` | Same-protocol passthrough / `responses_chat/` legacy channel |
| OpenAI Responses API | Responses / Chat / Anthropic | `/v1/responses` | **IR channel** (`protocol/ir/`) |

All three request formats can route to any target format. `/v1/responses` already routes directly through the IR abstraction layer (`IRProxyStep`); the other two routes are still being migrated to the IR channel. Any single request between two protocols goes through exactly one IR conversion (`client → IR → upstream`), with no cascading conversion overhead.

> Migration target: also wire `/v1/messages` and `/v1/chat/completions` into `IRProxyStep`; after full cutover, delete the `anthropic_openai/` and `responses_chat/` legacy channels.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Admin Panel](#admin-panel)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

### Codex Desktop Support

LLM Proxy deeply integrates with Codex Desktop's OpenAI Responses API, with full tool conversion support:

- **apply_patch passthrough + DSL repair**: `apply_patch` is downgraded to a single function tool; upstream `arguments` are passed through verbatim as `custom_tool_call.input` to Codex. The return path runs [`repair_apply_patch_dsl`](llm_proxy/protocol/responses_chat/tool_replacement.py) to fix common DSL format issues (missing `*** Begin Patch` / `*** End Patch`, malformed `@@` hunk headers, wrong casing on `Add File` / `Update File` / `Delete File` / `Move to` keywords, etc.).
- **apply_patch tool description injection**: the server replaces the tool description with `APPLY_PATCH_TOOL_DESCRIPTION`, which explicitly enumerates the `***` marker prefix, the `@@`-on-its-own-line rule, the line-prefix rules (` ` / `- ` / `+ `), and the strict character-level context matching requirement — so the model avoids common pitfalls before writing.
- **Non-standard tool downgrade**: `namespace` is recursively flattened (sub-tools named with `__` to satisfy upstreams that enforce `^[a-zA-Z0-9_-]+$`), `web_search` is executed client-side, other `custom` tools are passed through
- **Think tag extraction**: `<think>` tags from upstream responses are extracted into reasoning/thinking blocks
- **SSE event type safety**: Every SSE data payload automatically injects the `type` field (mirroring the `event:` header).
  Codex only reads the `type` field from the data JSON, not the `event:` header; missing `type` causes
  "stream closed before response.completed" errors
- **Protocol conversion**: Automatic bidirectional conversion regardless of upstream format (Chat Completions or Anthropic Messages)

See [`config.toml.example`](config.toml.example) to configure Codex Desktop with this proxy.

### Other AI Clients

| Tool | Protocol | Route |
|------|----------|-------|
| Claude Code | Anthropic Messages API | `/v1/messages` |
| OpenCode / Any OpenAI-compatible client | OpenAI Chat Completions | `/v1/chat/completions` |
| Any OpenAI Responses-compatible client | OpenAI Responses API | `/v1/responses` |

Point your tool's `api_base` to this proxy. **One proxy serves all your tools.**

### Core Features

| Feature | Description |
|---------|-------------|
| **Full Protocol Interop** | Anthropic / Chat / Responses — any protocol routes to any upstream. `/v1/responses` already runs through the IR abstraction layer; the other two routes use legacy channels during the migration window |
| **Unified IR Abstraction Layer** | `protocol/ir/` — zero-dependency intermediate representation with ProtocolConverter registry pattern; adding a new protocol means implementing a single subclass |
| **Legacy Channels in Migration** | `anthropic_openai/` serves Anthropic cross-protocol; `responses_chat/` serves Chat→Responses conversion and apply_patch DSL repair |
| **Multi-Upstream Aggregation** | One proxy for DeepSeek, MiniMax, GLM (iFlytek), OpenCode, and more |
| **RTK (Input Compression / Beta)** | Built-in input compression tool (Rust Token Killer), strips CLI output noise from tool_results, truncates long code blocks, collapses blank lines |
| **Endpoint Authentication** | API-Key-based isolation, each endpoint independently configures available models |
| **Model Routing & Fallback** | Model family failover chain, auto-switch on 429/503 |
| **Request Tracking** | Unique Request ID per call, structured logging, web admin panel filtering |
| **Tool Format Compatibility** | apply_patch passthrough + DSL repair; namespace flattened; other custom tools passed through |
| **Admin Panel** | Preact + Vite web console for endpoint/model/usage/log management |

---

## Architecture

```
Request (Anthropic / Chat / Responses)
  │
  ▼
Handler Pipeline (Auth → ModelResolve → ProtocolSelect → ... → Proxy)
  │
  ├── Same protocol ──→ Direct passthrough to upstream
  │
  └── Cross-protocol ──→ Protocol conversion layer
                           │
                           ├── IR channel (IRProxyStep) ★ currently used by /v1/responses
                           │   └── client_body → IRRequest → upstream_body
                           │   └── upstream SSE → IRStreamEvent → client SSE
                           │
                           └── Legacy channel (ProxyStep) ★ /v1/messages and /v1/chat/completions during migration
                               ├── anthropic_openai: Anthropic ↔ Chat
                               └── responses_chat: Chat → Responses (includes apply_patch DSL repair)
```

The protocol conversion layer is built around `protocol/ir/`:

```
                    ┌─────────────┐
   Anthropic ──────→│             │──────→ Anthropic
   Chat     ──────→│  IR Layer   │──────→ Chat
   Responses──────→│ (dataclass) │──────→ Responses
                    └─────────────┘
```

Conversion flow: **Request direction** `client_body → to_ir() → IRRequest → to_upstream() → upstream_body`; **Response direction** `upstream_body → response_to_ir() → IRResponse → response_from_ir() → client_body`. IR types are pure dataclasses with zero external dependencies; protocol-specific fields pass through `extensions` dict.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy config template and fill in your API keys
cp config.example.json config.json
# vi config.json

# Start
bash start.sh
# or manually
python -m uvicorn llm_proxy.main:app --port 4000
```

Once running, access the admin panel at http://localhost:4000/static/.

### Usage Examples

**Anthropic format (cross-protocol to OpenAI Chat):**
```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

**OpenAI Chat format:**
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

**OpenAI Responses format (cross-protocol to Chat):**
```bash
curl http://localhost:4000/v1/responses \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-pro", "input": "Hello", "max_output_tokens": 100}'
```

---

## Deployment

### Docker (Recommended)

```bash
# Build frontend first
cd static && npm ci && npm run build && cd ..

# Start
docker-compose up -d
```

The Docker image includes a built-in health check (every 30s via `GET /api/config`), ready to go.

### macOS launchd

Create `~/Library/LaunchAgents/com.llmproxy.plist`:

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

Create `/etc/systemd/system/llm-proxy.service`:

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

## Configuration

Models are configured in `config.json`. Each model entry contains:

| Field | Description |
|-------|-------------|
| `api_base` | Upstream API URL (without `/v1/...` suffix) |
| `api_key` | Upstream API Key |
| `upstream_model` | Actual model name sent to upstream |
| `upstream_protocol` | Scalar field (legacy compat); prefer `upstream_protocols` array |
| `upstream_protocols` | Array like `["anthropic", "openai"]`, protocol selection is automatic via reachability table |
| `upstream_paths` | Protocol-specific upstream paths (optional) |
| `vision_support` | Whether image input is supported |

Endpoints (API key auth, model allowlist, family routing) are configured at runtime via the admin panel or API, stored in SQLite, and support hot reload.

---

## API Reference

### Proxy API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Anthropic Messages format (passthrough or cross-protocol) |
| POST | `/v1/chat/completions` | OpenAI Chat Completions format (passthrough or cross-protocol) |
| POST | `/v1/responses` | OpenAI Responses format (apply_patch passthrough + DSL repair; namespace/web_search downgrade) |
| POST | `/v1/messages/count_tokens` | Token counting |
| GET | `/v1/models` | Model list |

### Admin API

| Method | Path | Description |
|--------|------|-------------|
| GET/PUT | `/api/config` | Read/write configuration |
| GET/POST/PUT/DEL | `/api/endpoints` | Endpoint CRUD |
| GET | `/api/usage[?days=&group_by=&granularity=]` | Usage statistics |
| GET | `/api/logs/list` | Log query |
| POST | `/api/latency` | Latency test |
| POST | `/api/detect-protocol` | Detect upstream protocol |

---

## Admin Panel

Built with Preact + TypeScript + Vite, providing:

- Endpoint CRUD management
- Model usage overview and heatmap
- Request log filtering and viewing
- Latency testing
- Configuration import/export

### Frontend Development

```bash
cd static
npm install
npm run dev      # HMR dev mode
npm run build    # Production build
npm run test     # Run tests
```

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# IR abstraction layer tests only
python -m pytest tests/test_ir_conversions.py tests/test_ir_streaming.py -v

# Smoke test
python tests/smoke_test.py
```

---

## Project Structure

```
llm-proxy/
├── llm_proxy/
│   ├── main.py                    # FastAPI entry point
│   ├── state.py                   # Runtime state management
│   ├── config_loader.py           # Configuration loader
│   ├── logging_config.py          # Unified log format
│   ├── routes/                    # HTTP routes (thin layer)
│   ├── handlers/                  # Pipeline processing
│   │   ├── base.py                # PipelineContext + HandlerStep + Pipeline
│   │   ├── *_handler.py           # Route-specific pipeline assembly
│   │   └── shared/                # Reusable steps
│   │       ├── auth.py
│   │       ├── model_resolve.py
│   │       ├── protocol_select.py
│   │       ├── proxy.py           # Legacy ProxyStep (uses legacy channels)
│   │       └── ir_proxy.py        # New IRProxyStep (uses IR layer)
│   ├── protocol/                  # Protocol conversion layer
│   │   ├── capabilities.py        # Protocol reachability table + upstream selection
│   │   ├── ir/                    # ★ IR Abstraction Layer (new)
│   │   │   ├── __init__.py        #   ProtocolConverter base class + REGISTRY
│   │   │   ├── types.py           #   IRRequest/IRResponse/IRMessage/IRContentBlock
│   │   │   ├── _common.py         #   Shared utility functions
│   │   │   ├── _stream.py         #   Streaming utilities
│   │   │   ├── anthropic.py       #   Anthropic ↔ IR
│   │   │   ├── chat.py            #   Chat ↔ IR
│   │   │   └── responses.py       #   Responses ↔ IR
│   │   ├── anthropic_openai/      # Anthropic ↔ Chat (legacy, retained)
│   │   ├── responses_chat/        # Responses ↔ Chat (legacy, retained)
│   │   ├── errors.py              # Error formatting
│   │   ├── sse.py                 # SSE passthrough
│   │   └── think_tag.py           # Think tag detection
│   ├── services/                  # Business services
│   │   ├── tool_call_fix.py       # Tool call repair
│   │   ├── vision_service.py      # Image→text fallback
│   │   └── input_compressor.py    # RTK input compression
│   ├── infra/                     # Infrastructure layer
│   │   ├── db.py                  # SQLite operations
│   │   ├── http_client.py         # Global HTTP client
│   │   └── archive.py             # Usage recording
│   └── middleware/                # Middleware
│       ├── request_id.py
│       ├── access_log.py
│       └── catch_all_exceptions.py
├── static/                        # Frontend (Preact + TypeScript + Vite)
├── tests/                         # Tests
├── docs/                          # Documentation
├── config.example.json            # Configuration template
├── Dockerfile / docker-compose.yml
└── start.sh                       # Startup script
```

---

## License

**AGPL-3.0**

This software is released under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

In short: you are free to use, modify, and distribute this software, but **if you use it for commercial purposes (including but not limited to serving as a backend component of a commercial service), you must make the complete source code (including your modifications and the complete system it interacts with) available to all users under the same license.**

Core requirement: **Commercial use requires open source; closed-source commercial use is prohibited.**

