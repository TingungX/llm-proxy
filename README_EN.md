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

---

## Table of Contents

- [Features](#features)
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

### Full Support for Mainstream AI Coding Tools

LLM Proxy is fully compatible with the communication protocols used by mainstream AI coding tools, acting as a transparent gateway to non-native models:

| Tool | Protocol | Route | Notes |
|------|----------|-------|-------|
| **Codex Desktop** | OpenAI Responses API | `/v1/responses` | Auto-converts Responses ↔ Chat Completions, full tool conversion support |
| **Claude Code** | Anthropic Messages API | `/v1/messages` | Pass-through or convert to OpenAI Chat for upstream |
| **OpenCode** | OpenAI Chat Completions | `/v1/chat/completions` | Pass-through or convert to Anthropic for upstream |
| Any OpenAI-compatible client | OpenAI Chat Completions | `/v1/chat/completions` | Standard OpenAI proxy, zero config |

Just point your tool's `api_base` to this proxy. **One proxy serves all your tools.**

See [`config.toml.example`](config.toml.example) (Codex Desktop) and [`config.example.json`](config.example.json) (configuration template) to get started.

### Core Features

| Feature | Description |
|---------|-------------|
| **Multi-Protocol Bidirectional Conversion** | Anthropic Messages ↔ OpenAI Chat Completions (full bidirectional), OpenAI Responses → Chat Completions |
| **Multi-Upstream Aggregation** | One proxy for DeepSeek, MiniMax, GLM (iFlytek), OpenCode, and more |
| **Built-in RTK** | Real-time usage statistics and request logging with hourly/daily aggregation and heatmap visualization |
| **Endpoint Authentication** | API-Key-based isolation, each endpoint independently configures available models |
| **Model Routing & Fallback** | When enabled, model family failover chain, auto-switch on 429/503 |
| **Request Tracking** | Unique Request ID per call, structured logging, web admin panel filtering |
| **Tool Format Compatibility** | Auto-conversion for namespace, apply_patch, and other non-standard tool types |
| **Admin Panel** | Preact + Vite web console for endpoint/model/usage/log management |

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

**Anthropic format:**
```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

**OpenAI format:**
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "x-api-key: your-endpoint-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
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

The Dockerfile includes a built-in health check (every 30s via `GET /api/config`), ready to go.

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
| `upstream_protocol` | `anthropic` or `openai`, auto-detected when empty |
| `upstream_paths` | Protocol-specific upstream paths (optional) |

Endpoints (API key auth, model allowlist, family routing) are configured at runtime via the admin panel or API, stored in SQLite, and support hot reload.

---

## API Reference

### Proxy API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Anthropic Messages format |
| POST | `/v1/chat/completions` | OpenAI Chat Completions format |
| POST | `/v1/responses` | OpenAI Responses format (converted to Chat) |

All requests require `x-api-key` or `Authorization: Bearer` header.

### Admin API

| Method | Path | Description |
|--------|------|-------------|
| GET/PUT | `/api/config` | Read/write configuration |
| GET/POST/PUT/DEL | `/api/endpoints` | Endpoint CRUD |
| GET | `/api/usage[?days=&group_by=&granularity=]` | Usage statistics |
| GET | `/api/logs` | Log query |
| POST | `/api/latency` | Latency test |

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
python -m pytest tests/ -v
python tests/smoke_test.py
```

---

## Project Structure

```
llm-proxy/
├── llm_proxy/           # Python backend
│   ├── main.py          # FastAPI entry point
│   ├── state.py         # Runtime state management
│   ├── config_loader.py # Configuration loader
│   ├── routes/          # HTTP routes (thin layer)
│   ├── handlers/        # Pipeline processing
│   ├── protocol/        # Bidirectional protocol conversion (pure Python)
│   ├── services/        # Business services
│   ├── infra/           # Infrastructure (SQLite, HTTP client)
│   └── middleware/      # Middleware (request_id, access_log)
├── static/              # Frontend (Preact + TypeScript + Vite)
├── tests/               # Tests
├── docs/                # Documentation
├── config.example.json  # Configuration template
├── Dockerfile
├── docker-compose.yml
└── start.sh / proxy.py  # Startup scripts
```

---

## License

**AGPL-3.0**

This software is released under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

In short: you are free to use, modify, and distribute this software, but **if you use it for commercial purposes (including but not limited to serving as a backend component of a commercial service), you must make the complete source code (including your modifications and the complete system it interacts with) available to all users under the same license.**

Core requirement: **Commercial use requires open source; closed-source commercial use is prohibited.**
