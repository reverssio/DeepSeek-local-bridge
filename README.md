# DeepSeek Local Bridge

OpenAI-compatible local API server and client library bridging the DeepSeek Web interface to developer tooling, autonomous coding agents, and OpenAI SDK integrations.

---

## Overview

DeepSeek Local Bridge turns a personal DeepSeek Web account into a local API endpoint operating at `http://127.0.0.1:8000/v1`. It translates standard OpenAI chat completion requests into DeepSeek Web API protocol transactions, enabling local tools and AI agents to utilize DeepSeek models with full support for:

- Streaming responses and DeepThink reasoning traces (`reasoning_content`)
- Structured tool and function calling across multiple markup dialects
- Genuine multimodal image analysis from local files, data URIs, and remote URLs
- Strict 1-to-1 session affinity and upstream conversation persistence
- Fast, local, zero-overhead conversation title generation for OpenCode
- Automatic proof-of-work (PoW) challenge resolution via WebAssembly
- Multi-tier authentication recovery and headless session management

> **Disclaimer**: This is an independent, open-source project. It is not affiliated with, endorsed by, or sponsored by DeepSeek. It automates web interface transactions for personal development use. Please use responsibly and in accordance with DeepSeek's Terms of Service.

---

## Table of Contents

- [Supported Environments](#supported-environments)
- [Architecture](#architecture)
- [Prerequisites and Requirements](#prerequisites-and-requirements)
- [Installation](#installation)
- [Authentication and Setup](#authentication-and-setup)
  - [Desktop Environments (Linux / macOS / Windows)](#desktop-environments-linux--macos--windows)
  - [Android and Termux (Non-Rooted)](#android-and-termux-non-rooted)
- [Running the Server](#running-the-server)
- [Verification and Health Checks](#verification-and-health-checks)
- [OpenCode Integration](#opencode-integration)
- [API Usage Examples](#api-usage-examples)
  - [cURL](#curl)
  - [Python OpenAI SDK](#python-openai-sdk)
  - [Direct Python Library](#direct-python-library)
- [Multimodal Vision Support](#multimodal-vision-support)
- [Session Management and Upstream Conversation Reuse](#session-management-and-upstream-conversation-reuse)
- [Configuration Reference](#configuration-reference)
- [Project Layout](#project-layout)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Supported Environments

| Environment | Status | Verification Details |
|---|---|---|
| **Android / Termux (aarch64)** | Fully Supported & Verified | Tested on native Termux (non-rooted) with Android 10+ via Wireless Debugging (ADB CDP). |
| **Linux (x86_64, aarch64)** | Fully Supported | Tested with Playwright Chromium and Python 3.9–3.14. |
| **macOS (Apple Silicon, Intel)** | Compatible | Supports Playwright Chromium and standard Python 3.9+. |
| **Windows (x64)** | Compatible | Supports PowerShell and CMD via Playwright Chromium. |

---

## Architecture

```
[ OpenCode / OpenAI Client ]
            │  POST /v1/chat/completions (OpenAI Schema)
            ▼
┌────────────────────────────────────────────────────────┐
│ DeepSeek Local Bridge (FastAPI @ localhost:8000)       │
├────────────────────────────────────────────────────────┤
│ • Rate Limiter & Concurrency Manager                  │
│ • Session Affinity (OpenCode ID ──> DeepSeek Chat ID)  │
│ • Local Title Generator (<1ms, 0 upstream cost)        │
│ • Multimodal Ingestion (Path / DataURI / URL)          │
│ • Tool Tag & Nonce Binding Bridge                     │
└────────────────────────────────────────────────────────┘
            │  chat.deepseek.com internal HTTPS / SSE
            ▼
┌────────────────────────────────────────────────────────┐
│ DeepSeek Web Services                                  │
├────────────────────────────────────────────────────────┤
│ • PoW Challenge Solver (WASM in background threadpool) │
│ • File Upload & Polling (/api/v0/file/upload_file)     │
│ • SSE Fragment Stream Parser (THINK vs RESPONSE)       │
└────────────────────────────────────────────────────────┘
```

---

## Prerequisites and Requirements

- **Python**: Version 3.9 or higher (tested up to Python 3.14).
- **DeepSeek Account**: Standard personal account registered at [chat.deepseek.com](https://chat.deepseek.com).
- **Network Access**: Outbound HTTPS connectivity to `https://chat.deepseek.com`.
- **Environment-Specific Tools**:
  - *Desktop*: Chromium browser (installed automatically via `playwright`).
  - *Android / Termux*: Google Chrome installed on the device, Android Wireless Debugging enabled, and `android-tools` installed in Termux.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/reverssio/DeepSeek-local-bridge.git
cd DeepSeek-local-bridge
```

### 2. Create and Activate Virtual Environment

On Linux, macOS, or Termux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Authentication and Setup

The bridge communicates with DeepSeek Web using a captured session token and cookies. This session is refreshed automatically once acquired and saved in `session/session.json` (excluded from git).

### Desktop Environments (Linux / macOS / Windows)

1. Install Playwright browser dependencies:
   ```bash
   playwright install chromium
   ```
2. Run the interactive login utility:
   ```bash
   python -m deepseek.auth
   ```
3. A browser window will open. Log in to your DeepSeek account and solve any human verification puzzle. Once logged in, the utility captures the session credentials to `session/session.json` and closes the browser.

### Android and Termux (Non-Rooted)

On Android devices, Chrome runs in a sandboxed application space. Termux connects to Chrome via Android's built-in Chrome Developer Protocol (CDP) through Wireless Debugging:

1. Install required packages in Termux:
   ```bash
   pkg install -y android-tools curl
   ```
2. Enable Developer Options on your Android device:
   - Go to **Settings > About Phone** and tap **Build Number** 7 times.
   - Go to **Settings > System > Developer Options**.
   - Enable **Wireless Debugging**.
3. Pair and connect ADB locally in Termux:
   - Open Wireless Debugging, tap **Pair device with pairing code**. Note the IP, Port, and 6-digit code.
   - In Termux:
     ```bash
     adb pair 127.0.0.1:<pairing_port> <pairing_code>
     adb connect 127.0.0.1:<connect_port>
     ```
4. Start Chrome with remote debugging enabled:
   - Close all existing Chrome tabs.
   - Launch Chrome from Termux with CDP enabled:
     ```bash
     am start -n com.android.chrome/com.google.android.apps.chrome.Main -d "https://chat.deepseek.com" --es "args" "--remote-debugging-port=9222"
     ```
5. Forward the CDP port to Termux:
   ```bash
   adb forward tcp:9222 localabstract:chrome_devtools_remote
   ```
6. Run the Android auth capture script:
   ```bash
   python -m deepseek.auth_android
   ```
   Log in to DeepSeek in the opened Chrome tab. The script will detect your active login, extract the bearer token and security cookies, and save them to `session/session.json`.

---

## Running the Server

### Starting the Daemon

On Linux or Termux:

```bash
./start.sh
```

Or run directly with Python:

```bash
python app.py
```

`./start.sh` automatically:
- Acquires a Termux CPU wake-lock (preventing Android background suspension)
- Reconnects ADB if needed via mDNS
- Starts `uvicorn` bound to `127.0.0.1:8000`
- Verifies server health at `http://127.0.0.1:8000/healthz`
- Stores the background process ID in `server.pid` and writes logs to `logs/server.log`

### Checking Status

```bash
./status.sh
```

Outputs:
- Server execution state (PID, health status)
- Session freshness and cookie count
- Count of active mapped conversations
- Local ADB connection status

### Stopping the Server

```bash
./stop.sh
```

Terminates background server instances and releases any active Termux wake-lock.

---

## Verification and Health Checks

1. Verify endpoint reachability:
   ```bash
   curl http://127.0.0.1:8000/healthz
   # Output: {"status":"ok"}
   ```

2. Verify model catalog:
   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

3. Run the automated network diagnostics suite:
   ```bash
   ./network-test.sh
   ```

---

## OpenCode Integration

DeepSeek Local Bridge is designed for direct pairing with [OpenCode](https://github.com/anomalyco/opencode) as a local OpenAI-compatible provider.

Add the following provider configuration to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "deepseek-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DeepSeek (Local Bridge)",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1"
      },
      "models": {
        "deepseek-chat": {
          "name": "DeepSeek Chat (Instant)",
          "limit": {
            "context": 64000,
            "output": 8000
          },
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "reasoning": true,
          "interleaved": {
            "field": "reasoning_content"
          },
          "variants": {
            "default": {},
            "reasoning": {
              "reasoningEffort": "medium"
            }
          }
        },
        "deepseek-expert": {
          "name": "DeepSeek Expert",
          "limit": {
            "context": 64000,
            "output": 8000
          },
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "reasoning": true,
          "interleaved": {
            "field": "reasoning_content"
          },
          "variants": {
            "default": {},
            "reasoning": {
              "reasoningEffort": "medium"
            }
          }
        }
      }
    }
  },
  "agent": {
    "title": {}
  }
}
```

### Key OpenCode Features Handled by the Bridge

- **Session Affinity**: The bridge reads OpenCode's `x-session-affinity` / `session-id` headers and maintains a strictly persistent upstream DeepSeek Web conversation for each OpenCode thread.
- **Local Title Generation**: OpenCode's background title requests (`agent="title"`) are intercepted and resolved locally in $<100\text{ ms}$ without creating throwaway conversations upstream.
- **Tool Calling**: Agent tools (`bash`, `edit`, `read`, `write`, `grep`, `glob`, etc.) are converted to structured tool tags with cryptographic nonces preventing model hallucination.

---

## API Usage Examples

### cURL

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement in two sentences."}
    ],
    "stream": false
  }'
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

# Streamed completion with DeepThink reasoning
response = client.chat.completions.create(
    model="deepseek-expert",
    messages=[
        {"role": "user", "content": "Write a Python function to compute Fibonacci numbers."}
    ],
    stream=True,
    extra_body={"thinking": True}
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(f"[Thinking] {delta.reasoning_content}", end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Direct Python Library

```python
from deepseek.client import DeepSeekClient
from deepseek.auth import get_session

client = DeepSeekClient(get_session(allow_interactive=False))

# Simple multi-turn chat
reply = client.chat("Remember the number 42.")
print(reply.text)

follow_up = client.chat("What number did I tell you to remember?", conversation_id=reply.conversation_id)
print(follow_up.text)
```

---

## Multimodal Vision Support

The bridge includes an autonomous image pipeline that allows DeepSeek models to analyze image pixels directly:

1. **Local Files**: Detects local file paths matching common formats (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.bmp`).
   - Allowed directories: `/storage/emulated/0`, `/data/data/com.termux/files`, `/sdcard`, user home directory, and current working directory.
   - Configurable via `MULTIMODAL_ALLOWED_ROOTS` in `.env`.
2. **Data URIs**: Ingests base64-encoded image payloads (`data:image/jpeg;base64,...`).
3. **Remote URLs**: Downloads and validates external images with SSRF protection against private IP spaces (RFC 1918, loopback, link-local).
4. **Upstream Upload**: Automatically negotiates the DeepSeek Web upload protocol (`/api/v0/file/upload_file`) with dedicated PoW solving, polls processing status, and caches uploaded SHA-256 hashes to prevent redundant uploads.
5. **Session Continuity**: Images can be attached on any turn of an ongoing conversation without destroying session continuity.

---

## Session Management and Upstream Conversation Reuse

The bridge maps external client sessions to native DeepSeek conversations:

- **1-to-1 Mapping**: Each OpenCode session ID maintains exactly one upstream conversation on `chat.deepseek.com`.
- **True Upstream Threading**: Captures `response_message_id` from SSE frames and correctly chains messages using `parent_message_id`.
- **Per-Session Concurrency Lock**: Prevents race conditions when parallel requests arrive for the same session.
- **Stale Session Recovery**: If a user deletes an active conversation in the DeepSeek Web interface, the bridge detects the invalid upstream session, invalidates the local mapping, and transparently initializes a replacement thread with full history replay.

---

## Configuration Reference

Settings can be specified in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Network interface to bind the API server. |
| `PORT` | `8000` | Port for the local API server. |
| `RATE_LIMIT_PER_MINUTE` | `120` | Maximum requests per minute allowed per client IP. |
| `SERVER_INTERACTIVE_LOGIN` | `true` | If true, launches a browser when authentication is missing. |
| `DEEPSEEK_PROFILE_DIR` | `session/profile` | Optional path to a persistent Chrome user data profile. |
| `MULTIMODAL_ALLOWED_ROOTS` | None | Colon/semicolon-separated list of additional paths permitted for local image access. |

---

## Project Layout

```
deepseek-local-bridge/
├── app.py                      # Server launcher script
├── start.sh                    # Linux/Termux background daemon starter
├── status.sh                   # Server and session status inspector
├── stop.sh                     # Daemon stop script
├── network-test.sh             # Diagnostic script for connectivity & DNS
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── deepseek/
│   ├── auth.py                 # Desktop Playwright authentication module
│   ├── auth_android.py         # Android / Termux ADB CDP authentication module
│   ├── client.py               # Pure-HTTP DeepSeek chat client & PoW manager
│   ├── multimodal.py           # Image ingestion, SSRF validation & upload pipeline
│   ├── pow.py                  # WebAssembly PoW challenge solver
│   └── sse.py                  # SSE streaming parser (THINK vs RESPONSE)
├── server/
│   ├── api.py                  # FastAPI OpenAI-compatible routing (/v1/chat/completions)
│   ├── config.py               # Configuration parser & model mapping
│   ├── openai_format.py        # OpenAI request/response formatting
│   ├── ratelimit.py            # Token bucket rate limiting middleware
│   ├── schemas.py              # Pydantic schemas for OpenAI API validation
│   ├── sessions.py             # Persistent session mapping & concurrency locks
│   ├── title_generator.py      # Local high-performance title generator
│   └── tools_bridge.py         # Structured tool calling & nonce tag parser
└── tools/
    └── adb_autoconnect.py      # mDNS auto-discovery for Android Wireless Debugging
```

---

## Security Considerations

- **Credential Isolation**: Bearer tokens and cookies are stored only on the local machine in `session/session.json`. They are never committed, logged, or exposed across the network.
- **Localhost Binding**: By default, the server binds exclusively to `127.0.0.1`. Do not bind to public interfaces without adding authentication and reverse-proxy TLS termination.
- **SSRF Hardening**: The multimodal pipeline resolves hostnames via DNS and blocks requests targeting internal IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `169.254.0.0/16`).
- **Filesystem Traversal Protection**: Local image access is restricted to verified path roots; attempts to read system directories (e.g. `/etc`, `/proc`) raise `PermissionError`.

---

## Troubleshooting

### Server Returns 503 `login_required`
- Your captured DeepSeek session has expired or was revoked.
- Re-run authentication:
  - Desktop: `python -m deepseek.auth`
  - Android/Termux: `python -m deepseek.auth_android`

### DNS Resolution Failures on Android
- If `network-test.sh` shows DNS resolution failure for `chat.deepseek.com`, verify that Android Private DNS is not blocking the hostname, or test toggling Wi-Fi / mobile data.

### Tool Call Parse Errors
- If the model emits unrecognized tool formatting, verify that the calling client provides schemas adhering to standard OpenAI tool definition formats. The bridge automatically translates between `<tool>`, DSML, and JSON schemas.

---

## Known Limitations

- **Immutable Model per Thread**: DeepSeek Web threads fix their model type (`default` vs `expert`) at creation. Switching models within the same OpenCode session triggers the creation of a new upstream conversation.
- **Web Protocol Dependency**: Relies on internal endpoints of `chat.deepseek.com`. Significant changes to upstream Cloudflare or WebAssembly challenges may require library updates.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
