# Android/Termux Port Notes — sums001/Deepseek-API

Date: 2026-09-07
Device: Android 15, aarch64, non-rooted, native Termux (NOT PRoot)
Python: 3.14.6 (Termux system python; NOT replaced)

## Verdict
The repository's architecture is sound for Android once the auth layer is
ported. Components:

| Component | Status on Termux | Notes |
|---|---|---|
| A. Python runtime (3.14) | OK | system python used as-is |
| B. HTTP client (httpx) | OK | pure python wheel installs fine |
| C. FastAPI/OpenAI server | OK | needed pydantic v1 + fastapi pin (see below) |
| D. PoW/WASM (wasmtime) | OK | official Android wheel exists; retag needed |
| E. auth/session acquisition | PORTED | Playwright replaced with CDP-over-ADB (auth_android.py) |
| F. browser automation | N/A on Android | desktop Playwright cannot ship Chromium to Termux; used real Chrome via CDP instead |
| G. OpenCode integration | OK | separate provider entry, nothing overwritten |

## Dependency changes vs upstream requirements.txt

Upstream pins: playwright>=1.44, httpx>=0.27, fastapi>=0.111,
uvicorn[standard]>=0.30, pydantic>=2.7, python-dotenv>=1.0, wasmtime>=21.0.

Installed instead (in ~/deepseek-api/.venv):

- httpx 0.28.1 — unchanged
- fastapi 0.115.14 — pinned: newer versions hard-require pydantic v2
- pydantic 1.10.26 — pydantic-core v2 has no Android wheel and building needs
  rust/maturin, which does not support the aarch64-unknown-linux-android
  triple from Termux. pydantic v1 is pure Python and satisfies FastAPI
  0.115.x. The server's schemas only use plain BaseModel + Optional fields,
  which pydantic v1 fully supports (server code unchanged).
- uvicorn 0.35.0 — pinned: `uvicorn[standard]` extras (uvloop, httptools)
  don't build on Android/bionic. Plain uvicorn + its bundled h11 works.
  websockets 17.1 (pure python) also installed for the CDP client.
- python-dotenv 1.2.3 — unchanged
- wasmtime 48.0.0 — upstream wants >=21.0; 48 satisfies it. NOTE: PyPI
  publishes an official `py3-none-android_26_arm64_v8a` wheel but pip's
  compatible-tag list on this Termux only includes android_24 and below
  (platform tag reporting), so pip refuses the android_26 wheel and falls
  back to the pure-python `py3-none-any` wheel that only bundles
  win32-x86_64 binaries. Fix: downloaded the android_26 wheel and rewrote
  its WHEEL tag to android_24 (device is API 35; the binary is fully
  compatible — only the tag name matters to pip). The native
  `_libwasmtime.so` loads and runs DeepSeek's sha3_wasm_bg.wasm PoW
  solver correctly on aarch64/bionic.
- playwright — NOT INSTALLED. Replaced by `deepseek/auth_android.py` which
  speaks the Chrome DevTools Protocol to the phone's REAL Chrome via
  `adb forward tcp:17025 localabstract:chrome_devtools_remote_<pid>`. This
  is the same legitimate Android automation mechanism Playwright documents
  (https://playwright.dev/docs/android), minus the desktop-only
  bundled-Chromium download. The user signs in with their own account in
  their own Chrome; the code only observes the resulting session token the
  way upstream's Playwright code does.

## Source modifications to upstream files

1. deepseek/auth.py — guarded imports: on Termux (sys.getandroidapilevel
   exists) the module imports Session/LoginRequired/get_session/login from
   auth_android instead of playwright, and re-binds those names at the end
   of the module. On non-Android platforms the file behaves exactly like
   upstream.
2. deepseek/auth_android.py — NEW file (the CDP/ADB auth implementation,
   mirrors upstream's public auth API).
No other upstream files were modified.

## Session handling on Android

Upstream keeps a persistent Chromium profile under session/profile. On
Android the "profile" is the real Chrome install — if you stay signed in to
chat.deepseek.com in Chrome, the headless refresh path just re-reads the
token from Chrome via CDP. session/session.json (token+cookies) is
git-ignored (already in upstream .gitignore).

## Known limitations

- Rate limits / serialization: the server serializes requests through one
  DeepSeek account; the PoW store isn't reentrant (upstream design).
- OpenAI params temperature/top_p/max_tokens accepted but ignored
  (upstream behavior).
- usage token counts are ~4-chars/token estimates (upstream behavior).
- No vision, no native tool calling via this bridge (upstream limitation).
