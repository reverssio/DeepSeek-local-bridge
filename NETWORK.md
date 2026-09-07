# Network Reliability: Wi-Fi / Mobile Data Switching

Verified behavior of this setup on Android 15 / Termux (non-rooted), for the
architecture:

    OpenCode ── localhost:8000 ──> DeepSeek bridge ── HTTPS ──> chat.deepseek.com

## Why localhost survives network switches

- The server binds **127.0.0.1 only** (AF_INET loopback). The loopback
  interface (`lo`) is always UP regardless of Wi-Fi/mobile-data state — it is
  not attached to any radio. Verified: `ifconfig lo` stays
  `UP,LOOPBACK,RUNNING` across network changes; the server kept serving
  through an actual Wi-Fi subnet change (192.168.1.x → 192.168.50.x) during
  testing without restart.
- `localhost` resolves to `127.0.0.1` only on this device (no IPv6 `::1`
  mismatch for Node/OpenCode).
- No configuration anywhere references a LAN IP. Verified by grep across
  the project + opencode.json: the only address OpenCode uses is
  `http://127.0.0.1:8000/v1`.

**Conclusion: OpenCode ↔ bridge communication is completely unaffected by
Wi-Fi ↔ mobile-data switching.** This was verified empirically, not assumed.

## What a switch DOES affect

1. **The bridge's own upstream HTTPS connections to chat.deepseek.com.**
   Android switches the default network cleanly; NEW sockets re-route
   automatically. But already-established (pooled keep-alive) TCP sockets die
   silently — no FIN arrives — so an in-flight request stalls until read
   timeout. Fixes applied (see "Patches"):
     - pooled connections expire after 15s idle instead of living forever
     - 2 transport-level connect retries
     - single bounded re-request with a fresh PoW header when a stream dies
       before any output was delivered
     - read timeout 120s (was 300s) so a dead stream is noticed sooner

2. **Chrome/CDP (only used for login & 6-hourly session refresh).**
   - The wireless-adb port rotates every time Wireless debugging restarts or
     the phone changes network. `tools/adb_autoconnect.py` re-discovers it
     via mDNS (`_adb-tls-connect._tcp.local`) — no pairing, no manual IP.
   - Chrome itself can wedge DNS after a network change (observed once on a
     captive-portal Wi-Fi: all tabs DNS_PROBE_POSSIBLE while Termux resolved
     fine). If that happens, force-stop Chrome and reopen it — the bridge
     doesn't depend on Chrome for normal requests, only for auth refresh.

3. **Session/token validity across IP changes.**
   DeepSeek bearer tokens are not IP-pinned (standard for web sessions);
   after a full Wi-Fi → mobile-data switch the same token keeps working.
   If DeepSeek ever invalidates it, the bridge returns HTTP 503
   `login_required` with instructions (never a silent failure), and
   `network-test.sh` section 11 confirms the exact state.

## Files / scripts

- `start.sh`   — starts server on 127.0.0.1:8000 (refuses duplicates, takes
                 wake-lock, verifies /healthz, best-effort adb recovery)
- `stop.sh`    — stops it, releases wake-lock
- `status.sh`  — running?, healthz, session freshness, adb state
- `network-test.sh` — full diagnostics (server, loopback, DNS, internet,
                 DeepSeek reachability, interfaces, adb, upstream auth)
- `tools/adb_autoconnect.py` — mDNS adb reconnect
- `tools/watch_login.py`    — background login watcher (captures session
                 once you finish signing in inside Chrome yourself)

## Recovery cheat-sheet

| Symptom | Do this |
|---|---|
| OpenCode says connection refused | `~/deepseek-api/start.sh` |
| 502 network_error from bridge | check mobile data/Wi-Fi is actually up (`network-test.sh`) |
| 503 login_required | `adb` reconnect (script does it), then `cd ~/deepseek-api && .venv/bin/python -m deepseek.auth` — sign in with your account in Chrome when it opens |
| Chrome tabs all DNS-fail | force-stop Chrome, reopen (Android/Chrome DNS wedge) |
| Server frozen after switch | `./stop.sh && ./start.sh` (patches make this rare) |

## Manual test still recommended (cannot be automated from Termux)

Toggling radios needs the quick-settings panel:

1. With Wi-Fi on: `~/deepseek-api/network-test.sh` → all green.
2. Turn Wi-Fi OFF (mobile data on): run `network-test.sh` again → sections
   1-4 (localhost) must stay green; 5-8 (internet) should also be green.
   Ask OpenCode something to confirm a live completion.
3. Turn Wi-Fi back ON: repeat. Nothing to reconfigure at any point.

Termux battery note: Android may kill Termux under memory pressure or
Deep Sleep even with a wake-lock; if that happens, just reopen Termux and
run `~/deepseek-api/start.sh`. The session survives (token valid ~6h, then
headless refresh via Chrome; full re-login only if DeepSeek invalidated it).
