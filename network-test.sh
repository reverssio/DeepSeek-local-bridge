#!/bin/sh
# Network diagnostics for the local DeepSeek API setup.
# Safe output: never prints tokens, cookies, or auth headers.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL="http://127.0.0.1:8000"

line() { echo "----- $1 -----"; }

line "1. local server"
PID="$(ps -ef | grep "[a]pp.py" | awk '{print $2}' | head -1)"
if [ -n "$PID" ]; then echo "running, pid $PID"; else echo "NOT RUNNING (start with $DIR/start.sh)"; fi

line "2. loopback reachability"
if python - <<'EOF' 2>/dev/null
import socket
s = socket.socket(); s.settimeout(2)
try:
    s.connect(("127.0.0.1", 8000)); raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
EOF
then echo "127.0.0.1:8000 reachable"; else echo "127.0.0.1:8000 NOT reachable"; fi

line "3. /healthz"
if curl -s -m 5 "$LOCAL/healthz" | grep -q ok; then echo OK; else echo "FAIL"; fi

line "4. /v1/models"
if curl -s -m 5 "$LOCAL/v1/models" | grep -q deepseek-chat; then echo OK; else echo "FAIL"; fi

line "5. DNS resolution"
if python - <<'EOF' 2>/dev/null
import socket, sys
try:
    socket.getaddrinfo("chat.deepseek.com", 443, proto=socket.IPPROTO_TCP)
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
EOF
then echo "chat.deepseek.com resolves OK"; else echo "DNS for chat.deepseek.com FAILED (check network / private DNS settings)"; fi

line "6. general DNS"
if python - <<'EOF' 2>/dev/null
import socket
try:
    socket.getaddrinfo("www.google.com", 443, proto=socket.IPPROTO_TCP)
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
EOF
then echo "general DNS OK"; else echo "general DNS FAILED"; fi

line "7. internet (HTTPS)"
code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' https://www.google.com/generate_204 2>/dev/null)"
if [ "$code" = "204" ]; then echo "internet OK (google 204)"; else echo "internet FAIL (google: $code)"; fi

line "8. DeepSeek reachability"
code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' https://chat.deepseek.com/ 2>/dev/null)"
case "$code" in
  000) echo "chat.deepseek.com UNREACHABLE (no route / no internet)";;
  403) echo "chat.deepseek.com reachable (403 WAF bot-check is EXPECTED for curl)";;
  *)   echo "chat.deepseek.com reachable (HTTP $code)";;
esac

line "9. network interfaces (Termux view)"
ifconfig 2>/dev/null | grep -E "^[a-z0-9]+:|inet " | sed 's/^/  /' || echo "  (ifconfig unavailable)"

line "10. adb (needed only for login/session refresh)"
python "$DIR/tools/adb_autoconnect.py" 2>&1 | sed 's/^/  /'

line "11. upstream API auth check (no secrets printed)"
if [ -f "$DIR/session/session.json" ]; then
    (cd "$DIR" && "$DIR/.venv/bin/python" - <<'EOF'
import json, sys, time, os
from pathlib import Path
sys.path.insert(0, os.getcwd())
try:
    from deepseek.client import DeepSeekClient
    c = DeepSeekClient(allow_interactive=False)
    # cheapest authenticated call: create + discard a chat session
    sid = c.create_chat_session()
    print(f"  upstream session creation OK (chat_session {sid[:4]}...)")
except Exception as e:
    msg = str(e)
    if "login" in msg.lower() or "No DeepSeek session" in msg:
        print("  auth: NO VALID SESSION — run: python -m deepseek.auth (needs adb+chrome)")
    elif "timed out" in msg.lower() or "connect" in msg.lower() or "ReadTimeout" in type(e).__name__:
        print("  auth: session exists but DeepSeek unreachable (network problem)")
    else:
        print(f"  auth: upstream call failed ({type(e).__name__}; first 120 chars: {msg[:120]})")
EOF
else
    echo "  no session.json — login not done yet"
fi

echo
echo "done. (server URL: $LOCAL — bind 127.0.0.1 only)"
