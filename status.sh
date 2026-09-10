#!/bin/sh
# Quick status of the local DeepSeek API server.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/server.pid"

PID=""
[ -f "$PIDFILE" ] && PID="$(cat "$PIDFILE" 2>/dev/null)"

if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "server: RUNNING (pid $PID)"
else
    PID="$(ps -ef | grep "[a]pp.py" | awk '{print $2}' | head -1)"
    if [ -n "$PID" ]; then
        echo "server: RUNNING (pid $PID, no pidfile)"
    else
        echo "server: STOPPED"
    fi
fi

if curl -s -m 3 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
    echo "healthz:  OK"
else
    echo "healthz:  FAIL (server not answering on 127.0.0.1:8000)"
fi

# session status (never prints the token itself)
if [ -f "$DIR/session/session.json" ]; then
    (cd "$DIR" && "$DIR/.venv/bin/python" - <<'EOF'
import json, time, os
from pathlib import Path
try:
    sess_file = Path("session/session.json")
    d = json.loads(sess_file.read_text())
    age_h = (time.time() - d.get("captured_at", 0)) / 3600
    status_str = f"ACTIVE ({age_h:.1f}h old)" if age_h < 72 else f"OLD ({age_h:.1f}h old, may need refresh)"
    print(f"session:  present ({len(d.get('cookies', {}))} cookies, {status_str})")
except Exception as e:
    print(f"session:  unreadable ({type(e).__name__})")
EOF
    )
else
    echo "session:  MISSING (run: python -m deepseek.auth after adb connect)"
fi

# conversation map (session-reuse state)
if [ -f "$DIR/session/conv_map.json" ]; then
    (cd "$DIR" && "$DIR/.venv/bin/python" - <<'EOF'
import json
from pathlib import Path
try:
    map_file = Path("session/conv_map.json")
    m = json.loads(map_file.read_text())
    convs = {r["conversation_id"].split(":")[0] for r in m.values()}
    print(f"convmap:  {len(m)} mapped OpenCode sessions -> {len(convs)} DeepSeek conversations")
except Exception as e:
    print(f"convmap:  unreadable ({type(e).__name__})")
EOF
    )
else
    echo "convmap:  (empty — first request will create it)"
fi

if command -v adb >/dev/null 2>&1; then
    TAB="$(printf '\t')"
    DEVS="$(adb devices | grep -c "${TAB}device")"
    if [ "$DEVS" -gt 0 ]; then
        echo "adb:      connected ($DEVS device)"
    else
        echo "adb:      disconnected (only needed for login/session refresh)"
    fi
fi
exit 0
