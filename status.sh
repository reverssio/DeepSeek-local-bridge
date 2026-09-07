#!/data/data/com.termux/files/usr/bin/bash
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
    "$DIR/.venv/bin/python" - <<'EOF'
import json, time
try:
    d = json.load(open("/data/data/com.termux/files/home/deepseek-api/session/session.json"))
    age_h = (time.time() - d.get("captured_at", 0)) / 3600
    fresh = "FRESH" if age_h < 6 else f"STALE ({age_h:.1f}h old, needs refresh)"
    print(f"session:  present ({len(d.get('cookies', {}))} cookies, {fresh})")
except Exception as e:
    print(f"session:  unreadable ({type(e).__name__})")
EOF
else
    echo "session:  MISSING (run: python -m deepseek.auth after adb connect)"
fi

if command -v adb >/dev/null 2>&1; then
    DEVS="$(adb devices | grep -c $'\tdevice')"
    if [ "$DEVS" -gt 0 ]; then
        echo "adb:      connected ($DEVS device)"
    else
        echo "adb:      disconnected (only needed for login/session refresh)"
    fi
fi
exit 0
