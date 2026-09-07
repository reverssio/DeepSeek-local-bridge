#!/data/data/com.termux/files/usr/bin/bash
# Start the DeepSeek local API server (localhost only).
# - refuses to double-start (PID file + /healthz check)
# - takes a Termux wake-lock so Android doesn't freeze the server
#   while the screen is off (release with stop.sh)
# - verifies /healthz after launch
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/server.pid"
LOG="$DIR/logs/server.log"
URL="http://127.0.0.1:8000/healthz"

mkdir -p "$DIR/logs"

# already running?
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        if curl -s -m 5 "$URL" >/dev/null 2>&1; then
            echo "server already running (pid $PID) — http://127.0.0.1:8000"
            exit 0
        fi
        echo "pid $PID alive but /healthz failing; restarting..." >&2
        kill "$PID" 2>/dev/null
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

# stale process from an old start (no pidfile)? check by port
if curl -s -m 3 "$URL" >/dev/null 2>&1; then
    echo "another instance already answers on 127.0.0.1:8000 (no pidfile)."
    echo "find it:  ps -ef | grep app.py   then stop it and re-run start.sh"
    exit 1
fi

# wake-lock so background execution isn't frozen (CPU only, no display)
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock >/dev/null 2>&1

# optional: recover adb (for auth) — best effort, never fatal
python "$DIR/tools/adb_autoconnect.py" >/dev/null 2>&1 || true

cd "$DIR"
PYTHONPATH="$DIR" nohup "$DIR/.venv/bin/python" app.py >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

# wait for /healthz
for i in $(seq 1 20); do
    if curl -s -m 3 "$URL" >/dev/null 2>&1; then
        echo "server started (pid $PID) — http://127.0.0.1:8000"
        echo "log: $LOG"
        exit 0
    fi
    kill -0 "$PID" 2>/dev/null || { echo "server died at startup; last log lines:" >&2; tail -5 "$LOG" >&2; rm -f "$PIDFILE"; exit 1; }
    sleep 1
done
echo "server did not become healthy in 20s; check $LOG" >&2
exit 1
