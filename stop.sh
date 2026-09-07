#!/data/data/com.termux/files/usr/bin/bash
# Stop the DeepSeek local API server and release the wake-lock.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/server.pid"

if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID" && echo "server stopped (pid $PID)"
    else
        echo "pid $PID not running (stale pidfile removed)"
    fi
    rm -f "$PIDFILE"
else
    # no pidfile: try to find it by command line
    PID="$(ps -ef | grep "[a]pp.py" | awk '{print $2}' | head -1)"
    if [ -n "$PID" ]; then
        kill "$PID" && echo "server stopped (pid $PID, found by ps)"
    else
        echo "no running server found"
    fi
fi

command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock >/dev/null 2>&1
exit 0
