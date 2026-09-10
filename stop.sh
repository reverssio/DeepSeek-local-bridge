#!/bin/sh
# Stop the DeepSeek local API server and release the wake-lock.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/server.pid"

stopped=0
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID" && stopped=1
    fi
    rm -f "$PIDFILE"
fi
# catch ANY strays (e.g. two starts happened)
for PID in $(ps -ef | grep "[a]pp\.py" | awk '{print $2}'); do
    kill "$PID" 2>/dev/null && stopped=1
done
sleep 1
# force if needed
for PID in $(ps -ef | grep "[a]pp\.py" | awk '{print $2}'); do
    kill -9 "$PID" 2>/dev/null
done
[ "$stopped" = "1" ] && echo "server stopped" || echo "no running server found"
command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock >/dev/null 2>&1
exit 0
