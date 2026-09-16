#!/bin/sh
# Restart the situation3d static server.
#
# Two details this script exists to handle:
#
#  * `pkill -f serve.py` run directly over ssh matches the ssh command line
#    itself (which contains the same literal) and kills the session before the
#    server can start, so the pattern lives in here instead of in the caller.
#  * The server is started as `python3 server/serve.py ...`, i.e. with a
#    relative path, so matching on the absolute path finds nothing and leaves
#    the old process holding the port.
#
# Usage: sh tools/restart_server.sh [port]
set -u

PORT="${1:-8848}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="/tmp/situation3d_serve.pid"
LOGFILE="/tmp/situation3d_serve.log"

if [ -f "$PIDFILE" ]; then
  OLD="$(cat "$PIDFILE" 2>/dev/null)"
  if [ -n "${OLD:-}" ]; then
    kill "$OLD" 2>/dev/null || true
  fi
fi

# Catch any server started before this pidfile existed. The pattern is split so
# it does not match this script's own command line.
PATTERN="server/ser""ve.py"
for pid in $(ps -eo pid,args 2>/dev/null | grep -F "$PATTERN" | grep -v grep | awk '{print $1}'); do
  [ "$pid" = "$$" ] && continue
  kill "$pid" 2>/dev/null || true
done

# Wait for the port to actually free up; binding too early is the failure mode
# this loop exists to prevent.
i=0
while [ "$i" -lt 10 ]; do
  if ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
    break
  fi
  sleep 1
  i=$((i + 1))
done

cd "$ROOT" || exit 1
setsid nohup python3 server/serve.py --port "$PORT" </dev/null >"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

# Probe until it answers or we give up.
i=0
CODE="000"
while [ "$i" -lt 15 ]; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/index.html" 2>/dev/null)"
  [ "$CODE" = "200" ] && break
  sleep 1
  i=$((i + 1))
done

echo "pid    : $(cat "$PIDFILE")"
echo "log    : $LOGFILE"
echo "probe  : HTTP $CODE"
if [ "$CODE" != "200" ]; then
  echo "--- log ---"
  cat "$LOGFILE" 2>/dev/null
  exit 1
fi
