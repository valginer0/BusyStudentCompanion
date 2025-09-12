#!/usr/bin/env bash
set -euo pipefail

NAME_FILTER="busystudentcompanion-app-cpu-1"
CID="$(docker ps --filter "name=${NAME_FILTER}" -q)"

if [ -z "$CID" ]; then
  echo "Container not found: ${NAME_FILTER}"
  docker ps
  exit 1
fi

echo "Container ID: $CID"

echo "=== Top CPU processes in container (best effort) ==="
if docker exec "$CID" sh -c 'command -v ps >/dev/null 2>&1'; then
  docker exec "$CID" sh -c 'ps -eo pid,ppid,comm,%cpu,%mem --sort=-%cpu | head -n 15'
else
  echo "procps (ps) not installed in container; showing python processes via /proc"
  docker exec "$CID" sh -lc 'for p in /proc/[0-9]*; do
    pid=$(basename "$p");
    [ -r "$p/comm" ] && comm=$(cat "$p/comm" 2>/dev/null || true);
    case "$comm" in
      python*|streamlit*)
        echo "PID $pid COMMAND $comm";
        ;;
    esac
  done'
fi

echo "=== Cache sizes ==="
docker exec "$CID" sh -c 'du -sh /cache/huggingface 2>/dev/null || true; du -sh /cache/torch 2>/dev/null || true; du -sh /app/cache 2>/dev/null || true'

echo "=== App directory check ==="
docker exec "$CID" sh -c 'ls -l /app/src/book_to_essay | sed -n "1,60p"'

echo "=== Recent logs (last 200 lines) ==="
docker logs --tail=200 "$CID"