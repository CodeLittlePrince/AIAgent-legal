#!/usr/bin/env bash
# wait-for-it.sh — wait until host:port accepts TCP connections

set -e

HOST=""
PORT=""
TIMEOUT=60

usage() {
  echo "Usage: $0 host:port [-t timeout_seconds]"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

TARGET="$1"
shift

if [[ "$TARGET" != *:* ]]; then
  echo "Error: expected host:port, got '$TARGET'"
  usage
fi

HOST="${TARGET%%:*}"
PORT="${TARGET##*:}"

while [ $# -gt 0 ]; do
  case "$1" in
    -t)
      TIMEOUT="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

echo "Waiting for $HOST:$PORT (timeout ${TIMEOUT}s)..."

start=$(date +%s)

while true; do
  if (echo >"/dev/tcp/$HOST/$PORT") >/dev/null 2>&1; then
    echo "$HOST:$PORT is available."
    exit 0
  fi

  now=$(date +%s)
  elapsed=$((now - start))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "Timeout after ${TIMEOUT}s waiting for $HOST:$PORT"
    exit 1
  fi

  sleep 1
done
