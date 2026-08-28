#!/bin/bash
# tmux lifecycle manager for vLLM servers and long-running jobs.
#
# WHY TMUX AND NOT pkill: `pkill -f "<pattern>"` matches against the full command line of
# every process, INCLUDING the shell that is running the pkill -- when the pattern is part
# of the command string (e.g. an ssh one-liner that mentions "vllm serve"), pkill kills its
# own parent shell. That failure mode cost several iterations and presents as "the model
# server mysteriously never starts". tmux sessions give us a named handle, so stopping is
# `tmux kill-session -t <name>` and never matches unrelated processes.
#
# Servers run in the FOREGROUND inside their tmux window, so the session's lifetime is
# exactly the server's lifetime.
#
# Usage:
#   tmuxctl.sh start <port> [...]   start servers for the given roster ports
#   tmuxctl.sh stop  <port> [...]   stop those servers
#   tmuxctl.sh status               print health for every port in the roster
#   tmuxctl.sh job <name> <cmd...>  run an arbitrary long job in a named session
#   tmuxctl.sh ls                   list sessions
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSTER_ENTRIES="${ROSTER_ENTRIES:-8020 8021 8022 8023 8024 8030 8031 8032}"

sess_name() { echo "vllm-$1"; }

cmd_start() {
  for PORT in "$@"; do
    local NAME; NAME="$(sess_name "$PORT")"
    if [ -n "$(tmux ls 2>/dev/null | grep "^${NAME}:")" ]; then
      echo "[tmuxctl] $NAME already exists"
      continue
    fi
    tmux new-session -d -s "$NAME" "bash $HERE/serve_one.sh $PORT 2>&1 | tee /home/ubuntu/logs/vllm_${PORT}.log"
    echo "[tmuxctl] started $NAME"
    sleep 2
  done
}

cmd_stop() {
  for PORT in "$@"; do
    local NAME; NAME="$(sess_name "$PORT")"
    tmux kill-session -t "$NAME" 2>/dev/null && echo "[tmuxctl] stopped $NAME" || echo "[tmuxctl] $NAME not running"
  done
}

cmd_status() {
  printf '%-8s %-10s %-8s %s\n' PORT HEALTH GPU_MIB MODEL
  for PORT in $ROSTER_ENTRIES; do
    local OUT; OUT="$(curl -s -m 4 "http://localhost:$PORT/v1/models" 2>/dev/null)"
    if [ -n "$OUT" ]; then
      local M; M="$(echo "$OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
      printf '%-8s %-10s %-8s %s\n' "$PORT" "UP" "-" "$M"
    else
      printf '%-8s %-10s %-8s %s\n' "$PORT" "down" "-" "-"
    fi
  done
  echo "gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"
}

cmd_wait() {
  # wait <port> <timeout_s>
  local PORT="$1" TIMEOUT="${2:-900}" t0 now
  t0=$(date +%s)
  while :; do
    if curl -sf -m 4 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      now=$(date +%s); echo "[tmuxctl] :$PORT ready after $((now-t0))s"; return 0
    fi
    now=$(date +%s)
    if [ $((now-t0)) -gt "$TIMEOUT" ]; then
      echo "[tmuxctl] :$PORT TIMEOUT after ${TIMEOUT}s; last log lines:"; tail -15 "/home/ubuntu/logs/vllm_${PORT}.log" 2>/dev/null; return 1
    fi
    sleep 5
  done
}

cmd_job() {
  local NAME="$1"; shift
  if [ -n "$(tmux ls 2>/dev/null | grep "^${NAME}:")" ]; then
    echo "[tmuxctl] session $NAME already exists; use a different name"; return 1
  fi
  tmux new-session -d -s "$NAME" "$*"
  echo "[tmuxctl] started job $NAME"
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  wait)   shift; cmd_wait "$@" ;;
  status) cmd_status ;;
  job)    shift; cmd_job "$@" ;;
  ls)     tmux ls ;;
  *) echo "usage: $0 {start|stop|wait|status|job|ls} ..." >&2; exit 2 ;;
esac
