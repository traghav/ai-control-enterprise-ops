#!/bin/bash
# E3 sequence: selective attack leg (Qwen-abl x FULL) -> all 3 monitors over new episodes
# -> final analysis.
set -u
AC="$HOME/ac"
LOG="$HOME/logs"
TM="$AC/ops/tmuxctl.sh"
step() { echo "[$(date +%H:%M:%S)] $*"; }

serve_and_wait() {
  local PORT="$1"
  step "serving :$PORT"
  bash "$TM" start "$PORT"
  if ! bash "$TM" wait "$PORT" 900; then
    step ":$PORT not healthy; retrying once after 20s"
    bash "$TM" stop "$PORT"; sleep 20
    bash "$TM" start "$PORT"
    if ! bash "$TM" wait "$PORT" 900; then
      step "FATAL: :$PORT failed after retry"; exit 1
    fi
  fi
}

serve_and_wait 8023
step "selective/qwen32b-abl FULL"
if bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py selective"; then
  step "selective/qwen32b-abl OK"
else
  step "selective/qwen32b-abl FAILED"
fi
bash "$TM" stop 8023; sleep 15

for M in 8030:Qwen3-8B 8031:Qwen2.5-7B-Instruct-AWQ 8032:gemma-3-27b; do
  PORT="${M%%:*}"; NAME="${M#*:}"
  serve_and_wait "$PORT"
  step "monitor/$NAME over new episodes"
  bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py monitor --monitor $NAME" \
    || step "monitor/$NAME FAILED"
  bash "$TM" stop "$PORT"; sleep 15
done

step "final analyze + figures"
bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py analyze" || step "analyze FAILED"
bash -c "cd $AC && source $AC/.venv/bin/activate && python analysis/figures.py" || step "figures FAILED"
step "E3 SEQUENCE COMPLETE"
