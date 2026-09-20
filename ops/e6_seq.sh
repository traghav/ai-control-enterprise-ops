#!/bin/bash
# E6 sequence: revived aligned-Llama legs (honest + probe, serialized history) ->
# all 6 monitor passes over new episodes -> analysis + figures.
# Everything is resumable: completed episode ids are skipped.
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

run_driver() {
  local LABEL="$1"; shift
  step "driver: $LABEL -- $*"
  if bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py $*"; then
    step "driver: $LABEL OK"
  else
    step "driver: $LABEL FAILED"
  fi
}

# ---- aligned-Llama legs (revived: official tool template + serialized history) ------
serve_and_wait 8021
run_driver "honest/llama8b" honest
run_driver "probe/llama8b"  probe
bash "$TM" stop 8021; sleep 15

# ---- monitors over new episodes (3 monitors x {cf, ticket-aware}) -------------------
for M in 8030:Qwen3-8B 8031:Qwen2.5-7B-Instruct-AWQ 8032:gemma-3-27b; do
  PORT="${M%%:*}"; NAME="${M#*:}"
  serve_and_wait "$PORT"
  step "monitor/$NAME context-free"
  run_driver "monitor/$NAME-cf" monitor --monitor "$NAME"
  step "monitor/$NAME ticket-aware"
  run_driver "monitor/$NAME-ta" monitor --monitor "$NAME" --ticket-aware
  bash "$TM" stop "$PORT"; sleep 15
done

# ---- analysis + figures ---------------------------------------------------------------
step "analyze + figures"
run_driver "analyze" analyze
bash -c "cd $AC && source $AC/.venv/bin/activate && python analysis/figures.py" || step "figures FAILED"
step "E6 SEQUENCE COMPLETE"
