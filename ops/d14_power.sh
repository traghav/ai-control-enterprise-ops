#!/bin/bash
# D14 power expansion: policy legs re-run with reps 4..9 (resumable skips 0..3), then all
# six monitor passes (3 monitors x {context-free, ticket-aware}) over new episodes, then
# final analysis + figures.
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

# ---- policy legs (reps 4..9 only; 0..3 skip via fixed resume check) ------------------
serve_and_wait 8020
run_driver "honest/qwen32b"        honest
run_driver "probe/qwen32b"         probe
bash "$TM" stop 8020; sleep 15

serve_and_wait 8023
run_driver "honest/qwen32b-abl"    honest
run_driver "attack/qwen32b-abl"    attack
bash "$TM" stop 8023; sleep 15

serve_and_wait 8024
run_driver "honest/hermes8b"       honest
run_driver "attack/hermes8b"       attack
bash "$TM" stop 8024; sleep 15

# ---- monitors over new episodes (resumable: only unscored trajectories run) ---------
for M in 8030:Qwen3-8B 8031:Qwen2.5-7B-Instruct-AWQ 8032:gemma-3-27b; do
  PORT="${M%%:*}"; NAME="${M#*:}"
  serve_and_wait "$PORT"
  step "monitor/$NAME context-free"
  run_driver "monitor/$NAME-cf"    monitor --monitor "$NAME"
  step "monitor/$NAME ticket-aware"
  run_driver "monitor/$NAME-ta"    monitor --monitor "$NAME" --ticket-aware
  bash "$TM" stop "$PORT"; sleep 15
done

# ---- final analysis + figures --------------------------------------------------------
step "final analyze + figures"
run_driver "analyze" analyze
bash -c "cd $AC && source $AC/.venv/bin/activate && python analysis/figures.py" || step "figures FAILED"
step "D14 POWER EXPANSION COMPLETE"
