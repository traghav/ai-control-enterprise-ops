#!/bin/bash
# Post-Gemma sequence: Lexi attack leg -> all monitors over new episodes -> analysis.
# Chained unattended because each step depends on the previous one's server state.
set -u
AC="$HOME/ac"
LOG="$HOME/logs"
TM="$AC/ops/tmuxctl.sh"

step() { echo "[$(date +%H:%M:%S)] $*"; }

# 0. wait for the gemma_mon session to finish (it frees :8032 when done)
step "waiting for gemma_mon session to finish"
while tmux has-session -t gemma_mon 2>/dev/null; do sleep 20; done
step "gemma monitor done: $(wc -l < $AC/runs/scores__gemma-3-27b-it-int4-awq_informed.jsonl) scores"

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

# 1. Lexi attack leg
serve_and_wait 8024
step "attack/lexi8b"
if bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py attack"; then
  step "attack/lexi8b OK"
else
  step "attack/lexi8b FAILED"
fi
bash "$TM" stop 8024; sleep 15

# 2. monitors over the new Lexi episodes (resumable: only unscored trajectories run)
for M in 8030:Qwen3-8B 8031:Qwen2.5-7B-Instruct-AWQ 8032:gemma-3-27b; do
  PORT="${M%%:*}"; NAME="${M#*:}"
  serve_and_wait "$PORT"
  step "monitor/$NAME over new episodes"
  bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py monitor --monitor $NAME" \
    || step "monitor/$NAME FAILED"
  bash "$TM" stop "$PORT"; sleep 15
done

# 3. analysis
step "analyze"
bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py analyze" || step "analyze FAILED"
step "FINISH SEQUENCE COMPLETE"
