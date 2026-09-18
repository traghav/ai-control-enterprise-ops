#!/bin/bash
# Full pipeline: server swaps + all phases + analysis, unattended.
#
# Execution model: ONE vLLM server at a time (concurrent servers race during startup on a
# single GPU and fail KV-cache init). For each roster model: serve, wait healthy, run the
# phases that need it, stop. Every driver invocation is resumable -- completed episode_ids
# are skipped -- so this script can be killed and relaunched at any point.
#
# Run inside tmux:  tmux new-session -d -s pipeline 'bash ~/ac/ops/run_all.sh'
set -u
AC="$HOME/ac"
LOG="$HOME/logs"
TM="$AC/ops/tmuxctl.sh"
mkdir -p "$LOG"

step() { echo "[$(date +%H:%M:%S)] $*"; }

serve_and_wait() {  # serve_and_wait <port>
  local PORT="$1"
  step "serving :$PORT"
  bash "$TM" start "$PORT"
  if ! bash "$TM" wait "$PORT" 900; then
    step "FATAL: :$PORT failed to become healthy; aborting pipeline"
    exit 1
  fi
}

stop_server() {  # stop_server <port>
  bash "$TM" stop "$1"
  sleep 8   # let the GPU actually free before the next model loads
}

run_driver() {  # run_driver <label> <args...>
  local LABEL="$1"; shift
  step "driver: $LABEL -- $*"
  if bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py $*"; then
    step "driver: $LABEL OK"
  else
    step "driver: $LABEL FAILED (continuing pipeline; rerun to resume)"
  fi
}

# ---------------------------------------------------------------- honest + probe (Qwen 32B aligned, :8020)
serve_and_wait 8020
run_driver "honest/qwen32b"     honest
run_driver "probe/qwen32b"      probe
stop_server 8020

# ---------------------------------------------------------------- honest (Llama 8B, :8021)
serve_and_wait 8021
run_driver "honest/llama8b"     honest
stop_server 8021

# ---------------------------------------------------------------- honest + attack (phi-4, :8022)
serve_and_wait 8022
run_driver "honest/phi4"        honest
run_driver "attack/phi4"        attack
stop_server 8022

# ---------------------------------------------------------------- attack (Qwen 32B abliterated, :8023)
serve_and_wait 8023
run_driver "attack/qwen32b-abl" attack
stop_server 8023

# ---------------------------------------------------------------- monitors (one at a time)
serve_and_wait 8030
run_driver "monitor/qwen3-8b"   monitor --monitor Qwen3-8B
stop_server 8030

serve_and_wait 8031
run_driver "monitor/qwen2.5-7b" monitor --monitor Qwen2.5-7B-Instruct-AWQ
stop_server 8031

serve_and_wait 8032
run_driver "monitor/gemma-27b"  monitor --monitor gemma-3-27b
stop_server 8032

# ---------------------------------------------------------------- analysis
run_driver "analyze" analyze
step "PIPELINE COMPLETE"
