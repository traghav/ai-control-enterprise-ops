#!/bin/bash
# GENERATED from serve.py::ROSTER by ops/gen_ops.py -- do not edit by hand.
#
# Serves exactly one roster model in the FOREGROUND so that tmux owns its lifetime.
# Foreground is deliberate: nothing here uses pkill, because `pkill -f` matches the
# invoking shell's own command line and has repeatedly killed the session that
# called it.
set -u
PORT="${1:?usage: serve_one.sh <port>}"
# The venv's bin/ MUST be on PATH. vLLM shells out to `ninja` during CUDA-graph
# capture; invoking the venv's `vllm` by absolute path does not put `ninja` on PATH,
# and the engine then dies with "FileNotFoundError: ninja" -- which surfaces only
# as a generic "Engine core initialization failed" in the API server log.
export PATH="/home/ubuntu/venv-vllm/bin:$PATH"
export HF_HOME=/home/ubuntu/hf_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
case "$PORT" in
  8020) MODEL='Qwen/Qwen2.5-32B-Instruct-AWQ'; UTIL=0.85; MAXLEN=16384; EXTRA=(--enable-auto-tool-choice --tool-call-parser hermes) ;;
  8021) MODEL='NousResearch/Meta-Llama-3.1-8B-Instruct'; UTIL=0.7; MAXLEN=16384; EXTRA=(--enable-auto-tool-choice --tool-call-parser llama3_json) ;;
  8022) MODEL='stelterlab/phi-4-AWQ'; UTIL=0.7; MAXLEN=16384; EXTRA=() ;;
  8023) MODEL='ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ'; UTIL=0.85; MAXLEN=16384; EXTRA=(--enable-auto-tool-choice --tool-call-parser hermes) ;;
  8024) MODEL='NousResearch/Hermes-3-Llama-3.1-8B'; UTIL=0.7; MAXLEN=16384; EXTRA=(--enable-auto-tool-choice --tool-call-parser hermes) ;;
  8030) MODEL='Qwen/Qwen3-8B-AWQ'; UTIL=0.6; MAXLEN=16384; EXTRA=() ;;
  8031) MODEL='Qwen/Qwen2.5-7B-Instruct-AWQ'; UTIL=0.6; MAXLEN=16384; EXTRA=() ;;
  8032) MODEL='gaunernst/gemma-3-27b-it-int4-awq'; UTIL=0.8; MAXLEN=16384; EXTRA=() ;;
  *) echo "no roster entry for port $PORT" >&2; exit 2 ;;
esac
mkdir -p /home/ubuntu/logs
echo "[serve_one] port=$PORT model=$MODEL util=$UTIL maxlen=$MAXLEN"
# NOTE the empty-array expansion idiom: "${EXTRA[@]:-}" would pass a literal
# empty-string argument when EXTRA=(), which vLLM rejects with
# "error: unrecognized arguments:" (observed on phi-4, the only no-flag model).
exec /home/ubuntu/venv-vllm/bin/vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --no-enable-log-requests \
  --no-enable-prefix-caching \
  ${EXTRA[@]+"${EXTRA[@]}"}
