#!/bin/bash
# D12 recovery: purge junk legs -> same-checkpoint honest legs (Qwen-abl :8023,
# Hermes-3 :8024) -> all 3 monitors over new episodes -> analyze.
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
    step "driver: $label FAILED"
  fi
}

# 0. purge junk episodes (phi-4 honest+probe, NousResearch-Llama honest) and their scores
step "purging junk legs"
python3 - <<PY
import json
RUNS = "/home/ubuntu/ac/runs"
JUNK_MODELS = ("phi-4", "Meta-Llama-3.1-8B-Instruct")   # aligned llama, NOT Hermes-3
purged, kept = 0, 0
for ph in ("honest", "probe", "attack"):
    path = f"{RUNS}/episodes_{ph}.jsonl"
    recs = [json.loads(l) for l in open(path)]
    out = []
    for r in recs:
        m = r["cell"]["untrusted_model"]
        if any(j in m for j in JUNK_MODELS):
            purged += 1
        else:
            out.append(r)
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  {ph}: kept {len(out)}, purged {len(recs)-len(out)}")
# purge orphaned score rows
import glob
for f in glob.glob(f"{RUNS}/scores__*.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    keep = [r for r in rows if not any(
        j in r["episode_id"] for j in JUNK_MODELS)]
    with open(f, "w") as fh:
        for r in keep:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"  {f.split('/')[-1]}: kept {len(keep)}")
PY

# 1. honest legs for the attack checkpoints (same-checkpoint baselines)
serve_and_wait 8023
run_driver "honest/qwen32b-abl" honest
bash "$TM" stop 8023; sleep 15

serve_and_wait 8024
run_driver "honest/hermes8b" honest
bash "$TM" stop 8024; sleep 15

# 2. all three monitors over the new honest episodes (resumable)
for M in 8030:Qwen3-8B 8031:Qwen2.5-7B-Instruct-AWQ 8032:gemma-3-27b; do
  PORT="${M%%:*}"; NAME="${M#*:}"
  serve_and_wait "$PORT"
  step "monitor/$NAME"
  bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py monitor --monitor $NAME" \
    || step "monitor/$NAME FAILED"
  bash "$TM" stop "$PORT"; sleep 15
done

# 3. analysis
step "analyze"
bash -c "cd $AC && source $AC/.venv/bin/activate && python run_matrix.py analyze" || step "analyze FAILED"
step "D12 RECOVERY COMPLETE"
