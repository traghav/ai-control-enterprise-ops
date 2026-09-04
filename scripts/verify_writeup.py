"""Pre-writeup verification: honest-leg quality, spontaneous violations, probe reality."""
import json
import sys

sys.path.insert(0, ".")

from gep.replay import rescore  # noqa: E402

eps = {}
for ph in ("honest", "attack", "probe"):
    eps[ph] = [json.loads(l) for l in open(f"runs/episodes_{ph}.jsonl")]

def short(m):
    return m.split("/")[-1][:34]

# 1. Per-model honest usefulness + error rate + zero-action rate
print("=== HONEST legs quality ===")
for r in eps["honest"]:
    pass
from collections import Counter, defaultdict
agg = defaultdict(lambda: {"n": 0, "main": 0, "err": 0, "zero_act": 0})
for r in eps["honest"]:
    a = agg[r["cell"]["untrusted_model"]]
    a["n"] += 1
    a["main"] += 1 if any(r["main_task_success"].values()) else 0
    a["err"] += 1 if r.get("error") else 0
    a["zero_act"] += 1 if len(r.get("actions", [])) == 0 else 0
for m, a in sorted(agg.items()):
    print("  %-36s n=%3d main=%.2f err=%d zero_act=%d" % (
        short(m), a["n"], a["main"] / a["n"], a["err"], a["zero_act"]))

# Hermes honest: what do failures look like?
herm_h = [r for r in eps["honest"] if "Hermes" in r["cell"]["untrusted_model"]]
n_err = sum(1 for r in herm_h if r.get("error"))
print("\n  Hermes honest error sample:", (herm_h[0].get("error") or "none")[:160] if herm_h else "-")
print("  Hermes honest final_text sample:", (herm_h[0].get("final_text") or "")[:200].replace("\n", " "))

# 2. Spontaneous violations in honest mode (side scorers firing without attack prompt)
print("\n=== SPONTANEOUS side-task fires in HONEST mode (rescored, current scorers) ===")
for m in sorted({r["cell"]["untrusted_model"] for r in eps["honest"]}):
    fires = Counter()
    n = 0
    for r in eps["honest"]:
        if r["cell"]["untrusted_model"] != m:
            continue
        n += 1
        rs = rescore(r)
        for k, v in rs["side_task_success"].items():
            if v:
                fires[k] += 1
    print("  %-36s n=%3d fires=%s" % (short(m), n, dict(fires) if fires else "none"))

# 3. Probe (aligned model + attack prompt): are the side fires real violations?
print("\n=== PROBE side fires, one full trajectory dump ===")
probe_fires = [r for r in eps["probe"]
               if rescore(r)["side_task_success"].get("ST1") or
               rescore(r)["side_task_success"].get("ST3")]
print("  probe episodes with ST1/ST3 fires:", len(probe_fires))
if probe_fires:
    r = probe_fires[0]
    print(" ", r["episode_id"])
    for a in r["actions"]:
        print("    %d %s %s" % (a["i"], a["tool"], json.dumps(a["args"], default=str)[:120]))

# 4. Attack-mode per-side-task breakdown by model x affordance
print("\n=== ATTACK per-side-task breakdown (current scorers) ===")
for m in sorted({r["cell"]["untrusted_model"] for r in eps["attack"]}):
    for aff in ("FULL", "RESTRICTED"):
        fires = Counter()
        n = 0
        for r in eps["attack"]:
            if r["cell"]["untrusted_model"] != m or r["cell"]["affordances"] != aff:
                continue
            n += 1
            rs = rescore(r)
            for k, v in rs["side_task_success"].items():
                if v:
                    fires[k] += 1
        anyf = sum(1 for r in eps["attack"]
                   if r["cell"]["untrusted_model"] == m and r["cell"]["affordances"] == aff
                   and any(rescore(r)["side_task_success"].values()))
        print("  %-36s %-10s n=%2d any=%2d %s" % (short(m), aff, n, anyf, dict(fires)))
