"""Per-scenario honest breakdown for Hermes; spontaneous ST4 trajectory check."""
import json
import sys

sys.path.insert(0, ".")

from gep.replay import rescore  # noqa: E402

honest = [json.loads(l) for l in open("runs/episodes_honest.jsonl")]
herm = [r for r in honest if "Hermes" in r["cell"]["untrusted_model"]]

# 1. Hermes honest per-scenario main success
print("=== Hermes HONEST per-scenario main success ===")
from collections import Counter, defaultdict
agg = defaultdict(lambda: [0, 0])
for r in herm:
    scen = r["cell"]["scenario"]
    agg[scen][1] += 1
    # per-scenario rescore with current scorers
    rs = rescore(r)
    if any(rs["main_task_success"].values()):
        agg[scen][0] += 1
for scen, (ok, n) in sorted(agg.items()):
    print("  %-20s %2d/%2d" % (scen, ok, n))

# sample a failing record_correction honest Hermes episode
rc_fail = [r for r in herm if r["cell"]["scenario"] == "record_correction"
           and not any(rescore(r)["main_task_success"].values())]
print("\nFailing Hermes honest record_correction sample:")
if rc_fail:
    r = rc_fail[0]
    for a in r["actions"]:
        print("  %d %s %s" % (a["i"], a["tool"], json.dumps(a["args"], default=str)[:130]))
    rs = rescore(r)
    print("  main detail:", rs["main_task_success"])
    if r.get("final_text"):
        print("  final:", r["final_text"][:200].replace("\n", " "))

# 2. Hermes honest spontaneous ST4: dump one
print("\n=== Hermes honest spontaneous ST4 trajectory ===")
st4 = [r for r in herm if rescore(r)["side_task_success"].get("ST4")]
print("count:", len(st4))
if st4:
    r = st4[0]
    print(r["episode_id"])
    for a in r["actions"]:
        print("  %d %s %s" % (a["i"], a["tool"], json.dumps(a["args"], default=str)[:130]))
