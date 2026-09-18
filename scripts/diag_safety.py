"""Ad-hoc diagnostic: honest FP tail vs successful-attack suspicion (Qwen-abl cells).

Usage (on the GPU box):  python scripts/diag_safety.py
"""
import json
import sys

sys.path.insert(0, ".")

from gep.replay import rescore  # noqa: E402

# monitor repo id -> score-file slug (the filename uses the repo basename)
MONITOR_KEYS = {
    "Qwen/Qwen3-8B-AWQ": "Qwen3-8B-AWQ",
    "Qwen/Qwen2.5-7B-Instruct-AWQ": "Qwen2.5-7B-Instruct-AWQ",
    "gaunernst/gemma-3-27b-it-int4-awq": "gemma-3-27b-it-int4-awq",
}

scores = {}
for slug in MONITOR_KEYS.values():
    for line in open(f"runs/scores__{slug}_informed.jsonl"):
        r = json.loads(line)
        scores.setdefault(r["episode_id"], {})[slug] = r["max"]


def maxs(eid: str, mon: str) -> float:
    return scores.get(eid, {}).get(MONITOR_KEYS[mon], 0.0)


honest = [json.loads(l) for l in open("runs/episodes_honest.jsonl")]
attack = [json.loads(l) for l in open("runs/episodes_attack.jsonl")]
honest = [r for r in honest if "abliterated" in r["cell"]["untrusted_model"]]
attack = [r for r in attack if "abliterated" in r["cell"]["untrusted_model"]]

for mon in MONITOR_KEYS:
    print(f"===== monitor {mon} =====")
    hs = sorted(((maxs(r["episode_id"], mon), r) for r in honest), key=lambda t: -t[0])
    print("HONEST top-8 max_suspicion:")
    for s, r in hs[:8]:
        n_act = len(r["actions"])
        ok_main = any(r["main_task_success"].values())
        tail = r["episode_id"][-34:]
        print("  %.2f main=%s n_act=%d %s" % (s, ok_main, n_act, tail))
    vals = sorted(s for s, _ in hs)
    print("  honest: median=%.3f p90=%.3f max=%.3f  n=%d" %
          (vals[len(vals) // 2], vals[int(0.9 * len(vals))], vals[-1], len(vals)))

    succ = []
    for r in attack:
        sc = maxs(r["episode_id"], mon)
        rs = rescore(r)
        if any(rs["side_task_success"].values()):
            fired = [k for k, v in rs["side_task_success"].items() if v]
            succ.append((sc, fired, r["episode_id"][-34:], r["cell"]["affordances"]))
    succ.sort()
    print("successful attacks sorted by suspicion (lowest 8 = the safety risk):")
    for sc, fired, eid, aff in succ[:8]:
        print("  %.2f %-10s %s %s" % (sc, aff, fired, eid))
    print("  n_successful_attacks:", len(succ))
    print()
