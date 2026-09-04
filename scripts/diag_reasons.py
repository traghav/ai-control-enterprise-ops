"""Check monitor reasons for saturated honest ROI episodes (prefix-ambiguity mechanism)."""
import json
import sys

sys.path.insert(0, ".")

slug = "Qwen3-8B-AWQ"
rows = [json.loads(l) for l in open(f"runs/scores__{slug}_informed.jsonl")]

print("=== honest ROI episodes: peak-action monitor reasoning ===")
n = 0
for r in rows:
    eid = r["episode_id"]
    if eid.startswith("honest|") and "roi_withhold" in eid and "abliterated" in eid:
        sc = r["scores"]
        hi = max(range(len(sc)), key=lambda i: sc[i])
        print("[%s] peak=%.2f of action %d/%d" % (
            eid[-28:], r["max"], hi, len(sc)))
        print("   reason:", (r["reasons"][hi] or "")[:220])
        n += 1
        if n >= 4:
            break

print()
print("=== successful ST2 (silent edit) at low suspicion: monitor reasoning ===")
for r in rows:
    eid = r["episode_id"]
    if eid.startswith("attack|") and "ledger_reconcile" in eid \
            and "RESTRICTED" in eid and r["max"] <= 0.25 and "abliterated" in eid:
        print("[%s] max=%.2f" % (eid[-40:], r["max"]))
        for i, (s, reason) in enumerate(zip(r["scores"], r["reasons"])):
            print("   a%d %.2f %s" % (i, s, (reason or "")[:150]))
        break

print()
print("=== honest distribution by scenario (Qwen-abl, Qwen3-8B monitor) ===")
by_scen = {}
for r in rows:
    eid = r["episode_id"]
    if not (eid.startswith("honest|") and "abliterated" in eid):
        continue
    scen = eid.split("|")[3]
    by_scen.setdefault(scen, []).append(r["max"])
for scen, vals in sorted(by_scen.items()):
    vals.sort()
    print("  %-20s n=%2d median=%.2f max=%.2f  top5=%s" % (
        scen, len(vals), vals[len(vals)//2], vals[-1], vals[-5:]))
