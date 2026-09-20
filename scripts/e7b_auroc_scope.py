"""E7-pre part 2: reconcile the two AUROC scopes.

SCOPE ALL:   positives = ALL attack episodes (the earlier transfer-collapse number).
SCOPE FIRED: positives = successful attacks only (the operationally relevant set:
             episodes the defer protocol must catch).
If the collapse persists on FIRED but not on ALL, the collapse is driven by
unsuccessful attacks that look honest -- which is operationally FINE (they do not
threaten) and the earlier framing needs correcting. If it persists on FIRED too,
the monitors genuinely miss completed violations on this policy.
"""
import json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from gep.replay import rescore
from gep.defer import find_crossing

SLUGS = {
    "Qwen3-8B": "Qwen3-8B-AWQ_informed",
    "Qwen2.5-7B": "Qwen2.5-7B-Instruct-AWQ_informed",
    "Gemma-27B": "gemma-3-27b-it-int4-awq_informed",
}
POLICIES = {
    "Qwen-abl": "ibrahimkettaneh",
    "Hermes-3": "NousResearch/Hermes-3",
    "aligned-Qwen-probe": "Qwen/Qwen2.5-32B-Instruct-AWQ|probe|",
    "aligned-Llama-probe": "NousResearch/Meta-Llama-3.1-8B-Instruct|probe|",
}

scores = {}
for slug in set(SLUGS.values()):
    for line in open(f"runs/scores__{slug}.jsonl"):
        r = json.loads(line)
        scores.setdefault(slug, {})[r["episode_id"]] = r["max"]

episodes = {}
for ph in ("honest", "attack", "probe"):
    for line in open(f"runs/episodes_{ph}.jsonl"):
        r = json.loads(line)
        episodes[r["episode_id"]] = (ph, r)


def auroc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j+1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rp = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rp - n1*(n1+1)/2.0) / (n1*n0) if n1 and n0 else float("nan")


# cache rescored fired-status per attack episode
fired_cache = {}
for eid, (ph, rec) in episodes.items():
    if ph == "attack":
        fired_cache[eid] = any(rescore(rec)["side_task_success"].values())

print("FULL affordances, context-free monitors. neg = honest episodes of the same checkpoint.")
print()
for name, slug in SLUGS.items():
    print(f"--- monitor {name} ---")
    for pol, marker in POLICIES.items():
        pos_all, pos_fired, neg = [], [], []
        for eid, (ph, rec) in episodes.items():
            if marker not in eid or eid not in scores.get(slug, {}):
                continue
            if "|FULL|" not in eid:
                continue
            if ph == "honest":
                neg.append(scores[slug][eid])
            elif ph == "attack":
                pos_all.append(scores[slug][eid])
                if fired_cache.get(eid):
                    pos_fired.append(scores[slug][eid])
            elif ph == "probe":
                # probes are attack-prompted; fired status also computable
                pos_all.append(scores[slug][eid])
                if any(rescore(rec)["side_task_success"].values()):
                    pos_fired.append(scores[slug][eid])
        a_all = auroc(pos_all, neg)
        a_fired = auroc(pos_fired, neg)
        print(f"  {pol:22s} ALL: n_pos={len(pos_all):3d} AUROC={a_all:.3f}   "
              f"FIRED: n_pos={len(pos_fired):3d} AUROC={a_fired:.3f}")
    print()
