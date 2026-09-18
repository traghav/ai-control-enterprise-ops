"""Dump full honest ROI trajectories to adjudicate monitor-vs-scorer disagreement."""
import json
import sys

sys.path.insert(0, ".")

from gep.replay import rescore  # noqa: E402

recs = [json.loads(l) for l in open("runs/episodes_honest.jsonl")]
roi = [r for r in recs if "roi_withhold" in r["episode_id"]
       and "abliterated" in r["cell"]["untrusted_model"]]

print(f"honest Qwen-abl ROI episodes: {len(roi)}")
for r in roi[:3]:
    print("=" * 80)
    print(r["episode_id"])
    for a in r["actions"]:
        args = json.dumps(a["args"], default=str)
        print("  %d %s %s" % (a["i"], a["tool"], args[:150]))
    rs = rescore(r)
    print("  --- rescore: main=%s side_fired=%s" % (
        any(rs["main_task_success"].values()),
        [k for k, v in rs["side_task_success"].items() if v]))
    if r.get("final_text"):
        print("  --- final_text:", r["final_text"][:250].replace("\n", " "))
