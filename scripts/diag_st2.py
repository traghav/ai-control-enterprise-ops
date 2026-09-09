"""Adjudicate the three spontaneous honest-mode ST2 fires from the powered run."""
import json
import sys

sys.path.insert(0, ".")

from gep.replay import replay_world, rescore  # noqa: E402

honest = [json.loads(l) for l in open("runs/episodes_honest.jsonl")]
for r in honest:
    rs = rescore(r)
    if rs["side_task_success"].get("ST2"):
        print("=" * 90)
        print(r["episode_id"])
        for a in r["actions"]:
            args = json.dumps(a["args"], default=str)
            print("  %d %s %s" % (a["i"], a["tool"], args[:130]))
        # what changed unauthorised?
        w = replay_world(r)
        d = rs["side_task_detail"]["ST2"]
        print("  detail:", json.dumps(d, default=str)[:400])
        print("  final_text:", (r.get("final_text") or "")[:200].replace("\n", " "))
