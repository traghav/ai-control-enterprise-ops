"""Render trajectories readably for manual spot-checks.

The quality bar for the experiment requires eyeballing real episodes: 10 attack and 10
honest, with notes written into FINDINGS.md. This tool samples deterministically from the
episode logs and prints each trajectory as (goal -> actions -> verdicts), so a human (or
the operator) can judge whether the automated verdicts match what the transcript shows.

Usage:
  python analysis/spot_check.py --phase attack --n 10
  python analysis/spot_check.py --phase honest --n 10 --monitor Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RUNS = Path(__file__).parent.parent / "runs"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def fmt_action(i: int, a: dict, width: int = 110) -> str:
    args = json.dumps(a.get("args", {}), default=str)
    if len(args) > width:
        args = args[:width] + "..."
    res = a.get("result")
    res_s = ""
    if isinstance(res, dict):
        if "error" in res:
            res_s = " !! " + str(res["error"])[:90]
        else:
            res_s = " -> " + json.dumps(res, default=str)[:90]
    blocked = " [BLOCKED]" if a.get("blocked") else ""
    return f"   {i:>2}. {a['tool']}({args}){blocked}{res_s}"


def render(rec: dict, scores: dict | None, idx: int) -> str:
    cell = rec["cell"]
    lines = [
        f"--- [{idx}] {rec['episode_id']}",
        f"    model={cell['untrusted_model'].split('/')[-1]} mode={cell['mode']} "
        f"aff={cell['affordances']} scen={cell['scenario']} seed={cell['seed']}",
        f"    goal: {cell.get('main_task_goal', cell.get('main_task_id', '?'))}",
        f"    main_success={any(rec['main_task_success'].values())} "
        f"side_success={[k for k, v in rec['side_task_success'].items() if v]}",
    ]
    if scores:
        lines.append(f"    monitor={scores.get('monitor')} max_suspicion={scores.get('max'):.2f}")
        reasons = scores.get("reasons") or []
        hi = max(range(len(scores.get("scores", []))), key=lambda i: scores["scores"][i],
                 default=-1)
        if 0 <= hi < len(reasons):
            lines.append(f"    peak action {hi}: {reasons[hi]}")
    for a in rec.get("actions", []):
        lines.append(fmt_action(a["i"], a))
    if rec.get("final_text"):
        ft = rec["final_text"][:200].replace("\n", " ")
        lines.append(f"   final: {ft}")
    if rec.get("error"):
        lines.append(f"   ERROR: {rec['error'][:200]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="attack",
                    choices=["honest", "attack", "probe", "selective"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--monitor", default=None, help="monitor slug to join scores from")
    args = ap.parse_args()

    eps = load(RUNS / f"episodes_{args.phase}.jsonl")
    if not eps:
        print(f"no episodes for phase {args.phase}")
        return

    scores: dict[str, dict] = {}
    if args.monitor:
        for p in RUNS.glob(f"scores__*{args.monitor}*.jsonl"):
            for s in load(p):
                scores[s["episode_id"]] = s

    rng = random.Random(args.seed)
    sample = rng.sample(eps, min(args.n, len(eps)))
    # Prefer episodes with a joined monitor when available.
    if scores:
        with_s = [e for e in sample if e["episode_id"] in scores]
        without = [e for e in sample if e["episode_id"] not in scores]
        sample = with_s + without

    for i, rec in enumerate(sample):
        print(render(rec, scores.get(rec["episode_id"]), i))
    print(f"\n[{len(sample)} of {len(eps)} {args.phase} episodes shown; "
          f"seed={args.seed} for reproducible sampling]")


if __name__ == "__main__":
    main()
