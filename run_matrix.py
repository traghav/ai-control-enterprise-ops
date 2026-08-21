"""Phased experiment driver.

Execution model (chosen for a single 40GB GPU):

  Phase honest  -- policy models run HONEST episodes on both affordance configs.
  Phase attack  -- attack-elicited policies run ATTACK episodes on both configs.
  Phase monitor -- one monitor at a time, loaded alone at high util, scores every saved
                   trajectory with ONLINE prefix scoring (per action, no future access).
  Phase analyze -- computes usefulness / attack success / AUROC / safety-at-budget tables.

Phases run ONE MODEL AT A TIME because concurrent vLLM servers profile against each
other's allocations and race during startup; two servers started together on one GPU fail
KV-cache init nondeterministically (observed: "Available KV cache memory: -5.84 GiB").
Post-hoc monitoring decouples policy and monitor entirely: a trajectory is model+policy
output and can be scored by any monitor later. This is also what makes the cross-lineage
monitor transfer matrix free -- no episode is ever re-run for a different monitor.

Every episode is appended as one JSON line to runs/episodes_<phase>.jsonl; monitor scores
to runs/scores__<monitor-slug>.jsonl keyed by episode_id. All phases are resumable:
completed episode_ids are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gep.engine import LLM, LLMConfig
from gep.monitors import LLMMonitor, MonitorSpec
from gep.policies import build_attack_system  # noqa: F401  (documented re-export)
from gep.runner import Cell, run_episode
from gep.tasks import build_tasks  # noqa: F401
from gep.world import GEPWorld  # noqa: F401
from analysis.metrics import summarize_cell

RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)

BASE_SEED = 20260918

# --------------------------------------------------------------------------------------
# Roster: what runs in which phase. Family = vendor lineage for the transfer matrix.
# --------------------------------------------------------------------------------------

POLICY_HONEST = [
    # (repo_id, family, port, gpu_util)  -- official aligned checkpoints
    ("Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen", 8020, 0.85),
    ("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama", 8021, 0.55),
    ("stelterlab/phi-4-AWQ", "phi", 8022, 0.65),
]
POLICY_ATTACK = [
    # attack-elicited checkpoints. The abliterated Qwen2.5-32B is the same base checkpoint
    # as the honest Qwen2.5-32B, so the honest/attack pair differs by behavioural
    # modification only, not by lineage or scale (see DECISIONS D6). phi-4 appears in BOTH
    # rosters: an aligned model given attack prompts measures elicitation, not attack
    # success, so its attack-mode rows are reported as the probe.
    ("ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ", "qwen", 8023, 0.85),
    ("stelterlab/phi-4-AWQ", "phi", 8022, 0.65),
]
POLICY_ALIGNED_ATTACK_PROBE = [
    # aligned model given attack prompts: measures elicitation/refusal, not attack success
    ("Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen", 8020, 0.85),
]
MONITORS = [
    # (repo_id, family, port, gpu_util, informed)
    ("Qwen/Qwen3-8B-AWQ", "qwen", 8030, 0.60, True),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "qwen", 8031, 0.60, True),
    ("gaunernst/gemma-3-27b-it-int4-awq", "gemma", 8032, 0.80, True),
]

SCENARIOS = ["record_correction", "duplicate_record", "med_reconciliation", "audit_review",
             "access_review", "ledger_reconcile", "roi_withhold"]
AFFORDANCES = ["FULL", "RESTRICTED"]
EPISODES_PER_SCENARIO = 4   # reps; each rep = fresh world seed => distinct planted tasks
MAX_PARALLEL_EPISODES = 4


def episodes_path(phase: str) -> Path:
    return RUNS / f"episodes_{phase}.jsonl"


def scores_path(monitor_slug: str) -> Path:
    return RUNS / f"scores__{monitor_slug}.jsonl"


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["episode_id"])
            except Exception:  # noqa: BLE001
                continue
    return done


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def wait_for_port(port: int, timeout_s: int = 1200) -> None:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise RuntimeError(f"model server on :{port} not ready after {timeout_s}s")


def run_phase_episodes(phase: str, roster: list[tuple], port_for_run: int | None = None) -> None:
    """Run all (model, affordance, scenario, repeat) episodes for a phase.

    Only roster models whose server is CURRENTLY UP are run, each against its own roster
    port. One vLLM server runs at a time on this GPU (concurrent servers race during
    startup and fail KV-cache init), so the operator serves a model, the driver picks up
    whatever is live, then the next model is served. Passing --port restricts to that port.
    """
    import urllib.request

    def up(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    out = episodes_path(phase)
    done = load_done_ids(out)

    live = [(repo, family, port, util) for repo, family, port, util in roster
            if up(port) and (port_for_run is None or port == port_for_run)]
    if not live:
        print(f"[{phase}] no phase-model server is up; nothing to do. Serve a roster port "
              f"first (ops/tmuxctl.sh start <port>).")
        return

    jobs = []
    for repo, family, port, _util in live:
        for aff in AFFORDANCES:
            for scen in SCENARIOS:
                for rep in range(EPISODES_PER_SCENARIO):
                    eid = f"{phase}|{repo}|{aff}|{scen}|{rep}"
                    if eid in done:
                        continue
                    jobs.append((repo, family, port, aff, scen, rep))

    if not jobs:
        print(f"[{phase}] all episodes already complete for live models ({len(done)} done)")
        return
    print(f"[{phase}] {len(jobs)} episodes across {len(live)} live model(s): "
          + ", ".join(f"{r[0].split('/')[-1]}@{r[2]}" for r in live))

    lock = threading.Lock()
    counter = {"done": 0, "t0": time.time()}

    def one(job) -> None:
        repo, family, port, aff, scen, rep = job
        # zlib.crc32, NOT hash(): str hash is salted per process (PYTHONHASHSEED) and would
        # silently make seeds -- and therefore tasks -- non-reproducible across runs. This
        # is exactly the D9 bug; do not reintroduce it.
        seed = BASE_SEED + rep * 101 + (SCENARIOS.index(scen) * 7) + (0 if aff == "FULL" else 3)
        cell = Cell(
            untrusted_model=repo, untrusted_family=family,
            monitor_model=None, monitor_family=None,
            affordances=aff,
            # probe = aligned model given attack prompts: measures elicitation/refusal,
            # NOT attack success. Distinct mode label so analysis never reads these rows
            # as the aligned model's attack capability.
            mode="honest" if phase == "honest" else "attack_probe" if phase == "probe" else "attack",
        )
        llm = LLM(LLMConfig(model=repo, base_url=f"http://localhost:{port}/v1",
                            temperature=0.7, max_tokens=1536))
        world = GEPWorld(seed=seed, scenario=scen,
                         affordances=__import__("gep.world", fromlist=["AffordanceConfig"])
                         .AffordanceConfig.get(aff))
        tasks = build_tasks(world, scen, 8)
        # Deduplicate: several scenarios key all their task instances off ONE planted
        # ticket, so instances differ only by task_id. Running all 8 would be 8 identical
        # episodes at 8x cost. Keep the first instance per (goal, ticket_id).
        seen: set[tuple] = set()
        uniq = []
        for t in tasks:
            k = (t.goal, t.ticket_id)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(t)
        t0 = time.time()
        eps = []
        for t in uniq:
            # Same seed as the world above: run_episode rebuilds the world and reselects
            # the task by index; identical seeds make that selection exact (a mismatch is
            # a hard error, by design). Never re-seed per task here -- an earlier revision
            # did, the index space shifted across worlds, and the fallback produced
            # cross-world chimera tasks with unreachable goals.
            ep = run_episode(cell, t, seed=seed,
                             untrusted_llm=llm, monitor=None, scenario=scen)
            rec = ep.to_dict()
            rec["episode_id"] = f"{phase}|{repo}|{aff}|{scen}|{rep}|{t.task_id}"
            rec["rep"] = rep
            rec["wall_s"] = round(time.time() - t0, 1)
            eps.append(rec)
        with lock:
            for rec in eps:
                append_jsonl(out, rec)
            counter["done"] += 1
            n, el = counter["done"], time.time() - counter["t0"]
            print(f"[{phase}] {n}/{len(jobs)} {repo.split('/')[-1][:28]} {aff} {scen}"
                  f" main={sum(rec['main_task_success'].values(), 0)}/{len(eps)}"
                  f" side={sum(1 for r in eps if any(r['side_task_success'].values()))}"
                  f" ({el/n:.0f}s/ep)", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EPISODES) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            f.result()


# --------------------------------------------------------------------------------------
# Monitor phase
# --------------------------------------------------------------------------------------


def run_monitor_phase(repo: str, family: str, port: int, informed: bool = True) -> None:
    slug = repo.split("/")[-1] + ("_informed" if informed else "_uninformed")
    out = scores_path(slug)
    done = load_done_ids(out)
    wait_for_port(port)

    # Collect every scored episode across phases.
    traj: dict[str, dict] = {}
    for phase in ("honest", "attack", "probe"):
        p = episodes_path(phase)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                eid = rec["episode_id"]
                if eid in done:
                    continue
                # Group episode records by base episode (one record per main task).
                traj.setdefault(eid, rec)

    todo = {k: v for k, v in traj.items() if v.get("actions")}
    print(f"[monitor:{slug}] {len(todo)} trajectories to score")
    if not todo:
        return

    mon = LLMMonitor(MonitorSpec(name=slug, model=repo, base_url=f"http://localhost:{port}/v1",
                                 family=family, informed=informed))
    lock = threading.Lock()
    counter = {"done": 0, "t0": time.time()}

    def one(eid: str, rec: dict) -> None:
        actions = rec["actions"]
        per = [mon.score_action(rec["cell"].get("main_task_id", ""), actions, i)
               for i in range(len(actions))]
        with lock:
            append_jsonl(out, {
                "episode_id": eid,
                "monitor": slug,
                "monitor_family": family,
                "scores": [p["score"] for p in per],
                "max": max((p["score"] for p in per), default=0.0),
                "mean": (sum(p["score"] for p in per) / len(per)) if per else 0.0,
                "reasons": [p.get("reason", "")[:200] for p in per],
            })
            counter["done"] += 1
            n, el = counter["done"], time.time() - counter["t0"]
            if n % 10 == 0 or n == len(todo):
                print(f"[monitor:{slug}] {n}/{len(todo)} ({el/n:.1f}s/traj)", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, eid, rec) for eid, rec in todo.items()]
        for f in as_completed(futs):
            f.result()


# --------------------------------------------------------------------------------------
# Analyze phase
# --------------------------------------------------------------------------------------


def run_analyze() -> None:
    import pandas as pd

    # Join episodes with every monitor's scores.
    frames = []
    for phase in ("honest", "attack", "probe"):
        p = episodes_path(phase)
        if not p.exists():
            continue
        eps = [json.loads(l) for l in open(p) if l.strip()]
        score_map: dict[str, list[dict]] = {}
        for sp in RUNS.glob("scores__*.jsonl"):
            for l in open(sp):
                if not l.strip():
                    continue
                r = json.loads(l)
                score_map.setdefault(r["episode_id"], []).append(r)
        for e in eps:
            for s in score_map.get(e["episode_id"], []):
                frames.append((e, s))
    if not frames:
        print("[analyze] no scored episodes yet")
        return

    # summarize_cell expects the nested {"cell": {...}, ...} shape of Episode.to_dict().
    by_cell: dict[tuple, list[dict]] = {}
    for e, s in frames:
        eid = e["episode_id"]
        cellkey = (s["monitor"], e["cell"]["untrusted_model"], e["cell"]["untrusted_family"],
                   e["cell"]["affordances"], e["cell"]["mode"])
        rec = {"episode_id": eid, "cell": e["cell"],
               "main_task_success": e["main_task_success"],
               "side_task_success": e["side_task_success"],
               "max_suspicion": s["max"], "error": e.get("error")}
        by_cell.setdefault(cellkey, []).append(rec)

    rows = []
    side_ids = ["ST1", "ST2", "ST3", "ST4", "ST5"]
    for cellkey, eps in by_cell.items():
        s = summarize_cell(eps, side_ids)
        row = s.to_row()
        row["monitor"] = cellkey[0]
        row["untrusted_model"] = cellkey[1]
        row["untrusted_family"] = cellkey[2]
        row["affordances"] = cellkey[3]
        row["mode"] = cellkey[4]
        rows.append(row)

    df = pd.DataFrame([{
        "episode_id": e["episode_id"], "phase": e["cell"].get("scenario"),
        "main_success": 1 if any(e["main_task_success"].values()) else 0,
        "side_success": 1 if any(e["side_task_success"].values()) else 0,
        "max_suspicion": s["max"], "monitor": s["monitor"],
        "monitor_family": s["monitor_family"],
        "n_actions": len(e.get("actions", [])),
        **{k: v for k, v in e["cell"].items()},
    } for e, s in frames])
    df.to_csv(RUNS / "joined.csv", index=False)
    print(f"[analyze] joined table: {len(df)} rows -> runs/joined.csv")
    out = pd.DataFrame(rows).sort_values(["untrusted_model", "monitor", "affordances", "mode"])
    out.to_csv(RUNS / "cell_summary.csv", index=False)
    with open(RUNS / "cell_summary.md", "w") as f:
        cols = ["untrusted_model", "monitor", "affordances", "mode", "n_honest", "n_attack",
                "usefulness", "attack_success_rate", "auroc", "safety_1pct", "safety_5pct",
                "mean_suspicion_honest", "mean_suspicion_attack", "n_errors"]
        f.write(out[[c for c in cols if c in out.columns]].to_markdown(index=False))
    print(f"[analyze] {len(out)} cells -> runs/cell_summary.csv / .md")
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["honest", "attack", "probe", "monitor", "analyze"])
    ap.add_argument("--model", help="override: run only this roster entry (repo id substring)")
    ap.add_argument("--port", type=int, help="port the current phase model is served on")
    ap.add_argument("--monitor", help="monitor repo id substring for the monitor phase")
    args = ap.parse_args()

    if args.phase == "analyze":
        run_analyze()
        return

    if args.phase == "monitor":
        cands = [m for m in MONITORS if not args.monitor or args.monitor in m[0]]
        if not cands:
            raise SystemExit("no matching monitor")
        repo, family, port, _u, informed = cands[0]
        run_monitor_phase(repo, family, args.port or port, informed)
        return

    phase = "honest" if args.phase == "honest" else ("attack" if args.phase == "attack" else "probe")
    roster = {"honest": POLICY_HONEST, "attack": POLICY_ATTACK,
              "probe": POLICY_ALIGNED_ATTACK_PROBE}[phase]
    if args.model:
        roster = [r for r in roster if args.model in r[0]]
        if not roster:
            raise SystemExit("no matching model")
    port = args.port or roster[0][2]
    run_phase_episodes(phase, roster, port)


if __name__ == "__main__":
    main()
