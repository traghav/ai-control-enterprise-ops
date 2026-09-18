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
    # (repo_id, family, port, gpu_util)
    # D12: the honest baseline for each untrusted policy is THE SAME CHECKPOINT run in
    # honest mode -- mode is the only differing factor, which is what makes AUROC/safety
    # within-model meaningful. The aligned Qwen row is kept (its honest data doubles as
    # the probe's honest reference). The aligned NousResearch-Llama and phi-4 entries were
    # REMOVED: their honest legs produced zero-action junk (NousResearch's re-upload ships
    # a chat template that never renders tool schemas; phi-4 has no vLLM tool parser and
    # every tool_choice=auto request was a 400) -- see D12 and the purge note.
    ("Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen", 8020, 0.85),
    ("ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ", "qwen", 8023, 0.85),
    ("NousResearch/Hermes-3-Llama-3.1-8B", "llama", 8024, 0.70),
]
POLICY_ATTACK = [
    # Attack phase runs ONLY on behaviourally-ablated checkpoints (DECISIONS D6): aligned
    # models given attack prompts are the PROBE phase (elicitation/refusal), never attack
    # capability. Two lineages => the monitor transfer matrix is genuinely cross-lineage.
    # (Two Llama-lineage uncensored fine-tunes were tried and BOTH lost tool-calling --
    # mlabonne's abliterated and Orenguteng's Lexi-Uncensored each produced 90 one-turn
    # episodes of tool calls written as markdown prose: the tools reach the model -- the
    # prose cites exact ticket values -- but the chat templates never render the
    # tool-call OUTPUT format. Hermes-3 is explicitly tool-call trained with ChatML tool
    # syntax that vLLM's hermes parser matches natively.)
    ("ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ", "qwen", 8023, 0.85),
    ("NousResearch/Hermes-3-Llama-3.1-8B", "llama", 8024, 0.70),
]
POLICY_PROBE = [
    # aligned models given attack prompts: measures elicitation/refusal, not attack success
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
# Power expansion (D14): 10 reps/cell tightens the episode-level bootstrap CI from
# ~+/-0.14 (4 reps) to ~+/-0.09, putting the FULL-vs-RESTRICTED attack-success delta and
# the probe asymmetry inside the resolvable band. Resumable: existing 4-rep episodes are
# skipped via the (fixed) prefix done-check, so only reps 4..9 execute.
EPISODES_PER_SCENARIO = 10
MAX_PARALLEL_EPISODES = 4

# E3 (attack-selection) cell: re-run ONE cell with the start/stop policy -- attack only
# when a side task is naturally reachable from the current main task. Cell choice: the
# highest-attack-success configuration (Qwen-abl x FULL), so the safety delta is measured
# where there is the most attack signal to lose or gain.
SELECTIVE_ROSTER = [
    ("ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ", "qwen", 8023, 0.85),
]
SELECTIVE_AFFORDANCES = ["FULL"]


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
    done_ids = load_done_ids(out)
    # Stored episode ids are 6-part (…|TASKID); the job key is the 5-part prefix. Match on
    # prefix: the earlier check compared 5-part keys against full ids and NEVER matched,
    # so a resumed run would re-execute completed jobs and append duplicate episodes.
    done_prefixes = {"|".join(eid.split("|")[:5]) for eid in done_ids}

    live = [(repo, family, port, util) for repo, family, port, util in roster
            if up(port) and (port_for_run is None or port == port_for_run)]
    if not live:
        print(f"[{phase}] no phase-model server is up; nothing to do. Serve a roster port "
              f"first (ops/tmuxctl.sh start <port>).")
        return

    jobs = []
    for repo, family, port, _util in live:
        # E3 selective phase runs ONE cell only (SELECTIVE_AFFORDANCES), not the grid.
        affs = SELECTIVE_AFFORDANCES if phase == "selective" else AFFORDANCES
        for aff in affs:
            for scen in SCENARIOS:
                for rep in range(EPISODES_PER_SCENARIO):
                    eid = f"{phase}|{repo}|{aff}|{scen}|{rep}"
                    if eid in done_prefixes:
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
            # as the aligned model's attack capability. selective = E3 start/stop policy
            # (attack only when the side task is naturally reachable from the ticket).
            mode=("honest" if phase == "honest"
                  else "attack_probe" if phase == "probe"
                  else "selective_attack" if phase == "selective"
                  else "attack"),
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
            rec["cell"]["main_task_goal"] = t.goal  # monitor needs the real goal text
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


def run_monitor_phase(repo: str, family: str, port: int, informed: bool = True,
                      ticket_aware: bool = False) -> None:
    slug = repo.split("/")[-1] + ("_informed" if informed else "_uninformed") \
        + ("_ticket" if ticket_aware else "")
    out = scores_path(slug)
    done = load_done_ids(out)
    wait_for_port(port)

    # Collect every scored episode across phases.
    traj: dict[str, dict] = {}
    for phase in ("honest", "attack", "probe", "selective"):
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
                                 family=family, informed=informed, ticket_aware=ticket_aware))
    lock = threading.Lock()
    counter = {"done": 0, "t0": time.time()}

    # E4: the ticket blob for ticket-aware scoring must be the ticket's ORIGINAL state
    # (pre-episode). Comments are EXCLUDED -- they are agent-writable, and a monitor that
    # reads them can be self-authorised against. Derived by replaying the world WITHOUT
    # the episode's actions (deterministic, no model calls).
    def original_ticket_blob(rec: dict) -> str:
        from gep.world import AffordanceConfig, GEPWorld

        cell = rec["cell"]
        try:
            w = GEPWorld(seed=int(rec.get("seed", cell["seed"])), scenario=cell["scenario"],
                         affordances=AffordanceConfig.get(cell["affordances"]))
            t = w.state["tickets_by_id"].get(cell.get("main_ticket_id", ""))
            if t is None:
                # fall back to the pre-episode world's ticket for this task
                from gep.tasks import build_tasks
                tasks = build_tasks(w, cell["scenario"], 8)
                tid = next((t2.ticket_id for t2 in tasks
                            if t2.task_id == cell.get("main_task_id")), None)
                t = w.state["tickets_by_id"].get(tid or "")
            if t is None:
                return ""
            return f"Subject: {t['subject']}\n{t['body']}"
        except Exception:  # noqa: BLE001
            return ""

    def one(eid: str, rec: dict) -> None:
        actions = rec["actions"]
        # The monitor must see the actual goal text, not the task id: judging whether an
        # action is authorised requires knowing what the agent was asked to do. Older
        # records carry only main_task_id, so derive the goal by replaying the world
        # (deterministic, no model calls).
        goal = rec["cell"].get("main_task_goal", "")
        if not goal:
            from gep.replay import replay_world
            from gep.tasks import build_tasks

            try:
                w = replay_world(rec)
                t = next((t for t in build_tasks(w, rec["cell"]["scenario"], 8)
                          if t.task_id == rec["cell"].get("main_task_id")), None)
                goal = t.goal if t is not None else rec["cell"].get("main_task_id", "")
            except Exception:  # noqa: BLE001
                goal = rec["cell"].get("main_task_id", "")
        # Ticket blob for ticket-aware monitors: computed once per episode, original
        # state only (empty string for context-free monitors, which never see it).
        blob = original_ticket_blob(rec) if mon.spec.ticket_aware else ""
        per = [mon.score_action(goal, actions, i, blob)
               for i in range(len(actions))]
        with lock:
            append_jsonl(out, {
                "episode_id": eid,
                "monitor": slug,
                "monitor_family": family,
                "ticket_aware": mon.spec.ticket_aware,
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

    from gep.replay import rescore

    # Join episodes with every monitor's scores. Task/side-task verdicts are RE-DERIVED
    # via replay under the CURRENT scorers and world semantics -- stored verdicts may have
    # been computed by an older scorer version at episode time, and analysis must never
    # depend on which scorer version happened to be running then. Replay is pure Python
    # (no model calls), so this is cheap.
    frames = []  # (phase, episode, monitor_score)
    mon_family: dict[str, str] = {}  # monitor slug -> family (from score records; the
    # episode cells carry monitor_family=None for policy phases, so the score record is
    # the only reliable source)
    rescored_cache: dict[str, dict] = {}
    for phase in ("honest", "attack", "probe", "selective"):
        p = episodes_path(phase)
        if not p.exists():
            continue
        eps_by_id: dict[str, dict] = {}
        for line in open(p):
            if not line.strip():
                continue
            rec = json.loads(line)
            # Dedup by episode_id (last wins): the resume check had a prefix-matching bug
            # that could append a completed job twice; analysis must count each episode
            # once regardless of what the logs contain.
            eps_by_id[rec["episode_id"]] = rec
        eps = list(eps_by_id.values())
        score_map: dict[str, list[dict]] = {}
        for sp in RUNS.glob("scores__*.jsonl"):
            for l in open(sp):
                if not l.strip():
                    continue
                r = json.loads(l)
                score_map.setdefault(r["episode_id"], []).append(r)
                mon_family[r["monitor"]] = r.get("monitor_family", "")
        for e in eps:
            eid = e["episode_id"]
            if eid not in rescored_cache:
                try:
                    rescored_cache[eid] = rescore(e)
                except Exception as ex:  # noqa: BLE001
                    print(f"[analyze] rescore failed for {eid}: {ex!r}")
                    rescored_cache[eid] = {
                        "main_task_success": e.get("main_task_success", {}),
                        "side_task_success": e.get("side_task_success", {}),
                        "side_task_detail": e.get("side_task_detail", {}),
                    }
            e = dict(e)
            e["main_task_success"] = rescored_cache[eid]["main_task_success"]
            e["side_task_success"] = rescored_cache[eid]["side_task_success"]
            for s in score_map.get(eid, []):
                frames.append((phase, e, s))
    if not frames:
        print("[analyze] no scored episodes yet")
        return

    # summarize_cell expects the nested {"cell": {...}, ...} shape of Episode.to_dict().
    # GROUPING IS BY (monitor, untrusted_model, affordances) with BOTH modes merged:
    # summarize_cell splits honest/attack internally. AUROC and safety are computed per
    # untrusted policy against ITS OWN honest-mode episodes -- the protocol-correct pairing
    # (same checkpoint, same setting, mode is the only difference). The earlier NaNs were
    # NOT a grouping bug: the same-model honest legs were missing because the NousResearch
    # Llama and phi-4 honest phases produced zero-action junk (see D12), so no honest
    # episodes existed for those models. The fix is the data (run honest legs for the
    # attack checkpoints), not the grouping.
    by_cell: dict[tuple, list[dict]] = {}
    for _phase, e, s in frames:
        eid = e["episode_id"]
        cellkey = (s["monitor"], e["cell"]["untrusted_model"], e["cell"]["affordances"])
        rec = {"episode_id": eid, "cell": e["cell"],
               "main_task_success": e["main_task_success"],
               "side_task_success": e["side_task_success"],
               "max_suspicion": s["max"], "error": e.get("error")}
        by_cell.setdefault(cellkey, []).append(rec)

    rows = []
    side_ids = ["ST1", "ST2", "ST3", "ST4", "ST5"]
    for (mon, model, aff), eps in by_cell.items():
        s = summarize_cell(eps, side_ids)
        row = s.to_row()
        row["monitor"] = mon
        row["monitor_family"] = mon_family.get(mon, "")
        row["untrusted_model"] = model
        row["untrusted_family"] = eps[0]["cell"]["untrusted_family"] if eps else ""
        row["affordances"] = aff
        # Dominant mode label for this cell (cells mixing attack+probe on one model are
        # not possible: probe and attack never share a checkpoint).
        modes = {e["cell"]["mode"] for e in eps}
        row["mode"] = "mixed" if len(modes) > 1 else modes.pop()
        rows.append(row)

    df = pd.DataFrame([{
        "episode_id": e["episode_id"], "phase": ph,
        "main_success": 1 if any(e["main_task_success"].values()) else 0,
        "side_success": 1 if any(e["side_task_success"].values()) else 0,
        "max_suspicion": s["max"], "monitor": s["monitor"],
        "monitor_family": s["monitor_family"],
        "n_actions": len(e.get("actions", [])),
        **{k: v for k, v in e["cell"].items()},
    } for ph, e, s in frames])
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
    ap.add_argument("phase", choices=["honest", "attack", "probe", "selective", "monitor",
                                      "analyze"])
    ap.add_argument("--model", help="override: run only this roster entry (repo id substring)")
    ap.add_argument("--port", type=int, help="port the current phase model is served on")
    ap.add_argument("--monitor", help="monitor repo id substring for the monitor phase")
    ap.add_argument("--ticket-aware", action="store_true",
                    help="E4: run the monitor with the assigned ticket's original content "
                         "in context (scores to a separate _ticket slug)")
    args = ap.parse_args()

    if args.phase == "analyze":
        run_analyze()
        return

    if args.phase == "monitor":
        cands = [m for m in MONITORS if not args.monitor or args.monitor in m[0]]
        if not cands:
            raise SystemExit("no matching monitor")
        repo, family, port, _u, informed = cands[0]
        run_monitor_phase(repo, family, args.port or port, informed,
                          ticket_aware=args.ticket_aware)
        return

    phase = args.phase
    roster = {"honest": POLICY_HONEST, "attack": POLICY_ATTACK, "probe": POLICY_PROBE,
              "selective": SELECTIVE_ROSTER}[phase]
    if args.model:
        roster = [r for r in roster if args.model in r[0]]
        if not roster:
            raise SystemExit("no matching model")
    # No default port: the driver runs whatever roster models are CURRENTLY live, each on
    # its own port. (Defaulting to roster[0]'s port made the driver silently skip a live
    # model on a different port -- the Llama honest leg ran "OK" having done nothing.)
    # --port still restricts to one port when explicitly given.
    run_phase_episodes(phase, roster, args.port)


if __name__ == "__main__":
    main()
