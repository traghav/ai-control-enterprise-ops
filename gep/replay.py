"""Replay: rebuild a world from a logged episode and score it with CURRENT scorers.

Episodes persist their action list, not their world state. That is deliberate: it lets the
analysis phase re-score old episodes whenever a scorer changes, instead of either (a)
keeping stale numbers that no longer match the repo's scorers, or (b) re-running expensive
GPU episodes to fix a scoring bug. Replaying is cheap (pure Python, no model calls) and is
exact because the world is deterministic given (seed, affordances) and dispatch is
order-dependent only through the action list itself.

Use this whenever side-task scorers or main-task programs change after episodes were run.
"""

from __future__ import annotations

import json
from typing import Any

from .engine import dispatch
from .side_tasks import score_all
from .tasks import build_tasks
from .world import AffordanceConfig, GEPWorld


def replay_world(record: dict[str, Any]) -> GEPWorld:
    """Rebuild the world for an episode record and replay its actions in order."""
    cell = record["cell"]
    world = GEPWorld(
        seed=int(record.get("seed", cell["seed"])),
        scenario=cell["scenario"],
        affordances=AffordanceConfig.get(cell["affordances"]),
    )
    for a in record.get("actions", []):
        if a.get("blocked"):
            # Blocked calls never mutated state; dispatch would re-derive the block, but
            # skipping keeps the replay faithful to what actually executed.
            continue
        dispatch(world, a["tool"], json.dumps(a.get("args", {}), default=str))
    return world


def rescore(record: dict[str, Any]) -> dict[str, Any]:
    """Return {main_task_success, side_task_success, side_task_detail} under current scorers."""
    world = replay_world(record)
    cell = record["cell"]
    scenario = cell["scenario"]
    task_id = cell.get("main_task_id")
    tasks = build_tasks(world, scenario, n_tasks=8)
    task = next((t for t in tasks if t.task_id == task_id), None)
    main = {task_id: bool(task.check(world))} if task is not None else {task_id: False}
    sides = score_all(world)
    return {
        "main_task_success": main,
        "side_task_success": {k: bool(v[0]) for k, v in sides.items()},
        "side_task_detail": {k: v[1] for k, v in sides.items()},
    }
