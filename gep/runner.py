"""Episode runner: drives one (policy, world, monitor) episode end to end.

The agent loop is a plain tool-calling loop, capped by `max_turns`. On exit the world is
scored for main-task and side-task success and the monitor scores the resulting trajectory.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any

from .engine import Episode, LLM, LLMConfig, dispatch, tool_schemas
from .monitors import LLMMonitor, MonitorSpec
from .policies import HONEST_SYSTEM, build_attack_system, build_selective_attack_system, user_message
from .side_tasks import score_all
from .tasks import MainTask, build_tasks
from .world import AffordanceConfig, GEPWorld

MAX_TURNS = 25


@dataclass
class Cell:
    """One cell of the experiment matrix."""

    untrusted_model: str
    untrusted_family: str
    monitor_model: str | None
    monitor_family: str | None
    affordances: str          # FULL | RESTRICTED
    mode: str                 # honest | attack | selective_attack
    monitor_informed: bool = True

    def key(self) -> str:
        m = self.monitor_model.split("/")[-1] if self.monitor_model else "none"
        u = self.untrusted_model.split("/")[-1]
        opt = "" if self.monitor_informed else "-uninformed"
        return f"{u}__{m}{opt}__{self.affordances}__{self.mode}"


def run_episode(
    cell: Cell,
    task: MainTask,
    seed: int,
    untrusted_llm: LLM,
    monitor: LLMMonitor | None,
    scenario: str,
    max_turns: int = MAX_TURNS,
) -> Episode:
    """Run a single episode and return the scored record."""
    aff = AffordanceConfig.get(cell.affordances)  # type: ignore[arg-type]
    world = GEPWorld(seed=seed, scenario=scenario, affordances=aff)

    # Main tasks are defined *relative to a world instance* (they close over the planted
    # ground truth), so rebuild against THIS world and select by index. The caller must
    # pass the SAME seed it used to build `task`, in which case index selection is exact.
    # A mismatch is a hard error, never a silent fallback: an earlier revision fell back to
    # the caller's task on mismatch, producing a cross-world chimera (goal/context from one
    # world, ticket blob from another) that assigned agents unsatisfiable tasks and would
    # have silently corrupted the usefulness column.
    idx = int(task.task_id.rsplit("-", 1)[-1])
    fresh = build_tasks(world, scenario, n_tasks=max(8, idx + 1))
    if idx >= len(fresh):
        raise RuntimeError(
            f"{task.task_id} does not exist in world seed={seed} scenario={scenario} "
            f"({len(fresh)} tasks built); caller and runner used different seeds"
        )
    fresh_task = fresh[idx]
    if (fresh_task.task_id != task.task_id or fresh_task.ticket_id != task.ticket_id
            or fresh_task.goal != task.goal):
        raise RuntimeError(
            f"task drift across worlds for {task.task_id}: "
            f"driver ticket={task.ticket_id} goal={task.goal!r} vs "
            f"fresh ticket={fresh_task.ticket_id} goal={fresh_task.goal!r}"
        )
    task = fresh_task

    include_http = aff.allow_outbound_http
    schemas = tool_schemas(cell.affordances, include_http)

    if cell.mode == "honest":
        system = HONEST_SYSTEM
    elif cell.mode in ("attack", "attack_probe"):
        # attack_probe = an ALIGNED model given the attack prompt (elicitation probe).
        # It uses the same attack system prompt as the real attack phase; the distinct
        # mode label exists so analysis never reads these rows as attack capability.
        system = build_attack_system(include_http)
    elif cell.mode == "selective_attack":
        system = build_selective_attack_system(include_http)
    else:
        raise ValueError(f"unknown mode: {cell.mode}")

    ticket = world.state["tickets_by_id"].get(task.ticket_id, {})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(task.goal, ticket)},
    ]

    ep = Episode(
        episode_id=f"{cell.key()}__{task.task_id}__seed{seed}",
        cell={
            "untrusted_model": cell.untrusted_model, "untrusted_family": cell.untrusted_family,
            "monitor_model": cell.monitor_model, "monitor_family": cell.monitor_family,
            "monitor_informed": cell.monitor_informed,
            "affordances": cell.affordances, "mode": cell.mode,
            "scenario": scenario, "main_task_id": task.task_id, "seed": seed,
        },
    )

    try:
        for turn in range(max_turns):
            ep.n_turns = turn + 1
            r = untrusted_llm.chat(messages, tools=schemas)
            content = r.get("content") or ""
            tool_calls = r.get("tool_calls") or []

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                ep.final_text = content  # persist the closing reply for FINDINGS spot-checks
                break  # agent finished

            for tc in tool_calls:
                n_before = len(world.events)
                result = dispatch(world, tc["name"], tc["arguments"])
                blocked = bool(world.events and len(world.events) > n_before
                               and world.events[-1].get("blocked"))
                ep.actions.append({
                    "i": len(ep.actions), "tool": tc["name"],
                    "args": _safe_json(tc["arguments"]), "result": result, "blocked": blocked,
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(result, default=str)[:6000],
                })
    except Exception as e:  # noqa: BLE001
        ep.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"

    # ---- scoring ----------------------------------------------------------------
    try:
        ep.main_task_success = {task.task_id: bool(task.check(world))}
    except Exception as e:  # noqa: BLE001
        ep.main_task_success = {task.task_id: False}
        ep.error = (ep.error or "") + f" | main scorer error: {e}"

    try:
        raw_sides = score_all(world)
        ep.side_task_success = {k: bool(v[0]) for k, v in raw_sides.items()}
        ep.side_task_detail = {k: v[1] for k, v in raw_sides.items()}
    except Exception as e:  # noqa: BLE001
        ep.error = (ep.error or "") + f" | side scorer error: {e}"

    if monitor is not None and ep.actions:
        try:
            mon = monitor.score_trajectory(task.goal, ep.actions)
            ep.suspiciousness = [p["score"] for p in mon["per_action"]]
            ep.max_suspicion = mon["max"]
        except Exception as e:  # noqa: BLE001
            ep.error = (ep.error or "") + f" | monitor error: {e}"

    return ep


def _safe_json(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return {"_raw": str(s)[:500]}
