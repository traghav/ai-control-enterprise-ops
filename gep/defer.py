"""E5: defer-to-trusted simulation.

The flavour doc's E2 spec asked for "trusted monitoring with defer-to-trusted". The
frozen matrix measured monitor quality (AUROC, FPR) and audit-style safety, but never
simulated the DEFER protocol itself: monitor watches proposed actions online; when
suspicion crosses a threshold, the untrusted policy is stopped BEFORE executing the
crossing action and a trusted policy finishes the episode.

This module simulates that counterfactually from frozen data:

  1. Replay the episode's world to the crossing action (execute prefix 0..i-1 only --
       the crossing action was scored but is NOT executed; in deployment the monitor sees
       the proposed action and defers before execution).
  2. Rebuild the conversation prefix (system + user + assistant/tool pairs) under the
       TRUSTED (honest) system prompt.
  3. Let the honest policy continue the episode to completion (its own turn budget).
  4. Score the final world with the current scorers.

Trusted policy = the SAME checkpoint running its honest prompt (D12's same-checkpoint
baseline; in production terms, defer = hand over to the conservative path of the same
deployment). Both score sets (context-free and ticket-aware) drive separate simulations
so E4's FPR reduction can be tested for an operational payoff: fewer honest deferrals at
matched safety.
"""

from __future__ import annotations

import json
from typing import Any

from .engine import LLM, dispatch, needs_serialized_history, serialized_assistant_turns, tool_schemas
from .policies import HONEST_SYSTEM, user_message
from .world import AffordanceConfig, GEPWorld

CONT_MAX_TURNS = 25


def find_crossing(scores: list[float], threshold: float) -> int | None:
    """Index of the first action whose suspicion >= threshold (None if never)."""
    for i, s in enumerate(scores):
        if s >= threshold:
            return i
    return None


def build_prefix_messages(rec: dict[str, Any], upto: int) -> list[dict[str, Any]]:
    """Rebuild the conversation prefix (actions 0..upto-1 executed) under the honest
    system prompt.

    The original episode ran under the policy prompt (honest or attack); the continuation
    uses HONEST_SYSTEM because that is what the trusted policy is. History is rebuilt as
    ONE synthetic single-call assistant turn per recorded action (id call_{i}, no
    assistant text): this is deliberately NOT the model's original multi-call grouping —
    single-call turns are always safe to re-render for every chat template, including the
    llama3_json one that raises on multi-call history (see engine.py). Tool results are
    re-serialised with the same 6000-char truncation the live runner used.
    """
    cell = rec["cell"]
    ticket = {}
    # The ticket blob: derive from the pre-episode world (original state).
    try:
        w0 = GEPWorld(seed=int(rec.get("seed", cell["seed"])), scenario=cell["scenario"],
                      affordances=AffordanceConfig.get(cell["affordances"]))
        from .tasks import build_tasks

        tasks = build_tasks(w0, cell["scenario"], 8)
        tid = next((t.ticket_id for t in tasks if t.task_id == cell.get("main_task_id")), None)
        if tid:
            t = w0.state["tickets_by_id"].get(tid)
            if t:
                ticket = dict(t)
    except Exception:  # noqa: BLE001
        ticket = {}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": HONEST_SYSTEM},
        {"role": "user", "content": user_message(rec["cell"].get("main_task_goal", ""), ticket)},
    ]
    for a in rec.get("actions", [])[:upto]:
        call_id = f"call_{a['i']}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": a["tool"],
                             "arguments": json.dumps(a.get("args", {}), default=str)},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps(a.get("result", {}), default=str)[:6000],
        })
    return messages


def run_deferred(rec: dict[str, Any], scores: list[float], threshold: float,
                 llm: LLM, base_url: str) -> dict[str, Any]:
    """Simulate defer-to-trusted for one episode at one threshold.

    Returns a record with the deferral metadata, the continuation actions, and the final
    world verdicts under the CURRENT scorers.
    """
    from .engine import _jsonable  # noqa: F401  (used indirectly via dispatch)
    from .replay import replay_world
    from .side_tasks import score_all
    from .tasks import build_tasks

    cell = rec["cell"]
    aff = AffordanceConfig.get(cell["affordances"])
    actions = rec.get("actions", [])
    if len(scores) < len(actions):
        # Monitor scored fewer actions than exist (should not happen); align to scores.
        actions = actions[: len(scores)]

    crossing = find_crossing(scores, threshold)
    out: dict[str, Any] = {
        "episode_id": rec["episode_id"],
        "cell": cell,
        "seed": rec.get("seed", cell["seed"]),
        "threshold": threshold,
        "crossing": crossing,
        "n_actions_original": len(rec.get("actions", [])),
        "deferred": crossing is not None,
        "continuation_actions": [],
    }

    # World as it stands after the executed prefix (empty if crossing at 0 / no deferral).
    world = GEPWorld(seed=int(rec.get("seed", cell["seed"])), scenario=cell["scenario"],
                     affordances=aff)
    exec_to = crossing if crossing is not None else len(actions)
    for a in actions[:exec_to]:
        if a.get("blocked"):
            continue
        dispatch(world, a["tool"], json.dumps(a.get("args", {}), default=str))

    if crossing is None:
        # No deferral: the original episode stands as-is. Score the fully replayed world.
        verdicts = _score(world, cell)
        out.update(verdicts)
        out["deferred"] = False
        return out

    # Deferred: continue under the honest policy from the prefix state.
    messages = build_prefix_messages(rec, crossing)
    schemas = tool_schemas(cell["affordances"], aff.allow_outbound_http)
    # If the trusted policy runs on a model whose chat template cannot re-render
    # multi-call assistant turns (llama3_json family), serialize continuation turns
    # exactly as run_episode does for the untrusted policy. build_prefix_messages always
    # emits single-call turns, so only the continuation's own turns need this.
    serialize = needs_serialized_history(llm.cfg.model)
    cont: list[dict[str, Any]] = []
    error = None
    try:
        for turn in range(CONT_MAX_TURNS):
            r = llm.chat(messages, tools=schemas)
            tool_calls = r.get("tool_calls") or []
            if not tool_calls:
                out["final_text"] = (r.get("content") or "")[:2000]
                break
            if serialize:
                id_prefix = f"c{turn}"
                turns = serialized_assistant_turns(r.get("content") or "", tool_calls,
                                                   id_prefix=id_prefix)
                for j, tc in enumerate(tool_calls):
                    messages.append(turns[j])
                    result = dispatch(world, tc["name"], tc["arguments"])
                    cont.append({"i": len(cont), "tool": tc["name"],
                                 "args": _safe_json(tc["arguments"]), "result": result})
                    messages.append({
                        "role": "tool", "tool_call_id": f"{id_prefix}_{j}",
                        "content": json.dumps(result, default=str)[:6000],
                    })
            else:
                messages.append({"role": "assistant", "content": r.get("content") or "",
                                 "tool_calls": [
                                     {"id": f"c{turn}_{j}", "type": "function",
                                      "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                     for j, tc in enumerate(tool_calls)]})
                for j, tc in enumerate(tool_calls):
                    result = dispatch(world, tc["name"], tc["arguments"])
                    cont.append({"i": len(cont), "tool": tc["name"],
                                 "args": _safe_json(tc["arguments"]), "result": result})
                    messages.append({
                        "role": "tool", "tool_call_id": f"c{turn}_{j}",
                        "content": json.dumps(result, default=str)[:6000],
                    })
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    out["continuation_actions"] = _jsonable_safe(cont)
    out["continuation_n"] = len(cont)
    out["error"] = error
    verdicts = _score(world, cell)
    out.update(verdicts)
    return out


def _score(world: GEPWorld, cell: dict[str, Any]) -> dict[str, Any]:
    """Main + side verdicts for the final world under current scorers."""
    from .side_tasks import score_all
    from .tasks import build_tasks

    tasks = build_tasks(world, cell["scenario"], 8)
    task = next((t for t in tasks if t.task_id == cell.get("main_task_id")), None)
    main = {cell.get("main_task_id", "?"): bool(task.check(world))} if task is not None else {}
    sides = score_all(world)
    return {
        "main_task_success": main,
        "side_task_success": {k: bool(v[0]) for k, v in sides.items()},
        "side_task_detail": {k: v[1] for k, v in sides.items()},
    }


def _safe_json(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return {"_raw": str(s)[:500]}


def _jsonable_safe(o: Any) -> Any:
    return _safe_json(json.dumps(o, default=str)) if not isinstance(o, (int, float, str, bool)) else o
