"""History-serialization tests (llama3_json multi-call flattening).

needs_serialized_history / serialized_assistant_turns were added so llama-3.1's chat
template never sees a multi-call assistant turn (it raises "This model only supports
single tool-calls at once!" on RENDERING, so any follow-up request with multi-call
history breaks the agent loop). These tests exercise the pure functions and simulate the
runner's two message-append shapes, asserting the OpenAI pairing invariants hold
exactly: every assistant tool_call id is answered by exactly one tool result under the
same id, with no orphans and no collisions.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from gep.engine import (  # noqa: E402
    needs_serialized_history,
    serialized_assistant_turns,
)

PASS, FAIL = [], []


def check(label: str, cond: bool) -> None:
    (PASS if cond else FAIL).append(label)


# --------------------------------------------------------------------------------------
# Simulation of gep/runner.py run_episode message-append order (serialize branch):
# id_prefix is fixed per model turn from len(ep.actions) BEFORE dispatch; then per call
# j: append the flattened assistant turn, dispatch, append its tool result. The parallel
# (non-serialize) branch keeps one multi-call assistant message and uses model-supplied
# ids.
# --------------------------------------------------------------------------------------


def simulate_serialize_episode(model_turns: list[tuple[str, list[dict]]]) -> list[dict]:
    messages: list[dict] = []
    actions = 0  # mirrors len(ep.actions)
    for content, tool_calls in model_turns:
        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            continue
        id_prefix = f"t{actions}"
        turns = serialized_assistant_turns(content, tool_calls, id_prefix=id_prefix)
        for j in range(len(tool_calls)):
            messages.append(turns[j])
            messages.append({
                "role": "tool", "tool_call_id": f"{id_prefix}_{j}",
                "content": json.dumps({"ok": True})[:6000],
            })
        actions += len(tool_calls)
    return messages


def simulate_parallel_episode(model_turns: list[tuple[str, list[dict]]]) -> list[dict]:
    messages: list[dict] = []
    for content, tool_calls in model_turns:
        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            continue
        assistant_msg: dict = {"role": "assistant", "content": content}
        assistant_msg["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in tool_calls
        ]
        messages.append(assistant_msg)
        for tc in tool_calls:
            messages.append({
                "role": "tool", "tool_call_id": tc["id"],
                "content": json.dumps({"ok": True})[:6000],
            })
    return messages


# --------------------------------------------------------------------------------------
# Conversation validators
# --------------------------------------------------------------------------------------


def conversation_errors(messages: list[dict]) -> list[str]:
    """OpenAI pairing invariants: no orphan tool results, no unanswered assistant calls,
    and no id used twice *within* the call set or *within* the result set. (Assistant
    calls and their results share the same id by design, so cross-set equality — not
    disjointness — is the invariant.)"""
    errors: list[str] = []
    call_ids: list[str] = []
    result_ids: list[str] = []
    for m in messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                call_ids.append(tc["id"])
        elif m["role"] == "tool":
            result_ids.append(m["tool_call_id"])
    if len(set(call_ids)) != len(call_ids):
        errors.append(f"duplicate assistant call ids {sorted(call_ids)}")
    if len(set(result_ids)) != len(result_ids):
        errors.append(f"duplicate tool result ids {sorted(result_ids)}")
    orphans = [i for i in result_ids if i not in set(call_ids)]
    if orphans:
        errors.append(f"orphan tool results {orphans}")
    unanswered = [i for i in call_ids if i not in set(result_ids)]
    if unanswered:
        errors.append(f"unanswered assistant calls {unanswered}")
    return errors


def immediate_pairing_errors(messages: list[dict]) -> list[str]:
    """Serialize-shape invariant: each assistant call is immediately followed by its own
    tool result. (The parallel shape pairs non-adjacently, so this is only applied to
    serialized conversations.)"""
    errors: list[str] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is None or nxt.get("role") != "tool" \
                    or nxt.get("tool_call_id") != tc["id"]:
                errors.append(f"assistant call {tc['id']} not immediately answered")
    return errors


def make_calls(n: int, *, model_ids: bool = False, salt: str = "") -> list[dict]:
    return [
        {"id": f"call_{salt}{j}" if model_ids else f"unused_{salt}{j}",
         "name": f"tool_{j}", "arguments": json.dumps({"j": j, "salt": salt})}
        for j in range(n)
    ]


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_needs_serialized_history() -> None:
    check("needs-serialized-llama3.1",
          needs_serialized_history("NousResearch/Meta-Llama-3.1-8B-Instruct") is True)
    for model in ("Qwen/Qwen2.5-32B-Instruct-AWQ",
                  "Qwen/Qwen3-8B-AWQ",
                  "NousResearch/Hermes-3-Llama-3.1-8B",
                  "ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ"):
        check(f"needs-serialized-false({model.split('/')[-1]})",
              needs_serialized_history(model) is False)


def test_serialized_assistant_turns() -> None:
    content = "hello"
    calls = [{"name": "a", "arguments": "{}"}, {"name": "b", "arguments": "{}"}]
    turns = serialized_assistant_turns(content, calls, id_prefix="t7")

    check("serialized-turn-count", isinstance(turns, list) and len(turns) == 2)
    for j, t in enumerate(turns):
        check(f"serialized-turn{j}-role", t.get("role") == "assistant")
        check(f"serialized-turn{j}-single-call",
              isinstance(t.get("tool_calls"), list) and len(t["tool_calls"]) == 1)
        tc = (t.get("tool_calls") or [{}])[0]
        check(f"serialized-turn{j}-id", tc.get("id") == f"t7_{j}")
        check(f"serialized-turn{j}-type", tc.get("type") == "function")
        fn = tc.get("function") or {}
        check(f"serialized-turn{j}-name-roundtrip", fn.get("name") == calls[j]["name"])
        check(f"serialized-turn{j}-args-roundtrip",
              fn.get("arguments") == calls[j]["arguments"])
    check("serialized-turn0-content", turns[0].get("content") == "hello")
    check("serialized-turn1-content-none", turns[1].get("content") is None)


def test_id_pairing_invariant() -> None:
    for n in range(1, 6):
        content = f"turn with {n} calls"
        calls = make_calls(n)
        model_turns = [(content, calls)]
        messages = simulate_serialize_episode(model_turns)

        # Runner emits ids f"{prefix}_{j}" with prefix fixed per model turn; here the
        # turn is first, so actions == 0 and prefix == "t0".
        expected = {f"t0_{j}" for j in range(n)}
        assistant_ids = [tc["id"] for m in messages if m["role"] == "assistant"
                         for tc in m.get("tool_calls") or []]
        result_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        check(f"pair-n{n}-assistant-ids", set(assistant_ids) == expected)
        check(f"pair-n{n}-result-ids", set(result_ids) == expected)
        check(f"pair-n{n}-no-orphans", conversation_errors(messages) == [])
        check(f"pair-n{n}-immediate-pairing", immediate_pairing_errors(messages) == [])
        check(f"pair-n{n}-counts",
              len(assistant_ids) == n and len(result_ids) == n)


def test_multi_turn_simulation() -> None:
    model_turns = [
        ("first", make_calls(1)),
        ("second", make_calls(3, salt="b")),
        ("third", make_calls(2, salt="c")),
    ]
    messages = simulate_serialize_episode(model_turns)

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    check("multi-assistant-count", len(assistant_msgs) == 6)
    check("multi-tool-count", len(tool_msgs) == 6)

    call_ids = [tc["id"] for m in assistant_msgs for tc in m.get("tool_calls") or []]
    result_ids = [m["tool_call_id"] for m in tool_msgs]
    check("multi-ids-unique",
          len(set(call_ids)) == len(call_ids) == 6
          and len(set(result_ids)) == len(result_ids) == 6)
    # Prefixes mirror len(ep.actions) at each model turn: 0 -> t0, 1 -> t1, 4 -> t4.
    expected = {f"t0_0"} | {f"t1_{j}" for j in range(3)} | {f"t4_{j}" for j in range(2)}
    check("multi-prefixes", set(call_ids) == expected and set(result_ids) == expected)
    check("multi-no-orphans", conversation_errors(messages) == [])
    check("multi-immediate-pairing", immediate_pairing_errors(messages) == [])
    # Content rides on the first assistant turn of each model turn only: the 3-call
    # turn contributes two Nones after "second", the 2-call turn one None after "third".
    contents = [m["content"] for m in assistant_msgs]
    check("multi-content-placement",
          contents == ["first", "second", None, None, "third", None])


def test_non_serialize_branch_sanity() -> None:
    model_turns = [
        ("first", make_calls(4, model_ids=True)),
        ("second", make_calls(2, model_ids=True, salt="b")),
    ]
    messages = simulate_parallel_episode(model_turns)

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    check("parallel-assistant-count", len(assistant_msgs) == 2)
    check("parallel-multi-call-shape",
          [len(m.get("tool_calls") or []) for m in assistant_msgs] == [4, 2])
    check("parallel-no-orphans", conversation_errors(messages) == [])
    call_ids = [tc["id"] for m in assistant_msgs for tc in m.get("tool_calls") or []]
    result_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    check("parallel-ids-unique",
          len(set(call_ids)) == 6 and len(set(result_ids)) == 6
          and set(call_ids) == set(result_ids))
    # The two shapes must not be conflated: serialized flattens to single-call turns,
    # parallel keeps one multi-call assistant message per model turn.
    ser = simulate_serialize_episode([("c", make_calls(4))])
    check("serialize-never-multi-call",
          all(len(m.get("tool_calls") or []) <= 1
              for m in ser if m["role"] == "assistant")
          and len([m for m in ser if m["role"] == "assistant"]) == 4)
    check("serialize-vs-parallel-distinct",
          len([m for m in ser if m["role"] == "assistant"]) != 1)


if __name__ == "__main__":
    test_needs_serialized_history()
    test_serialized_assistant_turns()
    test_id_pairing_invariant()
    test_multi_turn_simulation()
    test_non_serialize_branch_sanity()
    print(f"\n{'='*70}")
    print(f"PASSED: {len(PASS)}    FAILED: {len(FAIL)}")
    if FAIL:
        print("\nFAILURES:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL SERIALIZATION TESTS PASS")
