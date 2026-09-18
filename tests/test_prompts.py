"""Prompt-construction tests.

build_attack_system had a latent str.format() KeyError that only surfaced when the first
probe leg ran, because honest-only smoke tests never built the attack prompt. These tests
exercise every prompt builder for every affordance config so a broken prompt is caught
without GPU time.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from gep.policies import (  # noqa: E402
    HONEST_SYSTEM,
    build_attack_system,
    build_selective_attack_system,
    user_message,
)
from gep.engine import tool_schemas  # noqa: E402

PASS, FAIL = [], []


def check(label: str, cond: bool) -> None:
    (PASS if cond else FAIL).append(label)


def test_prompt_builders() -> None:
    check("honest nonempty", bool(HONEST_SYSTEM.strip()))
    for include_http in (True, False):
        for builder, name in ((build_attack_system, "attack"),
                              (build_selective_attack_system, "selective")):
            try:
                p = builder(include_http)
            except Exception as e:  # noqa: BLE001
                check(f"{name}-builds(http={include_http})", False)
                print(f"   {name}(http={include_http}) raised: {e!r}")
                continue
            check(f"{name}-builds(http={include_http})", bool(p.strip()))
            # Every side task must be listed.
            for sid in ("ST1", "ST2", "ST3", "ST4", "ST5"):
                check(f"{name}-mentions-{sid}(http={include_http})", sid in p)
            check(f"{name}-http-gate(http={include_http})",
                  ("http_post" in p) == include_http)
            # No unresolved template placeholders.
            check(f"{name}-no-placeholder(http={include_http})", "__HTTP_TOOL__" not in p)
            # Tool names the prompt cites must exist in the schema (typo guard).
            for tool in ("records_search", "records_update", "audit_export",
                         "directory_set_role"):
                check(f"{name}-cites-{tool}(http={include_http})", tool in p)


def test_tool_schemas() -> None:
    for aff in ("FULL", "RESTRICTED"):
        schemas = tool_schemas(aff, include_http=(aff == "FULL"))
        names = {s["function"]["name"] for s in schemas}
        expected = {"records_search", "records_read", "records_update", "records_export",
                    "tickets_list", "tickets_read", "tickets_reply", "tickets_close",
                    "ledger_query", "ledger_post", "audit_query", "audit_export",
                    "directory_list", "directory_set_role", "directory_remove"}
        check(f"schemas-cover-tools({aff})", expected <= names)
        check(f"schemas-http-gate({aff})", ("http_post" in names) == (aff == "FULL"))
        # JSON-schema sanity: every tool has a parameters object with string-typed keys.
        ok = all(isinstance(s["function"]["parameters"].get("properties"), dict)
                 for s in schemas)
        check(f"schemas-parameters-object({aff})", ok)


def test_user_message() -> None:
    msg = user_message("Do the thing.", {"ticket_id": "TKT-1", "subject": "s"})
    check("user-message-has-goal", "Do the thing." in msg)
    check("user-message-has-ticket", "TKT-1" in msg)


if __name__ == "__main__":
    test_prompt_builders()
    test_tool_schemas()
    test_user_message()
    print(f"\n{'='*70}")
    print(f"PASSED: {len(PASS)}    FAILED: {len(FAIL)}")
    if FAIL:
        print("\nFAILURES:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL PROMPT TESTS PASS")
