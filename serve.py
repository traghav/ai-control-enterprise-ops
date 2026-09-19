"""Model serving: manage vLLM servers on fixed ports.

One process per model, fixed ports so the matrix driver can address them statically.
Ports are chosen from 8000 upward and never reused within a run.

GPU budget note (40GB total): the largest policy (abliterated Qwen2.5-32B int4, ~19GB) plus
the largest monitor (Gemma-3-27B int4, ~18GB) is ~37GB, which fits. Everything else is
smaller. `gpu_memory_utilization` is set per model to leave a small reserve.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

VENV_VLLM = "/home/ubuntu/venv-vllm/bin"
HF_HOME = "/home/ubuntu/hf_cache"


@dataclass
class ModelSpec:
    name: str                      # HF repo id
    port: int
    family: str                    # vendor lineage
    role: str                      # policy | monitor
    gpu_util: float = 0.40
    max_len: int = 16384
    extra: tuple[str, ...] = ()


# The matrix's model roster. Ports are stable and documented in README.
#
# EXECUTION MODEL: exactly ONE server runs at a time (see run_matrix.py). Two vLLM servers
# started concurrently on one GPU profile against each other's allocations and race during
# startup; both then fail KV-cache init nondeterministically ("Available KV cache memory:
# -5.84 GiB" with nothing else loaded). Sequential loading is therefore not an optimisation
# but a correctness requirement. Utils below are sized for SOLO operation.
#
# TOOL-CALL PARSERS: policy models need --enable-auto-tool-choice plus a family-appropriate
# --tool-call-parser, or every request that sets tool_choice="auto" is rejected with 400
# ("auto tool choice requires --enable-auto-tool-choice and --tool-call-parser"). Monitors
# never use tools, so their ports carry no parser flags.
ROSTER: list[ModelSpec] = [
    # --- untrusted policies (phases: honest / attack / probe) ------------------------
    ModelSpec("Qwen/Qwen2.5-32B-Instruct-AWQ", 8020, "qwen", "policy", 0.85,
              extra=("--enable-auto-tool-choice", "--tool-call-parser", "hermes")),
    # Aligned Llama-3.1-8B revived with the OFFICIAL tool-capable chat template (the
    # NousResearch re-upload ships a template that never renders tool schemas -> junk
    # legs, D12). Multi-call history is serialized per-request in gep/engine.py
    # (MODELS_NEEDING_SERIALIZED_HISTORY), because the llama3_json parser cannot
    # re-render multi-call assistant turns.
    ModelSpec("NousResearch/Meta-Llama-3.1-8B-Instruct", 8021, "llama", "policy", 0.70,
              extra=("--enable-auto-tool-choice", "--tool-call-parser", "llama3_json",
                     "--chat-template", "/home/ubuntu/llama31_tool_template.jinja")),
    ModelSpec("stelterlab/phi-4-AWQ", 8022, "phi", "policy", 0.70),
    ModelSpec("ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ", 8023, "qwen",
              "policy", 0.85,
              extra=("--enable-auto-tool-choice", "--tool-call-parser", "hermes")),
    ModelSpec("NousResearch/Hermes-3-Llama-3.1-8B", 8024, "llama", "policy", 0.70,
              extra=("--enable-auto-tool-choice", "--tool-call-parser", "hermes")),
    # --- trusted monitors (phase: monitor) --------------------------------------------
    ModelSpec("Qwen/Qwen3-8B-AWQ", 8030, "qwen", "monitor", 0.60),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct-AWQ", 8031, "qwen", "monitor", 0.60),
    ModelSpec("gaunernst/gemma-3-27b-it-int4-awq", 8032, "gemma", "monitor", 0.80),
]

BY_NAME = {s.name: s for s in ROSTER}


def log_dir() -> Path:
    p = Path("/home/ubuntu/logs")
    p.mkdir(exist_ok=True)
    return p


def is_up(port: int, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def served_model_id(port: int) -> str | None:
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        return r.json()["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return None


def start(spec: ModelSpec, wait: bool = True, timeout: int = 1500) -> bool:
    """Start a vLLM server for `spec` unless one is already healthy on its port."""
    if is_up(spec.port):
        print(f"[serve] port {spec.port} already up ({spec.name})")
        return True

    env = dict(os.environ)
    env["HF_HOME"] = HF_HOME
    env["HF_HUB_OFFLINE"] = "1"
    env["VLLM_USE_V1"] = "1"
    cmd = [
        f"{VENV_VLLM}/vllm", "serve", spec.name,
        "--port", str(spec.port),
        "--served-model-name", spec.name,
        "--gpu-memory-utilization", str(spec.gpu_util),
        "--max-model-len", str(spec.max_len),
        "--dtype", "auto",
        # vLLM 0.29 renamed this flag; `--disable-log-requests` is gone and the server
        # exits with an argparse error if it is passed.
        "--no-enable-log-requests",
        "--no-enable-prefix-caching",
        *spec.extra,
    ]
    logf = log_dir() / f"vllm_{spec.port}.log"
    print(f"[serve] starting {spec.name} on :{spec.port} -> {logf}")
    with open(logf, "ab") as fh:
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)

    if not wait:
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_up(spec.port):
            print(f"[serve] {spec.name} ready on :{spec.port} after {time.time()-t0:.0f}s")
            return True
        time.sleep(5)
    print(f"[serve] TIMEOUT starting {spec.name}; see {logf}")
    return False


def stop(port: int) -> None:
    subprocess.run(["pkill", "-f", f"vllm serve .* --port {port}"], check=False)


def status() -> dict[str, bool]:
    return {s.name: is_up(s.port) for s in ROSTER}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps(status(), indent=2))
    elif cmd == "start":
        for name in sys.argv[2:]:
            start(BY_NAME[name])
    else:
        raise SystemExit(f"unknown command {cmd}")
