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
ROSTER: list[ModelSpec] = [
    # --- untrusted policy candidates -------------------------------------------------
    ModelSpec("huihui-ai/Qwen2.5-32B-Instruct-abliterated", 8001, "qwen", "policy", 0.46),
    ModelSpec("Qwen/Qwen3-32B-AWQ",                          8002, "qwen", "policy", 0.46),
    ModelSpec("huihui-ai/Qwen2.5-14B-Instruct-abliterated",  8003, "qwen", "policy", 0.28),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct",            8004, "llama", "policy", 0.22),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct-AWQ",                8005, "qwen", "policy", 0.18),
    ModelSpec("stelterlab/phi-4-AWQ",                        8006, "phi",  "policy", 0.30),
    # --- trusted monitor candidates ---------------------------------------------------
    ModelSpec("Qwen/Qwen3-8B-AWQ",                           8010, "qwen", "monitor", 0.22),
    ModelSpec("gaunernst/gemma-3-27b-it-int4-awq",           8011, "gemma", "monitor", 0.45),
    ModelSpec("stelterlab/phi-4-AWQ",                        8012, "phi",  "monitor", 0.30),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct-AWQ",                8013, "qwen", "monitor", 0.18),
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
        "--disable-log-requests",
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
