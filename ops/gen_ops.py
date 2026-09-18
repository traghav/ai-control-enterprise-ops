"""Generate ops/serve_one.sh from serve.py::ROSTER so the two never drift.

The generated script serves exactly one roster model in the FOREGROUND (tmux owns its
lifetime; nothing here uses pkill, because `pkill -f` matches the invoking shell's own
command line and has repeatedly killed the session that called it).

Run from the repo root:  python3 ops/gen_ops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from serve import ROSTER  # noqa: E402


def main() -> None:
    lines = [
        "#!/bin/bash",
        "# GENERATED from serve.py::ROSTER by ops/gen_ops.py -- do not edit by hand.",
        "#",
        "# Serves exactly one roster model in the FOREGROUND so that tmux owns its lifetime.",
        "# Foreground is deliberate: nothing here uses pkill, because `pkill -f` matches the",
        "# invoking shell's own command line and has repeatedly killed the session that",
        "# called it.",
        "set -u",
        'PORT="${1:?usage: serve_one.sh <port>}"',
        "# The venv's bin/ MUST be on PATH. vLLM shells out to `ninja` during CUDA-graph",
        "# capture; invoking the venv's `vllm` by absolute path does not put `ninja` on PATH,",
        '# and the engine then dies with "FileNotFoundError: ninja" -- which surfaces only',
        '# as a generic "Engine core initialization failed" in the API server log.',
        'export PATH="/home/ubuntu/venv-vllm/bin:$PATH"',
        "export HF_HOME=/home/ubuntu/hf_cache",
        "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}",
        'case "$PORT" in',
    ]
    for s in ROSTER:
        extras = " ".join(s.extra)
        lines.append(
            f"  {s.port}) MODEL={s.name!r}; UTIL={s.gpu_util}; MAXLEN={s.max_len};"
            f" EXTRA=({extras}) ;;"
        )
    lines += [
        '  *) echo "no roster entry for port $PORT" >&2; exit 2 ;;',
        "esac",
        "mkdir -p /home/ubuntu/logs",
        'echo "[serve_one] port=$PORT model=$MODEL util=$UTIL maxlen=$MAXLEN"',
        'exec /home/ubuntu/venv-vllm/bin/vllm serve "$MODEL" \\',
        '  --port "$PORT" \\',
        '  --served-model-name "$MODEL" \\',
        '  --gpu-memory-utilization "$UTIL" \\',
        '  --max-model-len "$MAXLEN" \\',
        '  --no-enable-log-requests \\',
        '  --no-enable-prefix-caching \\',
        '  "${EXTRA[@]:-}"',
    ]
    out = Path(__file__).parent / "serve_one.sh"
    out.write_text("\n".join(lines) + "\n")
    out.chmod(0o755)
    print(f"wrote {out} with {len(ROSTER)} roster entries")


if __name__ == "__main__":
    main()
