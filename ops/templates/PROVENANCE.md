# Model template provenance

`ops/templates/llama31_tool_template.jinja`

- **Source repo**: `unsloth/Meta-Llama-3.1-8B-Instruct` (ungated mirror of the official
  Meta-Llama-3.1-8B-Instruct tokenizer configuration).
- **Source file**: `tokenizer_config.json` → `chat_template` field (4,614 chars).
- **Fetched**: 2026-09-18, `curl https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/raw/main/tokenizer_config.json`,
  template extracted verbatim and validated to contain `"tools"` and `"python_tag"`
  (the official tool-calling markers the NousResearch re-upload's template lacks).
- **Why**: the NousResearch/Meta-Llama-3.1-8B-Instruct re-upload ships a chat template
  that never renders tool schemas, so the model writes tool calls as prose (see D12).
  Serving that checkpoint with `--chat-template ops/templates/llama31_tool_template.jinja`
  plus `--enable-auto-tool-choice --tool-call-parser llama3_json` restores tool-calling.
- **Companion harness change**: the llama3_json parser cannot re-render assistant turns
  containing multiple tool calls ("This model only supports single tool-calls at once!" —
  a RENDER-time guard in the template, not request validation). `gep/engine.py`
  (`MODELS_NEEDING_SERIALIZED_HISTORY`) + `gep/runner.py` flatten multi-call model turns
  to sequential single-call turns for this model. World behavior is identical.
- **Used for**: port 8021 (aligned NousResearch/Meta-Llama-3.1-8B-Instruct — honest
  baseline + probe, E6).
- **Verification**: live smoke test produced 3 real tool-call actions with zero errors
  under probe mode (2026-09-20, runs/episodes_probe.jsonl E6 rows).

Note: vLLM renders this template at SERVE time; the file must exist on the serving host
at the path given in `serve.py` / `ops/serve_one.sh` (the generated launcher copies the
repo path). The template text is unchanged from upstream; no local edits were made.
