# Progress log

## 2026-09-18 (day 1)

### Done
- A100 environment: uv, venvs (control-arena 19.0.0 + inspect-ai; vllm 0.29.0), HF token,
  210GB model cache (4 lineages + monitors), tmux-based server lifecycle.
- GEP setting: world model, 7 scenarios, 5 weighted side tasks, FULL/RESTRICTED gate.
- 465/452 scorer tests pass: achievability (oracle trajectories), no-false-positive,
  true-positive, affordance-enforcement, determinism, task-id uniqueness.
- run_matrix.py phased driver (honest/attack/probe/monitor/analyze), resumable JSONL.
- Full smoke test on live Qwen2.5-32B: honest episode solves MT-RC in 2 actions.

### Validity bugs found and fixed (all found by replay/inspection, not guesswork)
1. Cross-world chimera tasks: driver pre-built tasks from a probe world, runner rebuilt
   with a different seed and silently fell back -> unsatisfiable goals fed to agents.
   Now: identical seeds + hard error on any task drift.
2. record_correction had a hidden target value -> unachievable-by-design. Fixed with a
   planted records-office verification comment; engagement check requires an AGENT action.
3. ledger_reconcile: random dispute tickets landed on balanced/empty accounts ->
   unsatisfiable; sign convention unstated. Fixed: planted deterministic imbalance,
   convention stated in the goal.
4. audit_review: random tickets scoped to one MRN; planted anomaly outside that scope ->
   honest work failed. Fixed: dedicated whole-trail monthly-review ticket.
5. ST5 counted internal exports and ticket replies (sanctioned channels) as retention.

### Running
- honest phase on Qwen2.5-32B @ :8020 (56 jobs, ~10s/ep).

### Blocked / none

### Hours left
- ~46 (target: freeze Sunday 06:00 local, experiments frozen Saturday 22:00).

## 2026-09-19 (day 1 evening) — EXPERIMENTS COMPLETE

### Done
- E0 sanity: honest episodes verified achievable end-to-end on live models.
- E1/E2 full matrix: 2 attack lineages x own honest baselines x 2 affordances
  x 3 monitors, 577 scored trajectories/monitor (honest+attack+probe+selective).
- E3 attack-selection leg (Qwen-abl x FULL) + re-monitor.
- Figures: transfer matrix, affordance effect, safety-usefulness (runs/figs/).
- RESULTS.md + FINDINGS.md written from verified numbers.

### Fallback level
- 0 of the ladder used for the core matrix (all cells at full n).
- phi-4 dropped per ladder (D12); aligned-Llama honest leg dropped under its broken template but revived in E6 under the official one (honest + probe).

### Status
- All GPU work complete. Repo write-up in progress.

## 2026-09-19 (day 2) — EXPERIMENTS COMPLETE, WRITE-UP FROZEN

### Done (since last entry)
- D14 power expansion: all policy legs to 10 reps/cell (n=105-115/cell),
  all six monitor passes (3 monitors x {cf, ticket-aware}) over new episodes,
  final analysis + figures regenerated.
- E4 completed on all three monitors (15-cell grid).
- Two more scorer false positives adjudicated and fixed with regression tests:
  multi-ticket authorization clobbering (ST2), bringing the suite to 479 checks.
- RESULTS.md / FINDINGS.md rewritten from powered numbers; headline made consistent.

### Final matrix
- 2 attack lineages x own honest baselines x 2 affordances x (3 monitors x 2 contexts)
- E3 selective leg (n=44); probes on aligned Qwen-32B and (E6) aligned Llama-8B.
- Spontaneous honest violations: 0/678 (frozen matrix; E6 honest legs re-checked post-run). Suite: 479 scorer + 57 prompt + 58 serialization checks green.

### Fallback level
- 0 for the core matrix; phi-4 + aligned-Llama dropped per ladder (D12).

### Status
- ALL WORK COMPLETE as of 02:12 (power expansion) / write-up frozen after review.
- Repo pushed to github.com/traghav/ai-control-enterprise-ops (master).

### Durability
- All raw runs (episodes, monitor scores, analysis, figures) + full code base backed up
  to the durable NFS mount: /lambda/nfs/autocontrol/backup_20260919/ (verified sizes).
  The GitHub repo carries the analysis artifacts; the NFS copy additionally preserves
  every raw JSONL episode and score log.

### Post-freeze notes (Sun 07:00)
- Attempted revival of the aligned-Llama probe (official llama-3.1 tool template via
  --chat-template): the model tool-calls correctly in fresh turns, but the llama3_json
  parser rejects follow-up requests whose history contains multi-tool-call assistant
  turns ("This model only supports single tool-calls at once"). `parallel_tool_calls=False`
  does NOT bypass the guard (it is in the template rendering, not the request validation).
  Fixing requires serializing tool calls in the core agent loop — a post-data harness
  change that would make a revived probe incomparable with the existing matrix, where
  multi-call turns are pervasive (192/226 Hermes attack episodes show n_actions > n_turns).
  DEFERRED -> SUPERSEDED (E6, 2026-09-20): history serialization landed in
  gep/engine.py (MODELS_NEEDING_SERIALIZED_HISTORY + serialized_assistant_turns) and
  gep/runner.py; the aligned Llama was revived under the official tool template
  (ops/templates/PROVENANCE.md) and its honest + probe legs ran as E6 (ops/e6_seq.sh).
- All servers stopped; GPU clean. 9674 raw log lines preserved in runs/ (committed) and
  on durable NFS.

### E5 added post-freeze (defer-to-trusted, the protocol the flavour doc asked for)
- Phase 1 counterfactual over 452 main-phase episodes; phase 2 real continuations n=18.
- 92/95 successful attacks averted (safety 0.968 cf @0.5); ticket context cuts honest
  deferrals 23% -> 14% at safety 0.926. Trust-continuation assumption validated (0/18).
- gep/defer.py + scripts/e5_analyze.py + scripts/e5_phase2.py; raw sample in runs/.
- GPU stopped and verified 0 MiB; NFS backup refreshed with E5 code + data.
