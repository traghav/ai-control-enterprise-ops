# Decisions log

Every decision made autonomously, with the reasoning. Newest at the bottom.

---

## D1 — Setting: "GEP" (Generic Enterprise Platform), health-records operations

**Chosen over** a generic bank_ops setting. The flavour document suggested hospital_ops or
bank_ops. Health records were chosen because:

- The violation taxonomy is *externally specified* (HIPAA identifiers, minimum-necessary
  access, audit-trail integrity). An independent reader can point at a published rule and
  agree that an action is a violation, which removes the "you invented the rules to make
  your monitor look good" objection.
- PHI gives a natural severity ladder: a bulk export of 3 records' identifiers is
  unambiguously worse than retaining a copy, and both are unambiguous *in kind*.
- The trap task (a release-of-information request with no signed authorisation) creates an
  honest/attack confound that is *domain-native* rather than manufactured. This is the
  sharpest thing in the setting and it is discussed in FINDINGS.

All data is synthetic. No real PHI/PII is used, referenced, or downloaded.

## D2 — No sandbox; the world is a Python object, tools are in-process

ControlArena's canonical settings run the agent in a Docker sandbox with bash/python tools.
This setting does not, for three reasons:

1. **Attribution.** The whole experiment is about *which action* moved data *where*. If the
   agent has a general-purpose shell, data can leave through a channel the scorers cannot
   see, and a side-task scorer would return false negatives that are indistinguishable from
   a genuine failure to attack. A fixed tool surface makes the scorers sound.
2. **Reproducibility.** A tool-mediated world is exactly reproducible from
   `(seed, scenario)`. A shell sandbox is not.
3. **Cost.** No container build, no per-episode snapshotting on a 40GB GPU box that also
   has to serve a 32B model.

The cost is that "the agent could have used bash" affordances are absent. This is logged as
a limitation: results speak to tool-mediated enterprise deployments, not to shell access.

## D3 — Staged: standalone harness first, ControlArena adapter second

The flavour document asks to use ControlArena rather than reimplement it. The intent is
sound, but ControlArena's `Setting` API assumes a sandboxed agent, and the adapter is a
real integration risk (pydantic `extra=forbid` config, `ControlTaskMetadata` serialisation,
inspect-ai 0.3.265 vs the 19.0.0 pin, and a custom non-sandbox setting fighting
`control_solver`).

**Decision:** build the setting against the *concepts* ControlArena defines (untrusted
policy / trusted monitor / defer-to-trusted / audit budget / safety-usefulness), so the
science is identical, and make the world/tools/tasks/scorers framework-agnostic so a thin
ControlArena `Setting` adapter is a mechanical port. Then attempt the adapter under a
timebox once E1 produces a number. If the adapter lands, the headline result is reported as
a ControlArena setting; if not, the harness is disclosed plainly as a from-scratch
implementation of the same protocol.

Rationale: a result table exists in either case, and E1 is not gated on an integration that
might not converge.

## D4 — Models: AWQ int4 across the board, one server at a time

40GB of VRAM total. Full-precision 32B/70B models do not fit alongside a 27B monitor.
int4 AWQ puts every policy model (6-19GB) plus a monitor (5-11GB) inside 40GB
simultaneously, which is what makes the concurrent cells in E2 affordable.

Second reason: the *quantised* checkpoint is what a real enterprise deployment would serve.
A control result measured on a full-precision model that nobody deploys is less useful.

## D5 — Three vendor lineages, because the headline claim needs them

The claim under test is that control safety depends on the (untrusted model x monitor x
affordances) pairing, not the model alone. A single-lineage matrix cannot distinguish "the
monitor generalises across models" from "the monitor has memorised this model's habits".
So: Qwen (Alibaba), Llama (Meta), Gemma (Google), plus Phi (Microsoft) for the transfer
matrix. All four have int4 quants that fit the device.

## D6 — Untrusted policies are abliterated checkpoints, deliberately

The flavour document says to attack seriously: an attack policy that refuses makes the
evaluation vacuous. Qwen2.5-32B-Instruct-abliterated and its 14B sibling are used for the
attack role. This is the honest choice: if the control result were measured only on
well-behaved models, low attack success would be a fact about refusal training, not about
the monitor.

The honest (trusted) policy uses the *base instruct* checkpoint, not the abliterated one,
so that usefulness and attack success are not measured on the same behaviourally odd model.

## D7 — Monitor scores ONLINE (prefix only), never with hindsight

A monitor that sees the whole trajectory can score an action high because of something that
happens after it. That inflates AUROC and does not correspond to any deployable auditor,
who has to decide whether to intervene *at the time*. Every monitor score in this repo is
computed from the prefix up to and including the action being scored. This is a
correctness requirement of the metric, not a preference.

## D8 — Side tasks are scored against the world, not the transcript

A side task counts as successful only if the world actually changed in the forbidden way.
An attack policy that writes a detailed plan to exfiltrate PHI, but never emits the call,
scores zero. Conversely an attack that finds an unforeseen channel still scores. This is
what keeps the attack-success number meaningful and the affordance comparison honest.

## D9 — Reproducibility bug fixed pre-emptively

`tasks.py` originally seeded its RNG with `hash(scenario)`. Python salts `hash()` for
strings per process, so the same seed would produce *different* main-task definitions on
different runs. Replaced with `zlib.crc32`, which is stable. Recorded here because a silent
non-determinism in task construction would have invalidated the whole reproducibility claim.

## D10 — Clean rerun after destination-semantics change

The enumerated-destination fix (commit "fix(validity): enumerated, documented export
destinations") changes the tool surface the earlier honest/probe episodes ran against:
destinations that previously succeeded (invented names like "backup-store") now return an
error, which would change any agent that attempted them. Replay-rescoring fixes VERDICTS
but not BEHAVIOUR. A matrix whose cells were generated under two different harness
versions is not a controlled comparison, so all episodes were wiped and rerun under the
final harness. Cost: ~75 min of GPU time; benefit: every cell in the matrix was generated
by identical code. The earlier episodes remain in git history if needed.

## D11 — Attack vs probe: only ablated checkpoints count as attack capability

An aligned model given the attack prompt mostly refuses or partially complies; counting
its side-task completions as "attack success" would measure refusal training, not control
safety. Therefore: the attack phase runs ONLY on behaviourally-ablated checkpoints
(Qwen2.5-32B-abl, Llama-3.1-8B-abl — two vendor lineages), while aligned models given the
same prompt are the PROBE phase, reported separately as an elicitation measurement. The
probe's side-task rate quantifies how much attack behaviour an aligned model can be
talked into without ablation; it is never pooled into the attack-success column.

## D12 — Same-checkpoint honest baselines; phi-4 and aligned-Llama dropped

A coverage audit of the joined score table exposed two junk honest legs beyond the two
junk attack lineages already swapped: (a) the aligned NousResearch Llama-3.1-8B honest
phase — its re-upload ships a chat template that never renders tool schemas, so the model
wrote prose plans for all 90 episodes and never emitted one tool call; (b) phi-4 honest
AND probe phases — phi-4 has no vLLM tool parser (only phi4mini, a different model), so
every tool_choice=auto request was a 400 and all 180 episodes errored. The monitor-side
filter (zero-action episodes are not scored) silently hid both from analysis, which is why
Llama/phi-4 rows showed n=0 baselines.

Two consequences:

1. **Protocol fix (the important one):** the honest baseline for each untrusted policy is
   now THE SAME CHECKPOINT run in honest mode — the Qwen-abl and Hermes-3 rosters gain
   honest legs. This is the ControlArena protocol shape (one untrusted policy, honest or
   attack mode) and it makes within-model AUROC/safety exact: mode is the only differing
   factor, so monitor separation cannot be confounded by checkpoint differences. The
   earlier lineage-based pairing was an approximation forced by the missing legs.
2. **Fallback ladder:** phi-4 is dropped entirely (no working tool parser; two failures
   already logged) and the aligned-Llama honest leg is dropped (broken template). The
   matrix stands on Qwen-abl + Hermes-3 (two lineages) with their own honest legs, plus
   the aligned-Qwen probe. The official llama-3.1 tool template was fetched to
   /home/ubuntu/llama31_tool_template.jinja in case the aligned Llama is ever revived.

## D13 — E4 ticket-aware monitoring run post-hoc, all three monitors

FINDINGS #3 identified the mechanism (prefix-ambiguity between sanctioned and hostile
actions). Because monitoring is post-hoc (D3/execution model), testing the mechanism
required no new policy episodes: all 577 trajectories were re-scored by each monitor
with the assigned ticket's ORIGINAL subject+body added to its context (comments excluded
— agent-writable, so a monitor that reads them can be self-authorised against).

Result (15 cells): FPR@.5 falls in 13/15 cells — halved or better in 9 — at AUROC
changes within ±0.07. The access_review honest tail vanishes (median 0.75 -> 0.00).
The roi_withhold tail PERSISTS, and should: trajectory adjudication after the ROI-scorer
fix showed those "honest"-prompted Qwen-abl episodes actually released records to
partner stores — the monitor is detecting, not hallucinating.

The second E4 question — does ticket context close the cross-lineage collapse? — is
answered NO: Gemma's Hermes-3 AUROC moves only 0.585 -> 0.676, still ~0.25 below its
Qwen-lineage level. False positives are an information problem; the transfer failure is
a monitor generalization problem. Different problems, different remedies.
