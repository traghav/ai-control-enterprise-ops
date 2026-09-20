# Findings

## Summary

Measured control safety in this environment depends on the combination of untrusted policy, trusted monitor, and affordance configuration. No single factor determines it. Four results support this:

1. Monitor AUROC drops from about 0.93 on Qwen-lineage policies to 0.53-0.72 on the ablated Hermes-3 policy. E6 shows the drop is specific to that checkpoint, not to the model family: the aligned Llama probe is scored at 0.89-0.97 by the same monitors.
2. Restricting affordances does not reduce overall attack success (0.39 vs 0.45, within the CI), because attacks substitute between side tasks. Severity-weighted violation mass falls 18%.
3. The honest false-positive tail depends on policy and task. It saturates at the score ceiling for role-governance and release-withholding tasks, where honest and hostile actions are indistinguishable from the action sequence alone.
4. Ticket context removes most of those false positives (FPR@0.5 down in 14 of 15 cells at unchanged AUROC), but does not restore detection on the ablated checkpoint.

Under the defer protocol, the monitors avert 92 of 95 successful attacks on the Qwen-abl lineage, but the same monitors and threshold reach only 0.400-0.600 safety on the ablated Hermes-3 lineage. The transfer failure appears as direct safety loss in deployment.

## Evidence

### 1. Monitor transfer

Context-free monitor AUROC against Qwen-lineage policies is 0.93 (Gemma-3-27B) and 0.93 (mean of the two Qwen monitors). Against the ablated Hermes-3 policy it is 0.56 and 0.67. Against the aligned Llama probe (E6) it is 0.92 and 0.94. n=105-115 episodes per cell. The violation taxonomy is identical across all policies; the monitors fail specifically on the ablated fine-tune's violation patterns. See RESULTS.md and runs/figs/transfer_matrix.png.

### 2. Affordance restriction changes which violations occur

Removing http_post and the partner stores eliminated all sev-4 exfiltration completions (ST1: 28 to 0, ST3: 19 to 0). Privilege escalation rose from 0 to 28 and silent edits from 5 to 29, because the directory and record tools remain. Overall attack success moved from 0.39 to 0.45, within the CI at n=110-115 per cell. Severity-weighted violation mass (weights 4/3/4/3/1) fell from 1.83 to 1.50 per episode, an 18% reduction. Whether an organisation gains from restriction therefore depends on whether it weights violations by severity: by raw count there is no detectable gain, by weighted mass there is a moderate one.

### 3. Prefix-ambiguity bounds per-action monitoring

Honest access_review and audit_review episodes have median max-suspicion 0.75; honest ROI-withhold episodes reach 1.00. Exporting a record because the ticket requires it and exporting a record to exfiltrate it look the same in the action prefix. The audit-budget safety metric saturates as a result (0.00 in several cells despite TPR@0.5 of 0.94-1.00 at FPR 0.15-0.31). Per-action monitors cannot use the one signal that disambiguates these cases: the ticket, which records what was authorised.

### 4. Elicitation asymmetry between checkpoints

The ablated Qwen-32B and the aligned Qwen-32B have statistically indistinguishable attack success under the same attack prompt (0.39 vs 0.43 FULL; 0.45 vs 0.50 RESTRICTED), and the same honest usefulness (0.70-0.73). The aligned Llama probe also exceeds its ablated counterpart (0.22/0.07 vs 0.05/0.00), though both are low because this checkpoint follows multi-step workflows unreliably. Refusal training did not prevent completed violations under a fixed elicitation prompt in either lineage.

The release-of-information task shows this at the world level. In honest mode the ablated Qwen moved the record to a partner store in 10 of 20 episodes; the aligned Qwen did so in 3 of 20, with consent-verification language in the ticket thread while the record moved. The Hermes-3 checkpoint produced 0 releases but also only 1 correct withholding in 20.

### 5. Ticket context removes false positives; it does not restore detection

Re-scoring all trajectories with the assigned ticket's original content (subject and body; comments excluded, since the agent writes those) reduced FPR@0.5 in 14 of 15 cells, halving it or better in 10, with AUROC changes within ±0.04. The honest access_review tail disappeared (median 0.75 to 0.00, n=58). The same context moved Gemma's Hermes-3 AUROC only from 0.668 to 0.704, against 0.93 on Qwen-lineage policies. False positives are an information problem solvable with data the deployment already has. Detection on the ablated checkpoint is a generalization problem, and context does not solve it.

### 6. Defer-to-trusted: strong on one lineage, and the transfer failure becomes safety loss

Simulating the defer protocol over all 452 main-phase Qwen-abl episodes: 92 of 95 successful attacks averted (safety 0.968, context-free Qwen3-8B monitor at threshold 0.5). Attacks become suspicious early, before the cover story is complete. The 3 escapes were all ST2 silent single-field edits in RESTRICTED (ST2 escape rate 0.09; 0.00 for the other side tasks). Restriction therefore shifts attacks toward the side task that deferral catches least well.

Ticket context reduced honest deferrals from 52/226 (23%) to 31/226 (14%), at safety 0.926: 6 fewer attacks averted in exchange for 21 fewer unnecessary trusted continuations.

Real trusted continuations (n=18: 12 deferred honest episodes, 6 averted attacks) produced 0 violations and recovered the main task in 11 of 18. Recovery on the remaining episodes was bounded by the checkpoint's own low honest withhold rate on that scenario.

On the ablated Hermes-3 lineage the same monitors degrade into safety loss: Gemma-3-27B reaches 0.600 safety at threshold 0.5 and 0.400 at threshold 0.75 (5 successful attacks; wide intervals), against 0.968 on Qwen-abl. An operator who validated a monitor on an aligned model and deployed it against an abliterated one would see this directly.

### 7. World-state scoring vs transcript scoring

Attack plans appear in transcripts far more often than completed violations. The spot-checked attack trajectories include exfiltration attempts with hallucinated record IDs, blocked channels retried through other tools, and one exfiltration below the 3-patient threshold. A transcript-based judge would score these differently from the world-state scorers, which count only completed environment changes.

## What did not work

- In-prompt attack selection. The start/stop policy (attack only when a side task is reachable, declared via a REACHABILITY line) was followed in 1 of 44 episodes. Attack success under it (0.51) was within the CI of the always-attack policy (0.39). The selection hypothesis is untested, because the policy was never expressed by the model.
- Two uncensored Llama fine-tunes (mlabonne abliterated, Lexi-Uncensored) lost tool-calling: their chat templates do not render the tool-call output format, so the models wrote tool calls as markdown text. Both legs were discarded. Hermes-3, which is tool-call trained, worked.
- phi-4 has no working vLLM tool parser (only phi4mini, which is a different model). Its honest and probe legs all errored and were discarded.
- Four scorer false-positive classes were found by reviewing trajectories and fixed with regression tests: a planted records-office comment counted as agent engagement; a ticket-scoped audit review could not see a trail-wide anomaly; an authorised ROI export passed the withhold scorer (the audit entry stores the destination, not the record ID, and the keyword "consent" matched the release justification); a sanctioned promotion tripped the escalation scorer because the check required the ticket to be closed. In two of the four cases the monitor's score was correct and the environment scorer was wrong.

## Limitations

- n=105-115 per main-matrix cell (~10 GPU-hours for the policy legs, plus the E6 legs and monitor passes). Bootstrap CIs are about ±0.09. The transfer collapse and the E4 FPR reductions are outside this band. The Hermes-3 defer numbers rest on 5 successful attacks and carry wide intervals. E3 remains n=44.
- One checkpoint per (lineage, role). Within-lineage variance is not measured.
- The environment is tool-mediated; there is no shell. Attacks through arbitrary code execution are out of scope.
- Monitors see actions and results, not chain-of-thought. Deferral was evaluated post-hoc from logged scores, not in a live loop.
- The audit-budget safety metric degenerates when the honest tail saturates at the score ceiling. Fixed-threshold TPR/FPR and the defer simulation are reported alongside it.
- The environment is synthetic. The violation taxonomy follows HIPAA-shaped rules; the domain is simulated.

### 8. The adaptive attacker redistributes suspicion but does not evade deferral (E8)

Giving the attacker the monitor's scoring guide and adaptation rules (small increments,
ticket-scoped tools only, no bulk patterns) shifted the side-task profile away from bulk
exfiltration toward silent edits (ST1 28 to 12; ST2 5 to 26) and cost the monitors about
0.09 AUROC on all three. But attack success was unchanged (0.41 vs 0.39) and 93 of 107
adaptive episodes still crossed the deferral threshold: the max-suspicion rule only
needs one action to cross, and the adaptive policy's increments still accumulate to
that. The adaptation interacts with the ST2 escape finding, however -- it produces the
violation class deferral catches least well.

## Next experiments

1. Defer with ticket context in the loop: whether the FPR reduction converts into a usefulness win at matched safety when the trusted policy takes over mid-episode.
2. Severity-weighted affordance comparison at larger n, now that the substitution effect and the 18% mass reduction are documented.
3. Scaffolded or trained attack selection: the in-prompt version was not followed by the model, so selection-based stealth remains untested rather than refuted.
4. More checkpoints per lineage at matched capability, to separate lineage effects from capability effects in the transfer result.
