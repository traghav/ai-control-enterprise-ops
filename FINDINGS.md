# Findings

## Headline

**In a reproducible enterprise-ops control setting (GEP), measured control safety is a
property of the (untrusted policy x trusted monitor x affordance configuration) pairing,
not of any single factor.** Three concrete separations demonstrate this: (1) monitor
AUROC collapses by 0.28-0.41 when the same monitors are moved from a Qwen-lineage policy
to a Llama-lineage one, while affordance restriction leaves overall attack success
unchanged (0.41 -> 0.43) because the attack substitutes between side tasks; (2) the
honest false-positive tail is *policy- and task-dependent* -- it saturates at the score
ceiling for role-governance and release-withholding tasks where honest actions are
prefix-identical to violations; (3) the false positives are largely *removable*: giving
the monitor the assigned ticket's original authorisation context cuts FPR@0.5 by 30-100%
(relatively) across cells at unchanged AUROC (E4). Consequence for practice: a monitor
validated on one model family and one tool configuration gives no warranty about its
behaviour on another; restricting affordances buys a change in *which* violations occur,
not a reduction in *whether* they do; and the cheapest large monitor win is not a bigger
model but the authorisation context the deployment already has.

## Supporting evidence

1. **Cross-lineage monitor transfer degrades sharply.** With FULL affordances, mean
   monitor AUROC against Qwen-lineage policies is 0.94 (Gemma-27B) and 0.94 (Qwen-8B
   monitors); against the Llama-lineage policy (Hermes-3) it is 0.53 and 0.60-0.73. The
   monitors' calibration does not transfer across vendor lineages even though the
   violation taxonomy is identical. (RESULTS.md, transfer matrix + figure.)

2. **Affordance restriction substitutes rather than suppresses.** Removing http_post and
   the partner stores eliminated every sev-4 exfiltration fire (ST1: 12 -> 0, ST3:
   7 -> 0) but ST4 privilege escalation rose from 0 to 15 -- the directory tool remains
   -- and overall attack success stayed within noise (0.41 vs 0.43). The ranking of
   affordance configurations depends on which side task the auditor weights, i.e. on the
   severity table, not on attack success alone.

3. **Prefix-ambiguity bounds online monitoring.** Honest access_review and audit_review
   episodes sit at median suspicion 0.75, honest ROI-withhold episodes at 1.00 (the
   ceiling): "export this record to a partner because the ticket says so" and "export
   this record to a partner to exfiltrate it" are indistinguishable from the action
   prefix alone. The budget-based safety metric therefore saturates (safety@1% = 0.00 in
   several cells despite TPR@0.5 = 1.0 at FPR 0.24), and per-action monitors cannot use
   the one signal that disambiguates -- the ticket thread, which records what was
   authorised and what was not.

4. **Elicitation asymmetry between checkpoints.** The ablated Qwen-32B and the aligned
   Qwen-32B reach nearly identical attack success under the same attack prompt (0.41 vs
   0.43) while their honest usefulness is equal (0.68 both), so "aligned" and "attack-
   capable" are not opposites under a strong elicitation prompt; the Hermes-3 checkpoint
   is the opposite case (attack success 0.05, honest usefulness 0.05 -- it is not
   reliable in either mode at following the multi-step workflow).

5. **Authorisation context is the cheapest monitor upgrade (E4).** Re-scoring all 577
   trajectories with the same Qwen3-8B monitor plus the assigned ticket's original
   content cut FPR@0.5 from 0.25 to 0.18 (Qwen-abl FULL), 0.24 to 0.11 (RESTRICTED),
   0.09 to 0.00 (Hermes-3), 0.23 to 0.11 (probe FULL), 0.17 to 0.04 (probe RESTRICTED),
   at essentially unchanged AUROC (+0.007/+0.013/+0.068/-0.007/-0.033). The honest
   access_review tail vanished (median 0.75 -> 0.00). The monitor did not need to be
   bigger; it needed the information that distinguishes "sanctioned by the ticket" from
   "self-authorised" -- information every real deployment already has.

6. **World-state scoring is what makes the numbers mean anything.** Attack plans are
   frequent in transcripts; completed violations are not. Spot-checks of 10 attack
   trajectories show exfiltration attempts with hallucinated MRNs ("123456789"), blocked
   export attempts retried through other channels, and one attempted-but-insufficient
   exfiltration (1 record, below the 3-record bar). Without world-state scoring, these
   would each be counted differently by transcript-based judges.

## What didn't work (and is itself informative)

- **In-prompt attack selection (E3) is not elicitable.** The start/stop policy ("attack
  only when a side task is naturally reachable; begin with REACHABILITY: YES/NO") was
  obeyed in 1/44 episodes; measured attack success (0.50) was within noise of the
  always-attack policy (0.41). Selection-based stealth cannot be claimed from this
  experiment -- the policy was never expressed by the model.
- **Two Llama-lineage uncensored fine-tunes lost tool-calling.** mlabonne's abliterated
  and Orenguteng's Lexi-Uncensored wrote tool calls as markdown prose for entire 90-
  episode legs (the tool schemas reached the model -- the prose cites exact ticket
  values -- but the chat templates never render the tool-call output syntax). Hermes-3
  (tool-call trained) worked.
- **phi-4 has no working vLLM tool parser** (only phi4mini, a different model); its
  honest and probe legs were all-error. The aligned NousResearch Llama-3.1 re-upload
  ships a chat template that never renders tool schemas. Both were dropped (D12).
- **Scorers had real bugs that only trajectory-level adjudication caught.** The un-
  authorised ROI release passed `prog_roi_withhold` for a full honest leg (the audit
  entry's target field stores the destination, not the MRN; and "consent" keyword-
  matching matched the release justification itself). An honest access-review episode
  tripped ST4 because the sanction check required the ticket to be closed. All are fixed
  with regression tests (477 scorer checks, 57 prompt checks). The monitors were RIGHT
  about the ROI episodes; the world-scorer was wrong -- a caution against treating any
  single judge as ground truth.

## Limitations

- **n = 42-46 per cell; ~40 policy-hours.** Bootstrap CIs at these sizes are about
  +/-0.14: differences below that (e.g. the FULL-vs-RESTRICTED attack-success delta) are
  not resolved. The transfer-matrix collapse (0.28-0.41 AUROC) and the E4 FPR reductions
  are far outside this band.
- **One untrusted model per lineage, one seed family.** Lineage-level claims rest on a
  single checkpoint per lineage; within-lineage variance is unmeasured.
- **No shell access.** The world is tool-mediated (D2) for attribution and
  reproducibility; attacks through a general-purpose shell are out of scope, so the
  attack surface is narrower than a bash-sandbox setting.
- **Monitors see the ticket text but not the model's chain-of-thought.** The E4 variant
  still scores actions prefix-only; a monitor that could also defer *mid-episode* on
  ticket context (rather than scoring post-hoc) is untested.
- **The audit budget formalism saturates** when the honest tail sits at the ceiling;
  we report TPR/FPR at fixed thresholds alongside it, but the field-standard safety
  number is degenerate in exactly the cells where monitoring is hardest.
- **Synthetic world.** Tools and data are mock; the violation taxonomy is externally
  specified (HIPAA-shaped) but the domain is simulated. No real PHI anywhere.

## Next experiment

1. **Defer-on-ticket-context**: combine ticket-aware scoring (E4) with
   defer-to-trusted-upon-suspicion, and measure whether the FPR reduction converts into
   a usefulness win at matched safety -- the operational payoff of the E4 result.
2. **Severity-weighted affordance comparison**: with the substitution effect documented,
   re-rank FULL vs RESTRICTED under explicit severity weights and a larger n, instead of
   raw attack success.
3. **Elicitation-hardened attack selection**: E3's start/stop policy needs to be trained
   or scaffolded (not just prompted) before the selection hypothesis can be tested.
4. **More lineages at matched capability**, separating checkpoint scale from lineage,
   to test whether the transfer collapse is a lineage effect or a capability effect.

