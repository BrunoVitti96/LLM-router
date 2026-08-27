# Calibrated ModernBERT LLM router — investor brief

## The opportunity

Organizations often send every prompt to their strongest model because a cheap
router is valuable only if it does not create an unacceptable quality loss. This
project trains ModernBERT to answer a narrow, auditable question: “Can a smaller
model preserve the recorded quality of the trusted fallback for this prompt?”
The selector uses a smaller tier only when calibrated safety and analytical
speed gates both pass; otherwise it fails closed.

The initial customer is a high-volume enterprise AI platform team operating
multiple models behind one API. The likely buyer is a head of AI platform,
inference engineering, or AI FinOps. Inference providers and API gateways are
secondary customers. Low-volume, single-model teams are not the initial target.

## What the latest evidence actually says

The latest completed run is notebook 03's random seed 42 using Qwen2.5-1.5B,
3B, and 7B. It evaluated 1,726 sealed-test prompts, routed 636 (36.85%) to a
smaller tier, gained 40 correct answers, lost 32, and finished eight answers
ahead of fallback. Aggregate quality retention was 101.08%, with a one-sided
95% lower bound of 99.19%, above the 98% gate.

The run still has `single_run_passed=False`. Its macro-dataset quality-retention
lower bound missed the predeclared 98% gate. `bbh_object_counting` contributed
+10 net answers, more than the complete run's +8, so the other tasks were -2
without it. The harm-rate upper bound was 2.468%, only 0.032 percentage points
inside its 2.5% ceiling.

Analytical savings were 28.90% at the frozen 4 ms overhead and 28.24% at 20 ms.
ModernBERT end-to-end overhead was 46.66 ms at p50 and 63.00 ms at p95, below
the analytical break-even value of 711.49 ms. Candidate Qwen latency was not
measured, so these are feasibility economics rather than production savings.

## What v3 demonstrated

Notebook 03 replaced the old $7.0/8.2=85.4\%$ parameter ratio with three pinned
Qwen2.5 capacity tiers: 1.54B, 3.09B, and 7.61B. The small tier is only
$1.54/7.61=20.2\%$ of the strong tier's parameter count, which created materially
more analytical room for router overhead in seed 42.

V3 downloads pinned Open LLM Leaderboard per-example details for all three tiers;
it never loads Qwen weights. After removing two overlapping GPQA variants, it
keeps at most 300 aligned prompts from each of 37 tasks—at most 11,100 prompts
and 33,300 recorded prompt-model outcomes. Evidence repository revisions,
evaluation run IDs, task exclusions, and sampling are fingerprinted. Candidate
latency remains an analytical BF16 scenario.
The router still uses validation-only setup and threshold selection, per-candidate
Platt calibration, duplicate-content grouping, subgroup and harm gates, two
adjacent feasible thresholds, and exactly one sealed-test opening per run.

## What is differentiated

- The router predicts fallback-relative safety, not absolute correctness.
- Two independent safety heads can choose small, middle, or fallback tiers.
- Calibration and threshold selection are out-of-fold and validation-only.
- Prompt hashes prevent duplicate content crossing random splits.
- Policies need a stable threshold region, not one lucky grid value.
- Reports expose harm, subgroup retention, truncation, and router overhead.
- Candidate quality evidence is pinned; candidate latency remains analytical.

## Honest commercial position

This is not yet a “we reduced production cost” story. It is a disciplined
prototype with promising aggregate economics and an explicit cross-dataset
failure. The investable proposition is the evidence system and bounded
validation plan: three random seeds, three dataset-OOD seeds, target-hardware
candidate timing, and a design-partner pilot.

A defensible statement today is: “Our first sealed three-tier run routed 36.9%
of prompts to smaller models and estimated 28.2% analytical latency savings,
while a predeclared macro-dataset safety gate prevented us from calling the run
a pass.” A guaranteed percentage or dollar-saving claim would exceed the
evidence. The complete decision memo is
[`INVESTOR_READINESS_MEMO.md`](INVESTOR_READINESS_MEMO.md).

## What v4 now tests

Notebook 04 is the new, unrun investor-facing experiment. It trains only the
deployed safety objective for all 15 epochs, restores the minimum-validation-loss
checkpoint, and defaults to holding out complete datasets. The oracle code stays
compatible but has coefficient zero. For example, safety loss 0.223 plus an
oracle diagnostic 0.477 still yields v4 training loss 0.223.

Its OOD dashboard shows the retained learning curve, the validation
routing/savings frontier, quality retention versus routing for each unseen
dataset, and model allocation with harmful routes. V4 results must come from
the exported artifact; the clean notebook contains no claimed result.

## Fundable next step

Funding should buy validation rather than optimism: preserve and verify the
seed-42 artifact, complete the three versioned OOD and three random v4 runs,
measure complete target-hardware serving economics, and run a paid design
partnership with aggregate and per-domain quality audits. The milestone has
three acceptable outcomes: general router, domain-specific router, or a
negative result that safely stays on fallback.
