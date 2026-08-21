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

The latest completed notebook-02 run used random seed 44 and a narrow Fin-R1
7.0B/Qwen3-8B 8.2B panel. It routed 623 of 2,805 sealed-test prompts (22.21%),
gained 43 correct answers, lost 52, and ended nine answers behind fallback. Its
aggregate quality-retention lower bound was 98.73%, but it failed three other
predeclared gates: macro-dataset quality, routed safety precision, and guarded
worst-dataset quality. Therefore `single_run_passed=False` is the correct result.

The task mix explains why an attractive average would be misleading. MBPP gained
17 net answers, while FinQA lost nine and MMLU-Pro lost seven. Numerically,
$43-52=-9$. The router was useful on some code prompts and unsafe on several
knowledge/reasoning groups.

Economics were positive only around the median. Analytical savings were 2.52%
at the frozen 4 ms router assumption and 1.68% at 20 ms. The frozen policy broke
even at 52.14 ms. Measured ModernBERT end-to-end overhead was 42.55 ms at p50 and
67.23 ms at p95, so p50 was viable while p95 was not.

## What v3 changes

Notebook 03 replaces the old $7.0/8.2=85.4\%$ parameter ratio with three pinned
Qwen2.5 capacity tiers: 1.54B, 3.09B, and 7.61B. The small tier is only
$1.54/7.61=20.2\%$ of the strong tier's parameter count, creating materially
more analytical room for router overhead.

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
prototype that found a negative result, diagnosed why, and built a stronger test.
The investable proposition is the evidence system and bounded validation plan:
three random seeds, three dataset-OOD seeds, target-hardware router timing, and a
design-partner pilot. V3 must be run before publishing any new savings claim.

A defensible LinkedIn statement today is: “We built a calibrated router that
failed closed when a narrow model panel was unsafe, and we are now testing a
three-tier Qwen panel with a fivefold parameter span.” A claim that the product
already saves a guaranteed percentage or dollar amount would exceed the evidence.

## Fundable next step

Funding should buy validation rather than optimism: ingest and audit the v3
published evidence, publish all six run artifacts, measure ModernBERT p50/p95 on
target hardware, expand beyond the capped public panel, and run a paid design partnership
with aggregate quality audits. The milestone has three acceptable outcomes:
general router, domain-specific router, or a negative result that safely stays
on fallback.
