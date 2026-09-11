# Funded validation plan

> **Current status — 2026-09-11:** The old Qwen detail rows use obsolete MATH
> format scores. Notebook 06 now regrades recorded math responses under the
> pinned `qwen-math-verify-v2` contract. A boxed 968 against gold 968 changes
> from old score 0 to correctness 1. Earlier V4/V5/V6 performance claims below
> are historical and must not be presented as corrected results. A fresh Colab
> training/evaluation is required; see [README](../README.md) and the
> [current runbook](COLAB_RUNBOOK.md). Funding milestones should validate these
> corrected labels and customer serving economics before claiming savings.


## Plain-language plan

Investment does not guarantee that a larger run will perform better. It buys a
sequence of experiments that converts the current single-run result into a
commercial decision. Each milestone has a pass, narrow, or stop outcome so a
weak generalization result is reported rather than optimized away.

## Technical milestones and exit criteria

### Milestone 0 — Isolate the input bottleneck

Run notebook 05's `prefix_512`, `prefix_1024`, and `head_tail_1024` variants on
the same dataset-OOD seed-42 split. Hold loss, rank-4 LoRA, labels, calibration,
thresholds, and gates fixed. Validation chooses the representation; only that
winner opens the sealed test. For a 1,600-token prompt, this compares discarding
1,088 prefix-tail tokens with discarding 576 tokens or preserving both ends.

Advance the winning representation only if it materially improves unseen-task
validation, not merely training loss. If all three remain near the notebook-04
ROC-AUC of 0.5467, the bottleneck is more likely labels/domain coverage or model
capacity than truncation, and the next funded experiment should change one of
those factors under a new versioned contract.

### Milestone 1 — Dataset-OOD stress test

Run `qwen25_v5_context_ood_seed_42`, `qwen25_v5_context_ood_seed_43`, and
`qwen25_v5_context_ood_seed_44` as separate artifacts using the validation-
selected representation. Entire datasets must remain disjoint. Every run uses
the same rank-4 safety-only loss, all five
epochs, per-mini-batch loss logging, validation-best checkpoint, calibration,
threshold grid, and gates. Publish every
artifact and a cross-seed table. The shared sampled Open LLM Leaderboard evidence
must keep one fingerprint across all seeds. Exit evidence includes activation
rate, quality-retention LCB, routed safety
precision LCB, harm UCL, macro and guarded retention, conservative savings, and
feasible threshold-block size for all three seeds, plus the OOD dashboard.

If the router fails closed on OOD, reposition the product as a domain-tuned
router and test customer-specific adaptation rather than claiming universal
generalization.

Example: if aggregate OOD retention is 99% but one held-out dataset retains only
94% while routing 40% of its prompts, the domain chart must expose that failure;
the 99% aggregate must not be presented alone.

### Milestone 2 — Reproducible random-split feasibility

Run `qwen25_v5_context_random_seed_42`, `qwen25_v5_context_random_seed_43`, and
`qwen25_v5_context_random_seed_44` under that same v5 contract. Do not mix
notebook-03 or notebook-04 artifacts into the v5 cross-seed table. For example, if two seeds
route 12% but one routes 0%, the conclusion is “feasible but unstable,” not an
average 8% routing claim.

### Milestone 3 — Router-only target-hardware economics

Measure batch-one ModernBERT model-only and end-to-end latency after warmup on a
named GPU and software stack. Candidate LLM latency remains analytical. Compare
ModernBERT p50 and p95 with the frozen 4 ms assumption, 20 ms conservative
assumption, and each policy’s break-even overhead.

For example, if break-even is 26.9 ms, p50 is 14 ms, and p95 is 31 ms, median
requests remain economically viable but tail requests do not. The next action is
router batching, distillation, caching, or a larger candidate latency gap—not a
claim that the 14 ms median applies to every request.

### Milestone 4 — Stronger candidate economics

Evaluate the pinned, published Qwen2.5 1.54B/3.09B/7.61B detail datasets without
rerunning the candidates. Require every retained tier
to be selected by the outcome oracle on a non-zero fraction of prompts. The
parameter ratios are 20.2% and 40.6% for the small and middle tiers relative to
7.61B, compared with 85.4% for the historical 7.0B/8.2B pair. If a tier has zero
oracle usage or is no better than a faster tier, remove it before a production
panel is frozen.

### Milestone 5 — Design-partner pilot

Primary customer profile: an enterprise AI platform team with multiple LLMs,
centralized request traffic, measurable quality outcomes, and enough volume for
single-digit percentage savings to matter. Secondary profiles are inference
providers and API gateways. Pilot outputs are request volume, fallback rate,
aggregate router overhead, analytical candidate savings, sampled quality audits,
and operator overrides. No prompt content needs to be disclosed in an investor
report.

### Milestone 6 — Production decision

Proceed only if quality bounds, router overhead, operational reliability, and
customer economics all remain acceptable. Otherwise choose one explicit outcome:
domain-specific routing, a smaller/distilled router, a revised candidate panel,
or fallback-only. The milestone is complete when the decision is supportable,
not only when the desired answer is positive.

## Commercial customer definition

| Segment | Pain | Buyer | Why this router | Initial qualification |
|---|---|---|---|---|
| Enterprise AI platform | Strongest-model default is expensive | Head of AI Platform / AI FinOps | Safety-calibrated fallback policy | Multiple models and high request volume |
| Inference provider | Needs differentiated serving economics | VP Engineering / Product | Router as serving-layer feature | Controls model fleet and telemetry |
| API gateway | Customers need policy control across providers | Infrastructure product lead | Auditable probabilities and fallback | Can integrate per-request routing |

The first sales motion should be a paid design partnership or technical pilot,
not a promise of a universal autonomous router.
