# Funded validation plan

## Plain-language plan

Investment does not guarantee that a larger run will perform better. It buys a
sequence of experiments that converts the current single-run result into a
commercial decision. Each milestone has a pass, narrow, or stop outcome so a
weak generalization result is reported rather than optimized away.

## Technical milestones and exit criteria

### Milestone 1 — Reproducible random-split feasibility

Run `random_seed_42`, `random_seed_43`, and `random_seed_44` with the same three
setups, loss, calibration, threshold grid, and schema-v5 gates. Publish every
artifact and a cross-seed table. Do not change the seed-42 test after observing
it. Exit evidence includes activation rate, quality-retention LCB, routed safety
precision LCB, harm UCL, macro and guarded retention, conservative savings, and
feasible threshold-block size for all three seeds.

Example: if two seeds route 12% but one routes 0%, the conclusion is “feasible
but unstable,” not an average 8% routing claim.

### Milestone 2 — Dataset-OOD stress test

Run `dataset_ood_seed_42`, `dataset_ood_seed_43`, and `dataset_ood_seed_44` as
separate artifacts. Entire datasets must remain disjoint. If the router fails
closed on OOD while random splits pass, reposition the product as a domain-tuned
router and test customer-specific adaptation rather than claiming universal
generalization.

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

Add candidates only when pre-collected quality outcomes and sourced analytical
model facts are available. Require a non-dominated panel: every retained model
must be selected by the outcome oracle on a non-zero fraction of prompts. Favor
meaningful latency separation because a 7.0B versus 8.2B panel leaves limited
room after router overhead.

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
