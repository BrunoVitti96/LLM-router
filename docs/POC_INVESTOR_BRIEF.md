# Calibrated ModernBERT LLM router — POC investor brief

## The opportunity

Organizations often send every prompt to their strongest model because a cheap
router can save money only if it does not create an unacceptable quality loss.
This project trains ModernBERT to answer a narrower, auditable question: “Can a
faster candidate preserve the recorded quality of the trusted fallback for this
prompt?” The deployed selector chooses a faster candidate only when calibrated
safety and analytical speed gates both pass; otherwise it fails closed.

The initial customer is a high-volume enterprise AI platform team operating two
or more LLMs behind a shared API. The economic buyer is typically a head of AI
platform, inference engineering, or AI FinOps. Secondary customers are inference
providers and API gateways that can expose routing as an infrastructure feature.
Low-volume teams using one model are not the initial target because small routing
savings may not justify another production component.

## Current evidence

The schema-v5 random-split seed-42 run evaluated one frozen policy on 2,812
sealed-test prompts. It routed 345 prompts (12.27%) from Qwen3-8B to Fin-R1,
gained 27 correct answers, lost 15, and finished 12 answers ahead of fallback.
Its quality-retention point estimate was 100.60% and its one-sided 95% lower
confidence bound was 100.07%, above the predeclared 98% test gate. Routed safety
precision was 95.65% with a 93.46% lower bound. All aggregate, macro-dataset,
harm-rate, guarded-dataset, conservative-overhead, and threshold-stability gates
passed.

Candidate latency is analytical. Under the frozen scenario, mean fallback
latency was 1.8846 seconds. The router saved 1.22% at an assumed 4 ms overhead
and 0.37% at 20 ms; break-even overhead was 26.93 ms. Numerically, 1.8846 seconds
multiplied by 1.22% is about 22.9 ms net savings per request. At one million
requests that is approximately 22,900 aggregate seconds, or 6.4 compute-hours,
before translating hardware time into a customer-specific monetary value.

## What is genuinely differentiated

- The router predicts fallback-relative safety, not absolute correctness.
- Platt calibration and validation-only threshold selection are out-of-fold.
- Normalized prompt hashes prevent duplicate question content crossing splits.
- The test opens only after setup and threshold selection freeze.
- A policy needs at least two adjacent feasible thresholds, not one lucky point.
- Reports expose harm, subgroup retention, truncation, calibration, and overhead.
- The exported artifact restores the best validation epoch rather than the last
  overfitted epoch.

## Limitations stated plainly

This is single-run feasibility evidence, not a production or universal-routing
claim. MBPP contributed +18 net answers while the overall net was +12; excluding
MBPP leaves the other datasets at -6. Calibration was strong, but AUROC was
0.739 and the policy recalled only 15.31% of safe faster opportunities. Candidate
latency has not been measured, by design, and ModernBERT overhead still needs a
named target-hardware distribution. Multi-seed and dataset-OOD evidence is not
yet complete.

## Fundable next step

Funding would validate rather than assume scale benefits: complete random seeds
42/43/44, run separate dataset-OOD seeds, measure only the ModernBERT decision
path on target hardware, expand to a non-dominated candidate panel with existing
quality results, and run a high-volume design-partner pilot. Candidate latency
will remain analytical during this POC phase so capital is spent on routing
evidence rather than repeatedly loading large candidate LLMs in Colab.

The near-term investment proposition is therefore: a disciplined feasibility
result, a reproducible evidence pipeline, and a bounded plan to determine whether
the router becomes a general platform, a domain-specific product, or a negative
result that safely defaults to fallback.
