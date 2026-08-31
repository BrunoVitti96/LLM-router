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

The latest completed run is notebook 04's dataset-OOD seed 42. Training loss
fell from 0.2220 to 0.2023, while validation loss was 0.2390, 0.2414, 0.2360,
0.2408, and 0.2411. Epoch 3 was retained, but validation ROC-AUC was only
0.5467. The model fit its training tasks without learning a reliable safety
ranking for unseen tasks.

At 512 tokens, 64.26% of examples were truncated. Every one of the 588 routed
test prompts was truncated, while none of the 500 non-truncated prompts was
routed. The 1.5B routes gained 13 answers and lost none; the 3B routes gained 28
but lost 35. The run failed macro-quality and harm-bound gates. This association
does not prove truncation caused the failure, which is why v5 tests it directly.

Notebook 03's random seed 42 remains promising historical evidence: it routed
36.85% of test prompts, estimated 28.24% analytical savings at 20 ms overhead,
and ended eight answers ahead of fallback. It also failed its macro-dataset
gate. Candidate Qwen latency was not measured in either run.

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

A defensible statement today is: “A random-split run showed promising routing,
but the first dataset-OOD run did not generalize; our safety gates blocked
deployment and our next controlled test isolates lost prompt context.” A
guaranteed percentage or dollar-saving claim would exceed the evidence. The complete decision memo is
[`INVESTOR_READINESS_MEMO.md`](INVESTOR_READINESS_MEMO.md).

## What v5 now tests

Notebook 05 compares `prefix_512`, `prefix_1024`, and `head_tail_1024` while
holding the safety-only loss, rank-4 LoRA, split, calibration, and gates fixed.
For a 1,600-token input, the variants discard 1,088, 576, and 576 middle tokens,
respectively. Validation selects one representation and only that winner opens
the sealed test. The three variants request concurrent GPU execution with batch
size 4 and fall back to sequential execution when the memory preflight fails.

Its dashboard compares learning curves, truncation rate versus validation loss,
validation routing versus savings, and the selected model's OOD domain outcomes.
V5 is unrun; its clean notebook contains no claimed result.

## Fundable next step

Funding should buy validation rather than optimism: run the controlled v5 input
test, repeat the winning representation across OOD and random seeds,
measure complete target-hardware serving economics, and run a paid design
partnership with aggregate and per-domain quality audits. The milestone has
three acceptable outcomes: general router, domain-specific router, or a
negative result that safely stays on fallback.
