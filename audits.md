# LLM Router audit history

This file records completed experiments, changes between policy contracts, and
the evidence that motivated those changes. The current project behavior belongs
in [`README.md`](README.md); this file is historical and should not be read as
the active specification.

In plain language, the README answers "How does the router work now?" This audit
answers "What did earlier runs teach us, and why did the design change?"

## 2026-08-19 — Documentation separated from the current specification

The three experiment retrospectives formerly embedded near the top of the
README were moved here. Historical candidate-removal rationale and references
to superseded result schemas were moved with them. The README now explains the
current analytical-latency POC in present tense and links to this audit.

No routing code, loss equation, training setting, or evaluation gate changed in
this documentation-only update. For example, the sealed-test aggregate quality
gate remains a one-sided 95% retention lower confidence bound of 98%, and the
validation-only margin remains one percentage point, requiring 99% on
validation before the sealed test is opened.

## First analytical dataset-OOD run

### What happened

The first dataset-OOD run completed correctly but routed every sealed-test
prompt to Qwen3-8B. The outcome oracle showed about 4.9% analytical headroom,
so the pipeline and safety guard worked while the learned router did not find a
safe non-fallback route.

In plain language, faster choices existed in hindsight, but the model was not
confident enough to identify them from prompts in unseen datasets. Choosing the
fallback for every prompt was therefore a safe negative result, not evidence of
a useful router.

### Changes made afterward

1. Random-split feasibility became the first experiment; dataset-OOD became a
   separate generalization stress test.
2. Safety BCE became class-balanced instead of learning the majority safe rate.
3. Safety logits received per-candidate Platt calibration, with out-of-fold
   probabilities used for validation threshold selection.
4. Exact dataset identifiers were removed from ModernBERT inputs.
5. A 9B candidate that was slower than the fallback and had a 0% oracle-selection
   rate was removed from the default panel.
6. The mixed-precision loop stopped advancing the learning-rate scheduler when
   `GradScaler` skipped an optimizer step.

The removed candidate was NVIDIA-Nemotron-Nano-9B-v2. In addition to being
analytically dominated in this panel, NVIDIA documents it as a Mamba-2/
Transformer hybrid, so it would require a separate architecture-sensitivity
assumption rather than the ordinary dense-Transformer treatment used here. See
[`nvidia/NVIDIA-Nemotron-Nano-9B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2).

## First successful random-split run

### What happened

Seed 42 routed 20.77% of 2,807 sealed-test prompts to Fin-R1. It retained
99.55% of fallback quality, with a one-sided 95% lower confidence bound of
98.75%, and saved 2.33% analytical latency after the assumed 4 ms router
overhead. This was a successful single-run feasibility result, not a multi-seed
or dataset-OOD claim.

The practical latency arithmetic was:

$$
48.7\text{ ms gross savings}-4\text{ ms router cost}
=44.7\text{ ms net savings per prompt}.
$$

A 20 ms router would have reduced the same frozen policy to about 28.7 ms of
net savings. An overhead near 48.7 ms would have erased the benefit.

### Changes made afterward

The result showed that a strong aggregate number could still conceal leakage,
uncertainty, or group-specific harm. The next contract added:

1. normalized prompt-content hashes and stratified group splitting, so a
   repeated question could not cross random train, validation, and test under
   different benchmark IDs;
2. a one-percentage-point validation safety margin, requiring a 99% retention
   LCB on validation before evaluating against the sealed 98% test gate;
3. a dense threshold grid from 0.85 through 0.95 in increments of 0.005;
4. per-dataset retention, a one-sided Wilson upper bound on quality-loss rate,
   and safety precision among non-fallback routes;
5. exact ModernBERT-tokenizer truncation diagnostics;
6. separate learning rates for LoRA and the new heads, plus validation early
   stopping; and
7. router-overhead sensitivity and a break-even-overhead calculation.

## Completed seed-42 review and the current policy contract

### What the detailed review found

The reviewed seed-42 notebook routed 634 of 2,812 test prompts, gained 54
correct answers, lost 48, and finished six correct answers ahead of fallback.
However, 20 of the net gains came from MBPP, while ARC-Challenge, FinQA, and
GPQA lost 6, 5, and 2 answers respectively. Aggregate quality therefore hid
task-mix sensitivity.

Numerically, $54-48=6$ net answers sounds positive. But one dataset contributing
20 net gains means the other datasets collectively contributed $6-20=-14$ net
answers. That is why the aggregate alone was not accepted as sufficient safety
evidence.

### Changes made afterward

The stricter policy contract requires the router to be safe overall, avoid
large benchmark-group regressions, control uncertainty about harmful routes,
and remain useful when routing is slower than the optimistic estimate. It added
these predeclared gates:

- macro-dataset quality-retention LCB of at least 98%;
- quality-loss-rate one-sided 95% UCL of at most 2.5%;
- routed-safety-precision one-sided 95% LCB of at least 90%;
- worst quality-retention LCB of at least 90% among datasets with at least 100
  evaluation prompts; and
- positive analytical savings at both nominal 4 ms and conservative 20 ms
  router overheads.

The 100-prompt rule prevents an 11-example dataset from controlling the entire
experiment through an extremely wide confidence interval. It is a
catastrophic-harm floor, not a substitute for the stricter 98% aggregate and
macro gates. For example, routed-safety precision with a 92% point estimate but
an 89% lower bound fails the 90% gate.

Because the guarded statistic is the worst among several eligible datasets,
its per-dataset confidence bounds use a Bonferroni adjustment. With ten guarded
datasets and 95% desired family-wise confidence, every one-sided bound uses:

$$
1-\frac{1-0.95}{10}=0.995=99.5\%.
$$

This is deliberately more conservative than taking the minimum of ten ordinary
95% bounds.

Reports were expanded with the Wilson lower bound on routed safety precision,
safe-opportunity recall, conservative-overhead savings, worst guarded-dataset
retention, every candidate's calibrated safety probability, and per-prompt
ModernBERT token length and truncation status. Calibration diagnostics added
the constant-prior Brier baseline, Brier skill, ROC AUC, and safe and unsafe
average precision.

The original saved seed-42 artifact used report schema v3. Reruns made after
these changes use schema v4; their pass/fail contracts are different and their
outcomes must not be compared as though the gates were identical.

## Earlier measured-latency line of work

The repository also retains an earlier experiment that used measured candidate
latencies and a different three-model panel. It is not the recommended current
POC. Its reproducibility contract is documented in
[`llm_router_project_scope_4.md`](llm_router_project_scope_4.md), and its driver
is `notebooks/01_train_modernbert_router.ipynb`.

That experiment reused 2,700 immutable prompt-model measurements: 900 prompts
multiplied by three candidate models. Its version history explains why it
ultimately deployed the fallback and how fallback-relative safety, Platt
calibration, decision-aligned checkpointing, and rank-4 LoRA were explored
before the current analytical-latency POC became the recommended workflow.
