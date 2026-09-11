# Colab V6 scoring-v2 runbook

The old math detail rows used obsolete format-based scores. The corrected
notebook grades saved answers using pinned Math-Verify before training. A boxed
968 against gold 968 now scores 1 instead of inheriting the old zero. The old
V6 model cannot be reused: its training labels, class weights, calibration, and
threshold were based on the previous scoring contract.

## Immediate rerun using the source ZIP

1. Run `python scripts/build_colab_bundle.py` locally. This writes
   `dist/colab/llm_router_colab_scoring_v2.zip` and a copy of notebook 06.
2. Upload that notebook in Colab and choose a **fresh GPU runtime**. An A100 is
   preferred; T4 is supported and training can take several hours.
3. In Colab's Files sidebar upload `llm_router_colab_scoring_v2.zip` directly to
   `/content`. Do not extract it manually. Setup validates and extracts the
   bundle, installs dependencies, and verifies the scoring contract.
4. Enable your `HF_TOKEN` Colab secret. Accept the three gated evidence datasets
   with the same account; details are in the historical access section below.
5. Keep `RUN_ID = "qwen25_v6_scoring_v2_ood_seed_42"` and **Run all**.
6. Before training, inspect `qwen_scoring_reconciliation.csv`: all full-task
   counts, task coverage, binary/provenance checks and non-math means must pass.
   Math source-aggregate differences are diagnostic and shown explicitly; local
   pinned scores are used. For example, 7B counting locally scores 55/123 versus
   the source's 57/123. We never assign labels to force agreement with a total.
7. Let all five epochs finish. Epoch selection, calibration, thresholds and test
   evaluation run afresh. Download the new result ZIP before closing Colab.

Without a source ZIP, setup fetches `poc` with checked Git commands. Both the
corrected notebook and Python helpers must already be on that branch. Updating
only the notebook is insufficient. Restart after source/dependency changes.

The source ZIP includes package files and a SHA-256 manifest, not credentials,
model weights, old results or the local environment. It allows a rerun before
these local changes are pushed. A source-bundle hash is recorded in exported
provenance when the bundle is used.

## Scoring and experiment identity

The scorer uses Math-Verify 0.5.2, latex2sympy2-extended 1.0.6, ANTLR runtime
4.13.2 and SymPy 1.13.3. Gold is parsed from the full original worked solution;
candidate text is parsed from the recorded response. Empty/unusable gold or
incompatible dependencies abort. Non-math metrics remain the published binary
scores. Linux/Colab retains upstream symbolic timeouts; Windows validation uses
an isolated process with a hard watchdog rather than silently disabling limits.

The report preserves original/updated scores and raw grading evidence, including
all pre-cap math rows in `qwen_math_rescored_records.parquet`. Older source
aggregates cannot all be reproduced from the stored responses and do not pin
an exact rescoring environment; the audit preserves that limitation explicitly.

The Qwen router still uses the first 1,023 tokens plus its one-token
`<|endoftext|>` sentinel, rank-4 LoRA, five epochs and effective batch size four.
For example, 5,274 prompts produce 1,319 optimizer updates per epoch. The
safety label remains `candidate_quality >= fallback_quality`. If a fallback
score is corrected from 0 to 1 and the candidate stays 0, safety changes from
1 to 0. No candidate generation or policy-gate change is introduced.

Further development runs use `qwen25_v6_scoring_v2_ood_seed_43` or `_44`;
random runs use `qwen25_v6_scoring_v2_random_seed_{42,43,44}`. Existing seeds
reuse benchmark evidence, so reserve new task families/customer data for
independent confirmation. Old V5/V6 metrics are not comparable baselines.
The 4/20 ms overhead settings remain assumptions; inspect measured timing too.

## Historical V5 input-representation runbook

The following is retained for history. Its MATH detail labels are obsolete;
use the corrected notebook 06 workflow above for new work.

## Goal

Run `notebooks/05_train_modernbert_input_representation_ood_v5.ipynb` in a
fresh GPU Colab. The notebook tests whether the 512-token prefix caused the weak
notebook-04 OOD validation result. It downloads published Qwen prompts, answers,
and correctness scores; it does not load Qwen model weights.

The default `qwen25_v5_context_ood_seed_42` run compares three representations
on exactly the same split:

| Variant | Budget | Strategy |
|---|---:|---|
| `prefix_512` | 512 | beginning only; v4 control |
| `prefix_1024` | 1,024 | beginning only |
| `head_tail_1024` | 1,024 | beginning plus ending |

For example, a 1,600-token input loses 1,088 tokens under the control and 576
under either 1,024-token representation. Only validation chooses the winner;
only the winner opens the sealed test.

## One-time Hugging Face access

Accept access to all three auto-gated Open LLM Leaderboard detail datasets and
create a read-capable token:

- `open-llm-leaderboard/Qwen__Qwen2.5-1.5B-Instruct-details`;
- `open-llm-leaderboard/Qwen__Qwen2.5-3B-Instruct-details`; and
- `open-llm-leaderboard/Qwen__Qwen2.5-7B-Instruct-details`.

Add it to Colab Secrets as case-sensitive `HF_TOKEN` and enable notebook access.
If it is absent, the notebook shows a hidden session-only prompt. A 401 after a
valid token normally means one dataset's terms were not accepted or the token
lacks read access to gated repositories.

## Run order

1. Select **Runtime → Change runtime type → GPU**.
2. Run every cell from top to bottom with the default OOD seed 42.
3. Verify the evidence audit has exactly three binary outcomes per retained
   prompt. The maximum panel is $37\times300=11{,}100$ prompts and 33,300
   prompt-model outcomes.
4. Inspect the parallel-training preflight.
5. Let all three variants finish five epochs. Each worker prints every optimizer
   step with its variant name; validation is evaluated once per epoch.
6. Confirm each variant restores its minimum-validation-loss epoch and that the
   selected test setup came only from the validation comparison.
7. Inspect the four-panel representation/OOD dashboard and run the final Gradio
   prompt showcase.
8. Download the ZIP before closing Colab.

The next repeat IDs are `qwen25_v5_context_ood_seed_43` and
`qwen25_v5_context_ood_seed_44`. Random-split diagnostics use
`qwen25_v5_context_random_seed_{42,43,44}`. Do not mix v3, v4, and v5 artifacts
because their experiment contracts differ.

## Parallel GPU behavior

`PARALLEL_TRAINING_REQUESTED=True` asks for three Python workers on one GPU.
The notebook enables them only when the GPU reports at least 14 GiB total memory
and at least 80% free memory. Each worker uses batch size 4. Model construction
is serialized to reduce simultaneous allocation spikes.

This is a memory-based preflight, not a guarantee: three models also share GPU
compute and memory bandwidth. If CUDA reports out-of-memory, restart the runtime,
set `PARALLEL_TRAINING_REQUESTED=False`, and rerun all cells. Do not continue
from a partially failed parallel run. Sequential fallback changes wall-clock
time, not the scientific comparison.

## Frozen training contract

Every variant uses rank-4 LoRA with alpha 8, the same class-balanced safety BCE,
and zero oracle coefficient:

$$
\mathcal L_{v5}=\mathcal L_{safety}+0\mathcal L_{oracle}.
$$

Thus, safety loss 0.223 plus oracle diagnostic 0.477 still optimizes 0.223. Each
variant runs all five epochs with early stopping disabled. If its validation
losses are 0.239, 0.241, 0.236, 0.241, and 0.241, epoch 3 is restored.

## Download checklist

- `qwen_quality_audit.csv` has only complete binary outcomes;
- `input_representation_comparison.csv` contains all three variants;
- `parallel_training_facts.json` records enabled workers and memory facts;
- `v5_input_representation_contract.json` records budgets, strategies, and loss;
- every setup has training, calibration, and input diagnostics;
- the selected setup alone has sealed-test decisions;
- the dashboard's values agree with its CSV data;
- ModernBERT p50/p95 are compared with analytical break-even;
- the final prompt demo uses the selected tokenization strategy; and
- explicit gate failures are retained, including fallback-only results.

A lower truncation rate is not sufficient. The useful result is lower unseen-task
validation loss or stronger validation ranking, followed by sealed-test quality
and harm gates that pass.
