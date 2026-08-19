# Colab multi-run and demo runbook

## What to run

Open `notebooks/02_train_modernbert_hybrid_poc.ipynb` in a fresh GPU Colab. In
the experiment-control cell, change only `RUN_ID`. Run and download one ZIP at a
time in this order:

1. `random_seed_43`
2. `random_seed_44`
3. `dataset_ood_seed_42`
4. `dataset_ood_seed_43`
5. `dataset_ood_seed_44`

`random_seed_42` already has a completed schema-v5 artifact in
`results/modernbert_hybrid_random_seed_42_v3/`. Rerun it only to verify
reproducibility, not to replace the observed test with a more favorable result.

## What remains identical

Every run imports the same immutable setup menu: rank-4 hybrid, rank-4
safety-only, and rank-4 dataset-balanced hybrid. It also keeps eight maximum
epochs, minimum epoch 2, patience 2, the same calibration method, gate values,
threshold grid, candidate model facts, and analytical output-length policy.
The scenario identity date remains `2026-08-19`; it does not change with the
calendar date of a confirmation run.

Numerical example: choosing `random_seed_44` changes the grouped fold assignment
and training seed from 43 to 44. It does not change the 98% sealed-test retention
gate, 90% routed-precision LCB gate, or 20 ms conservative overhead.

## ModernBERT timing

The notebook samples 100 validation prompts after 10 warmup requests and reports
batch-one model-only and end-to-end p50/p95. End-to-end includes tokenizer,
host-to-device transfer, and ModernBERT. Fin-R1 and Qwen3-8B are not loaded,
called, or timed. The measured distribution is diagnostic and does not alter the
frozen policy after validation selection.

Exported timing files are:

- `modernbert_overhead_benchmark.json`;
- `modernbert_overhead_samples.csv`; and
- `modernbert_overhead_comparison.csv`.

## Interactive demo

After export, the Gradio cell launches a temporary Colab share link. Enter a
prompt and an estimated candidate-token count. The JSON result shows calibrated
safety probability, eligibility, selected model, whether fallback was used,
analytical latency for every candidate, measured ModernBERT overhead for that
request, and estimated net savings.

The demo is a routing visualization. It does not generate an answer because no
candidate LLM is loaded.

## Download checklist

Confirm the ZIP name matches `RUN_ID`, then download it before closing Colab.
Keep random and dataset-OOD ZIPs in separate result directories. Report failed
and fallback-only runs alongside passing runs.
