# Colab v3 Qwen-tier runbook

## Plain-language run order

Open `notebooks/03_train_modernbert_qwen_tiers_poc.ipynb` in a fresh GPU Colab.
The first run builds one shared quality-evidence cache from Qwen2.5-1.5B,
Qwen2.5-3B, and Qwen2.5-7B. It then trains and evaluates the router. Later runs
reuse the cached candidate answers, so only the ModernBERT router is retrained.

Run and download one ZIP at a time in this order:

1. `qwen25_random_seed_42`
2. `qwen25_random_seed_43`
3. `qwen25_random_seed_44`
4. `qwen25_dataset_ood_seed_42`
5. `qwen25_dataset_ood_seed_43`
6. `qwen25_dataset_ood_seed_44`

Authorize Google Drive when prompted. Candidate checkpoints are written after
each dataset, so a disconnected Colab resumes rather than repeating completed
generations. Keep the Drive evidence directory; do not rename partial Parquet
files or mix them with a different `EVIDENCE_TAG`.

## Frozen evidence contract

V3 samples 150 prompts from each of GSM8K, ARC-Challenge, MMLU, BoolQ,
HellaSwag, and WinoGrande. That is 900 prompts and:

$$
900\text{ prompts}\times3\text{ Qwen tiers}=2{,}700\text{ scored outcomes}.
$$

The evidence tag freezes model and dataset revisions, the sampling seed, prompt
template, deterministic decoding, NF4 4-bit quantization, batch size, and maximum
input length. If any of these changes, the tag changes and the old cache is not
silently reused.

The exact candidates are 1.54B, 3.09B, and 7.61B parameters. These are the
closest official Qwen2.5 instruction checkpoints to the requested 1B/3B/8B
tiers. The fallback is chosen using training quality; it is not forced to be 7B.

## Frozen router contract

Every router run keeps the same rank-4 hybrid, rank-4 safety-only, and rank-8
hybrid setups; eight maximum epochs; minimum epoch 2; patience 2; calibration;
threshold grid; gate values; candidate facts; and analytical output-length
policy. Only the split mode and seed change with `RUN_ID`.

For example, changing seed 42 to 43 changes prompt fold membership and model
initialization. It does not change the 98% sealed-test retention LCB gate, 90%
routed-precision LCB gate, 20 ms conservative overhead, or requirement for two
adjacent feasible thresholds.

## Timing contract

Qwen generation creates offline quality evidence. It is not accepted as candidate
latency evidence. Candidate latency remains analytical and uses parameter count,
4-bit precision, prompt length, expected output length, and declared hardware
assumptions.

After validation freezes the setup and threshold, the notebook measures up to
100 batch-one ModernBERT validation requests after 10 warmups. End-to-end timing
includes tokenizer, host-to-device transfer, and ModernBERT. It exports:

- `modernbert_overhead_benchmark.json`;
- `modernbert_overhead_samples.csv`; and
- `modernbert_overhead_comparison.csv`.

Numerical example: if break-even is 80 ms, p50 is 43 ms, and p95 is 68 ms, both
median and tail router overhead fit the analytical latency opportunity. If p95 is
95 ms, median economics pass but tail economics do not.

## Result and demo checklist

The Gradio demo shows calibrated safety for both smaller tiers, eligibility,
selected model, fallback use, and analytical latency. It does not load Qwen or
generate a live answer.

Before closing Colab:

- confirm all 2,700 candidate outcomes exist;
- confirm the ZIP name matches `RUN_ID`;
- save the evidence tag and pinned contract with the report;
- inspect every explicit failure reason;
- compare ModernBERT p50 and p95 with break-even;
- download the ZIP; and
- keep random and dataset-OOD results separate.

Report failed and fallback-only runs alongside passing runs. A fallback-only run
shows that the guard worked; it does not show that prompt-level routing works.
