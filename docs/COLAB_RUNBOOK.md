# Colab v5 input-representation runbook

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
