# Investor presentation — safety-first LLM routing

## 1 — The opportunity

Many prompts do not need the largest model. This project uses a small
ModernBERT router to choose the fastest Qwen tier predicted to preserve the
7B fallback's recorded quality. Low-confidence prompts stay on the fallback.

Example: at a frozen safety threshold of 0.87, predicted safety of 0.92 makes a
smaller tier eligible; 0.81 does not. Candidate speed is still analytical, not
measured production latency.

## 2 — What has been built

- pinned per-prompt results for Qwen2.5 1.5B, 3B, and 7B;
- rank-4 LoRA training on ModernBERT;
- calibrated safety probabilities and validation-only threshold selection;
- sealed testing with quality, harm, subgroup, and savings gates; and
- a fail-closed interactive demonstration.

The optimized loss is class-balanced safety BCE. The old oracle diagnostic
remains compatible but has coefficient zero. Numerically, safety loss 0.223 and
oracle loss 0.477 produce $0.223+0\times0.477=0.223$.

## 3 — What the latest OOD run taught us

Notebook 04's dataset-OOD seed 42 did not pass. Training loss fell from 0.2220
to 0.2023, but validation loss was 0.2390, 0.2414, 0.2360, 0.2408, and 0.2411.
The retained epoch was 3 and validation ROC-AUC was 0.5467. The model learned
the training tasks but barely ranked safe replacements on unseen tasks.

The strongest clue is representation: 64.26% of prompts were truncated at 512
tokens. All 588 routed test prompts were truncated; none of the 500
non-truncated prompts was routed. The 1.5B routes were +13 net, while 3B routes
were 28 gains and 35 losses, or -7 net. Macro-quality and harm-bound gates
rejected the run. This is useful evidence because the safety system refused to
turn a weak OOD model into a deployment claim.

## 4 — The controlled v5 test

Notebook 05 changes only the router's input representation:

| Variant | What the router sees |
|---|---|
| `prefix_512` | first 512 tokens; exact v4 baseline |
| `prefix_1024` | first 1,024 tokens; tests more context |
| `head_tail_1024` | beginning plus ending; tests lost tail content |

For a 1,600-token prompt, these variants discard 1,088 tokens, 576 tokens, and
576 middle tokens respectively. Loss, LoRA rank, split, seed, calibration,
thresholds, and gates remain fixed. Validation selects one representation; only
that winner opens the sealed test. V5 is built but unrun, so it has no result yet.

The three variants request parallel execution on one GPU with batch size 4 per
worker. A memory preflight falls back to sequential execution on smaller or busy
GPUs. Parallel execution may shorten the experiment, but shared GPU compute
means it should not be described as three independent GPUs.

## 5 — Charts investors should inspect

1. Three train/validation loss curves with each retained epoch.
2. Truncation rate versus best validation loss for all representations.
3. Validation routed fraction versus conservative analytical savings.
4. For the selected representation, held-out-domain routing versus quality.

The decision is positive only if longer or head-tail context improves unseen-task
validation and the selected model subsequently passes the sealed safety gates.
A prettier training curve alone is insufficient.

## 6 — Investment position

The investable asset is the auditable safety-and-evidence system, not a proven
savings percentage. Funding should verify the representation result across
multiple OOD seeds, measure complete target-hardware serving economics, and run
a design-partner pilot. Acceptable outcomes are a general router, a
domain-specific router, or a negative result that safely remains on fallback.
