# Colab v3 published-Qwen-evidence runbook

## Plain-language run order

Open `notebooks/03_train_modernbert_qwen_tiers_poc.ipynb` in a fresh GPU Colab.
It downloads published per-example prompts, Qwen answers, and correctness scores,
then trains ModernBERT. It does **not** download or run Qwen model weights.

The canonical notebook currently uses the repository's `poc` branch. Its first
cell fetches and switches `/content/LLM_Router` to `poc`, installs that checkout,
and confirms `src/llm_router/qwen_evidence.py` exists. If an older Colab session
already cloned `develop`, rerunning the setup cell switches that checkout before
imports. A missing helper is reported as a repository/source mismatch rather
than a generic Python import failure.

Run and download one ZIP at a time:

1. `qwen25_random_seed_42`
2. `qwen25_random_seed_43`
3. `qwen25_random_seed_44`
4. `qwen25_dataset_ood_seed_42`
5. `qwen25_dataset_ood_seed_43`
6. `qwen25_dataset_ood_seed_44`

## One-time Hugging Face access

The three Open LLM Leaderboard detail repositories are auto-gated. Sign into
Hugging Face, accept access on all three dataset pages, create a read token, and
add it to Colab secrets as `HF_TOKEN`:

- `open-llm-leaderboard/Qwen__Qwen2.5-1.5B-Instruct-details`;
- `open-llm-leaderboard/Qwen__Qwen2.5-3B-Instruct-details`; and
- `open-llm-leaderboard/Qwen__Qwen2.5-7B-Instruct-details`.

In Colab, open the **Secrets** panel with the key icon in the left sidebar,
choose **Add new secret**, set the name to exactly `HF_TOKEN`, paste the token as
the value, and turn on **Notebook access**. The name is case-sensitive. For
example, `hf_token`, `HF-TOKEN`, and a disabled `HF_TOKEN` are all unavailable
to `userdata.get("HF_TOKEN")`. After adding it, rerun the configuration cell;
there is no need to run or download any Qwen weights.

If you do not create a Colab secret, the same cell now displays a hidden
session-only prompt. Paste the read token there and press Enter. For example,
with zero configured secrets the cell asks once; after one nonempty token is
entered, all three evidence repositories reuse that in-memory value. The token
is not echoed, written to notebook output, or placed in the exported report.

The token downloads JSONL and evaluation metadata. The notebook downloads one
Qwen tokenizer only to count tokens; tokenization is not model inference.

### If the second cell returns `401 GatedRepoError`

Authentication and gated-dataset approval are separate. First, the notebook
calls `whoami()` to confirm that the token itself is valid. It then attempts a
pinned file download. If that download returns 401, open the reported dataset
URL while signed into the same Hugging Face account, accept its access
conditions, and confirm the token has **Read access to gated repositories**.
Repeat this for all three Qwen detail datasets and rerun the cell.

For example, one valid read token plus approvals for only two of three datasets
still produces an incomplete panel and must stop. One valid read token plus all
three approvals permits the notebook to download the three aligned result sets.

## Correctness and quality audit

The detail files already contain task-aware per-example grading. The notebook
does not regenerate an answer and does not replace those graders with a generic
string comparison. It requires each selected metric to be exactly 0 (incorrect)
or 1 (correct), stores the same result as `is_correct`, and calculates:

$$
\text{quality}=\frac{\text{correct published outcomes}}
{\text{all published outcomes}}.
$$

For example, 255 correct outcomes among 300 records produce 85% quality. The
run stops before ModernBERT training if a score is fractional, metrics conflict,
a prompt/model pair is duplicated, prompts or metrics disagree across models,
or any prompt lacks one of the three Qwen outcomes. Inspect and retain
`qwen_quality_audit.csv` in the downloaded ZIP.

## Frozen evidence contract

The three pinned repositories contain the same 39 task files. V3 removes GPQA
main and extended because they overlap with GPQA Diamond. It keeps at most 300
aligned prompts from each of the remaining 37 tasks. The maximum panel is:

$$
37\times300=11{,}100\text{ prompts},\qquad
11{,}100\times3=33{,}300\text{ published outcomes}.
$$

Tasks with fewer than 300 rows contribute every row. Evidence repository commits,
evaluation run timestamps, exclusions, cap, and sampling seed are hashed into
`EVIDENCE_TAG`. Every retained key must have the same document hash and rendered
prompt for all three candidates.

## Frozen router and timing contracts

Every run compares rank-4 hybrid, rank-4 safety-only, and rank-8 hybrid setups.
It keeps the same calibration, gate values, threshold grid, eight maximum epochs,
minimum epoch 2, patience 2, and requirement for two adjacent feasible
thresholds. Only `RUN_ID` changes the split mode and seed.

Candidate latency remains analytical. The scenario uses BF16 because the
published quality evidence does not establish that 4-bit quantization preserves
every answer. Published candidate runtime is ignored. After validation freezes
the policy, the notebook measures only ModernBERT batch-one overhead and compares
p50/p95 with the policy's break-even value.

For example, if a policy breaks even at 80 ms and ModernBERT measures 43 ms p50
and 68 ms p95, both median and tail overhead fit the analytical opportunity. If
p95 is 95 ms, median economics pass but tail economics do not.

## Download checklist

Before closing Colab:

- confirm all retained keys have three published outcomes;
- confirm `score` and `is_correct` agree and inspect `qwen_quality_audit.csv`;
- confirm `qwen_weights_loaded=False` in the evidence log;
- inspect the published metric used by every task;
- confirm the ZIP name matches `RUN_ID`;
- inspect every explicit validation and sealed-test failure reason;
- compare ModernBERT p50 and p95 with break-even;
- download the ZIP; and
- keep random and dataset-OOD artifacts separate.

A fallback-only result shows that the guard worked. It does not show that learned
prompt-level routing works.
