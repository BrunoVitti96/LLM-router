# LLM Router audit history

This file records completed experiments, changes between policy contracts, and
the evidence that motivated those changes. The current project behavior belongs
in [`README.md`](README.md); this file is historical and should not be read as
the active specification.

In plain language, the README answers "How does the router work now?" This audit
answers "What did earlier runs teach us, and why did the design change?"

## 2026-09-07 — Fix V6 mixed-dtype post-training inference

In plain language, notebook 06 finished training but could not score prompts
because its encoder and classifier used different numeric types. The saved
execution selected epoch 3 (validation safety loss 0.2461), then failed with
`mat1 and mat2 must have the same dtype, but got Half and Float`. It did not
produce a calibrated policy or completed test evaluation.

Technically, training and validation used autocast, but the post-training
inference loop did not. `QwenLastTokenRouter.forward` now casts the final-token
representation to each head's weight dtype before projection. Thus
`s_m = w_m^T cast(h_route, dtype(w_m)) + b_m` works with FP16/BF16 encoder
states and FP32 heads without requiring every caller to enable autocast.
For example, `[4, 4]` promoted from FP16 to FP32 still produces `8` for unit
weights and zero bias. Casting preserves autograd; it cannot recover precision
already lost inside the encoder. Model loading precision and policy contracts
remain as before.

Regression coverage checks FP16, BF16, and FP32 states, both heads, and last
non-padding pooling (lengths five and three yield logits 8 and 4). Notebook 06
and its builder document the fix and the need to sync companion source before
restarting Colab. Existing failed-run output is preserved; the notebook contract
test now accepts retained outputs as the V5 test already does. The README
distinguishes this incomplete run from a successful V6 result.

Validation: all 53 tests pass on the local CPU runtime, including the three
encoder-dtype regression cases; source compilation and `git diff --check` pass.
Repository-wide Ruff reports nine existing issues in unrelated files. A full
Qwen GPU training rerun has not been performed locally.

## 2026-09-04 — V6 Qwen2.5-1.5B final-token router ablation created

The completed V5 dataset-OOD seed-42 result reduced router truncation from
64.26% at 512 tokens to 11.74% at 1,024 tokens, but the selected prefix router's
validation ROC-AUC remained only 0.5690. Its test allocation was almost entirely
dataset-level: 0 of 1,000 BBH prompts, all 423 math prompts, and 178 of 250 MUSR
prompts were routed. Although test quality increased by 12 net answers and
conservative analytical savings were 34.31%, the 2.88% harm-rate upper bound
exceeded the 2.5% gate and the 81.33% guarded-dataset retention lower bound
missed the 90% gate. The single run therefore failed.

Notebook 06 was added to test a new representation hypothesis without changing
the evidence or policy objective. It loads `Qwen/Qwen2.5-1.5B-Instruct` as the
router, appends Qwen's existing one-token `\n<|endoftext|>` sentinel, and applies the existing two independent
safety heads to the final non-padding hidden state. It never calls generation
and does not train a vocabulary softmax to emit one model name. Technically, for
candidate $m$ it still estimates

$$
p_m=\sigma(w_m^\top h_{\mathrm{route}}+b_m)
$$

and the deterministic analytical selector chooses the fastest candidate above
the calibrated threshold. Class-balanced fallback-relative BCE, zero oracle
coefficient, rank-4/alpha-8 LoRA, five complete epochs, validation-best
checkpointing, the first 1,023 prompt tokens plus the preserved sentinel, split,
seed, calibration, gates, and
analytical candidate scenario remain fixed.

The 1.5B router is much larger than ModernBERT, so V6 uses gradient checkpointing
and micro-batch size one with four-step gradient accumulation. In numerical
terms, four micro-batches produce one effective batch-four optimizer update;
5,274 seed-42 training prompts therefore produce
$\lceil5{,}274/4\rceil=1{,}319$ updates per epoch. Shared training code gained an
optional router builder/text formatter and gradient accumulation, with defaults
that preserve every existing ModernBERT caller. The artifact manifest now
records pooling and prompt-wrapper identity, and the runtime loader reconstructs
either masked-mean ModernBERT or final-token Qwen.

V6 adds probability-span and within-dataset ROC-AUC diagnostics, a frozen V5
comparison table, standalone Qwen-router timing, and a reconstructable ZIP. The
4 ms and 20 ms overhead assumptions remain frozen only for controlled policy
comparison; measured Qwen p50/p95 are compared with break-even separately, and
no KV-cache reuse is assumed. Because V5 seed-42 outcomes already motivated this
change, V6 seed 42 is explicitly development evidence rather than a fresh sealed
claim. Untouched seeds or new task families are required for confirmation.
The notebook regression accepts V5's retained executed evidence but still
requires widget-free metadata whenever V5 is regenerated output-free; the new
V6 notebook is always checked as clean and output-free before distribution.

## 2026-09-01 — V5 export used the shared oracle coefficient

A notebook-05 execution reached the artifact cell and raised
`KeyError: 'oracle_auxiliary_weight'`. The export code still expected the v4
schema, where each setup declared its own oracle coefficient. V5 setup entries
describe only input representation: token budget and truncation strategy. The
coefficient is frozen once for all three variants in `V5_INPUT_CONTRACT`.

The export call now reads
`V5_INPUT_CONTRACT["oracle_auxiliary_weight"]`, which is `0.0`, and the unused
`selected_spec` variable was removed. In numerical terms, all three variants
export the same $1.0\mathcal L_{safety}+0.0\mathcal L_{oracle}$ contract; selecting
`prefix_512`, `prefix_1024`, or `head_tail_1024` changes the encoded tokens but
never the loss coefficient. A notebook regression check rejects the stale
per-representation lookup.

This was an export-schema bug, not a training, selection, or routing-policy
change. In a still-live Colab runtime, rerunning only the corrected export cell
is sufficient because `result`, `trainings`, and the selected checkpoint already
exist in memory.

## 2026-09-01 — V5 threaded ModernBERT compile failure fixed

The first notebook-05 Colab attempt downloaded and audited the evidence, passed
the 14 GiB/80%-free parallel preflight on a Tesla T4, and trained the three input
variants through most or all of epoch 5. It then stopped before validation-only
representation selection. The saved traceback ended in ModernBERT's
`compiled_mlp` with `Detected that you are using FX to symbolically trace a
dynamo-optimized function`.

In plain language, the experiment did not fail a quality check. Three training
workers collided inside an optional PyTorch compilation mechanism, so the
notebook never selected a representation and never opened the sealed test. No
v5 result is claimed from the partial run.

Technically, pinned Transformers 4.53.1 decorates ModernBERT's reference MLP
with `torch.compile(dynamic=True)` and enables it when Triton is available.
PyTorch Dynamo/FX compilation relies on shared state that is not safe for this
multi-threaded workload. Model initialization had already been serialized, but
the first forward pass for a new sequence shape can compile later during
training or validation, outside that lock. This explains why the error appeared
late rather than at model construction.

All project ModernBERT constructors now pass `reference_compile=False`, and
hybrid artifacts record `encoder_reference_compile=false`. Training and both
artifact loaders therefore use eager execution consistently. A regression test
checks both trainable router builders, and notebook metadata plus Markdown
record the compatibility choice. The deterministic notebook builder removes the
failed execution state and regenerates a clean rerunnable notebook.

This is not a model or policy change. For a numerical example, the v5 plan still
trains three variants for five epochs, or $3\times5=15$ model-epochs, with the
same rank-4 LoRA and safety BCE. Only the optional compile optimization changes
from enabled to disabled. Wall-clock time may increase, but safety targets,
checkpoint selection, calibrated probabilities, thresholds, and analytical
candidate latencies are unchanged.

## 2026-08-31 — V5 input-representation isolation after the v4 OOD failure

The executed notebook-04 dataset-OOD seed-42 run reduced training loss from
0.2220 to 0.2023, but validation loss stayed approximately flat at 0.2390,
0.2414, 0.2360, 0.2408, and 0.2411; epoch 3 was retained. Validation ROC-AUC
was 0.5467. In plain language, the router learned the training tasks without
learning a dependable safety ranking for unseen tasks.

The input audit provided a concrete first hypothesis: 64.26% of examples were
truncated at 512 tokens, all 588 routed sealed-test prompts were truncated, and
none of the 500 non-truncated prompts was routed. The 1.5B routes were +13 net;
the 3B routes gained 28 answers and lost 35, or -7 net. Macro-quality and harm
gates rejected the policy. This correlation does not establish causation, so a
new version isolates representation before changing the loss or dataset.

Notebook 05 compares `prefix_512`, `prefix_1024`, and `head_tail_1024` on the
same split and labels. The first is the exact v4 input control, the second tests
token budget, and the third preserves both prompt ends. For a 1,600-token input,
they discard 1,088 tokens, 576 tail tokens, and 576 middle tokens respectively.
All variants retain rank-4/alpha-8 LoRA, the class-balanced safety BCE, zero
oracle coefficient, five complete epochs, validation-best checkpointing,
calibration, thresholds, and safety gates. Validation selects one variant and
only that variant opens the sealed test. Notebook 05 is output-free and no v5
result is claimed by this change.

The three variants request concurrent execution because the observed v4 run
left GPU-memory headroom. Each uses batch size 4. A preflight requires at least
14 GiB total and 80% free GPU memory; otherwise execution is sequential. Model
construction is protected by a shared lock to reduce initialization spikes, and
every optimizer step is printed with its variant name so three worker logs remain
attributable. This check cannot guarantee speed or prevent all OOMs because
the workers still share one GPU's compute and memory bandwidth; the documented
recovery is a fresh runtime with parallel training disabled.

Shared source now makes router tokenization an artifact-level contract.
`RouterConfig` accepts a bounded token budget and `prefix` or `head_tail` strategy;
training, calibration, overhead measurement, exported inference, and the Gradio
runtime all call the same encoder. Old artifacts default to `prefix`. Training
also accepts an optional initialization lock for safe notebook concurrency.
Unit tests cover exact head-tail token selection, configuration/export defaults,
notebook cleanliness and contracts, and backward-compatible runtime behavior.
The README, Colab runbook, investor memo, short brief, and funded-validation plan
now report the executed v4 failure and the unrun v5 hypothesis separately.

## 2026-08-27 — V4 reduced to five epochs with per-step loss logging

Before any v4 result was produced, the notebook training schedule changed from
15 to five complete epochs. In plain language, Colab now gives feedback after
every training mini-batch instead of appearing silent until an epoch finishes.
It still compares validation after each epoch and keeps the best checkpoint.

Technically, notebook 04 now freezes `EPOCHS=5`, `MINIMUM_EPOCHS=5`, and
`EARLY_STOPPING_PATIENCE=None`. The shared trainer gained an optional,
backward-compatible `step_progress_callback` invoked once after every optimizer
step. Its payload includes epoch, step within epoch, global step, batch size,
current total loss, current safety loss, inactive oracle diagnostic, running
training-loss average, and mixed-precision skip status. Existing callers that do
not provide the callback retain their previous behavior.

For example, with 517 training prompts and batch size 8, an epoch has
$\lceil517/8\rceil=65$ optimizer steps and the five-epoch run has 325 printed
step records. A line such as `step=12/65 loss=0.384210 safety=0.384210
running=0.417832` reports the current mini-batch and running average. Because
the oracle coefficient remains zero, active total and safety loss are equal.
The validation loss is not replaced by these noisy step values: if epoch 3 has
validation loss 0.181 and epoch 5 has 0.196, epoch 3 is restored.

The README, v4 notebook Markdown and metadata, Colab runbook, investor
documents, funded-validation plan, and notebook contract tests now describe the
five-epoch schedule and per-step output. A trainer regression test verifies one
callback event per mini-batch. The loss, LoRA rank, calibration, thresholds,
OOD split, sealed-test gates, charts, exports, and final interactive showcase
are unchanged. V4 remains unexecuted, so this change creates no new result.

## 2026-08-27 — V4 safety-only 15-epoch OOD notebook

Notebook 04 introduces a versioned experiment contract rather than altering the
executed notebook-03 evidence. In plain language, the router now practices only
the yes/no safety decision it uses in production, completes all 15 training
epochs, and keeps the epoch that performed best on validation. The old oracle
coach remains visible for compatibility but has no vote in learning.

Technically, the single enabled setup is `safety_only_r4`, with LoRA rank 4,
`oracle_auxiliary_weight=0.0`, `EPOCHS=15`, `MINIMUM_EPOCHS=15`, and
`EARLY_STOPPING_PATIENCE=None`. The checkpoint rule remains minimum validation
loss. The `oracle_choices` labels, oracle head, `model_oracle_loss`, and exported
diagnostic fields remain present, but multiplication by zero blocks their
gradient contribution. For example, safety loss 0.223 and oracle diagnostic
0.477 yield $0.223+0\times0.477=0.223$.

The new run IDs are `qwen25_v4_{random,dataset_ood}_seed_{42,43,44}`; the default
is `qwen25_v4_dataset_ood_seed_42`. Versioning prevents later results from being
presented as unchanged continuations of the v3 three-setup, eight-epoch contract.
The notebook is output-free and no v4 result is claimed in this change.

For investor review, notebook 04 adds a four-panel OOD dashboard: training and
validation safety loss with the retained epoch; validation routed fraction and
conservative savings across thresholds; per-held-out-dataset quality retention
versus routed fraction; and model allocation with harmful routes. It exports the
dashboard PNG, the underlying dataset CSV, and `v4_training_contract.json`.
The final notebook cell launches the interactive Gradio prompt router so a user
can type a prompt and inspect the selected model, probabilities, analytical
latency, and fallback behavior.

Documentation now recommends notebook 04, explains the zero-coefficient oracle
precisely, provides a v4 Colab order, and shortens the investor memo into a
didactic presentation. `scripts/build_v4_notebook.py` deterministically builds
the notebook from the v3 workflow while clearing execution state, and notebook
contract tests protect the 15-epoch schedule, inactive oracle, OOD exports, and
final interactive cell. Routing thresholds, analytical candidate facts,
evidence revisions, calibration, and sealed-test gates are otherwise unchanged.

## 2026-08-27 — Notebook-03 seed-42 result and investor documentation

The README and investor documents now report the completed Qwen-tier random
seed-42 run instead of treating notebook 02 as the latest evidence. A new
[`docs/INVESTOR_READINESS_MEMO.md`](docs/INVESTOR_READINESS_MEMO.md) separates a
fundable validation story from an unsupported production-savings claim. The
shorter investor brief was updated to match it.

In plain language, the router found a promising aggregate result but did not
pass every safety check. It routed 636 of 1,726 test prompts to smaller tiers,
gained 40 answers, lost 32, and ended eight answers ahead of fallback. Modeled
savings remained 28.24% at the conservative 20 ms overhead. The run still
failed because performance was uneven across datasets.

Technically, validation selected `safety_only_r4` at threshold 0.870 with a
feasible block of 19 thresholds. The sealed-test quality-retention point
estimate was 101.08% and its one-sided 95% lower bound was 99.19%. The harm UCL
was 2.468%, routed-safety-precision LCB was 93.34%, and the macro-dataset
retention LCB missed its 98% gate. `bbh_object_counting` supplied +10 net answers
while the whole test supplied +8, demonstrating the task-mix problem.

The documentation now also records two limitations exposed by the run. First,
64.26% of router inputs were truncated at 512 tokens; truncated test prompts
were -2 net answers while non-truncated prompts were +10. Second, the 100-prompt
guarded-dataset rule matched zero test datasets and therefore returned its
neutral value. The macro gate still prevented activation, but a new versioned
study needs an achievable non-vacuous subgroup contract.

The loss explanation now distinguishes setting
`oracle_auxiliary_weight=0` from deleting the oracle head. Seed-42 validation
favored safety-only training, but removing the oracle variants before the
remaining seeds would change the frozen comparison after seeing a sealed test.
The documented options are to finish the existing ablation plan or declare seed
42 exploratory and start a new versioned safety-only study.

The README also warns that raw total loss is not comparable across safety-only
and hybrid objectives. The approximately 0.208 safety-only validation loss omits
the oracle term included in the approximately 0.603 hybrid loss; setup selection
therefore relies on validation policy metrics rather than the smaller raw number.
It now describes the executed notebook outputs as an inspectable working copy
and the reconstructable ZIP as the authoritative run record.

No Python, notebook source, routing behavior, loss coefficient, threshold,
candidate profile, evidence record, or evaluation gate changed in this
documentation-only update.

## 2026-08-26 — Unsuffixed IFEval strict metric accepted explicitly

The first authenticated notebook-03 evidence download reached IFEval and then
stopped before alignment or training. Those published rows expose four binary
fields—loose and strict scores at instruction and prompt level—but use the key
`prompt_level_strict_acc` without the `,none` suffix recognized by the frozen
priority. Because the generic fallback requires all accuracy-like fields to
agree, a row with different loose and strict outcomes correctly failed as
ambiguous under the old implementation.

In plain language, the benchmark supplied several related report-card marks,
and the loader had not recognized the exact spelling of the one the experiment
had already chosen. The fix recognizes that spelling instead of averaging or
silently choosing a more favorable score.

Technically, `BINARY_METRIC_PRIORITY` now places the unsuffixed IFEval
`prompt_level_strict_acc` immediately after its suffixed alias and before every
generic fallback. A regression test supplies all four disagreeing IFEval fields
and requires the strict prompt-level value. For example, loose prompt accuracy
1 and strict prompt accuracy 0 now produce `is_correct=False`; the result is not
reclassified as correct by either loose or instruction-level accuracy. This
changes the hashed correctness policy and therefore the notebook evidence tag,
but it does not change the router loss, split, calibration, latency model, or
quality gates. The failed partial notebook output was cleared so the canonical
notebook can be run from the first cell with the corrected companion source.

## 2026-08-26 — Gated-repository 401 split into token and approval checks

The next Colab run supplied `HF_TOKEN` but the first pinned Qwen2.5-1.5B results
file returned `401 GatedRepoError`. Hugging Face authentication and dataset
approval are separate: a syntactically present token can still be invalid, can
lack gated-repository read scope, or can belong to an account that has not
accepted the dataset conditions.

In plain language, possessing a key does not mean the account has accepted the
door's terms. All three dataset doors must be approved for a complete panel.

Technically, the evidence cell now calls `HfApi.whoami()` before downloads. An
invalid token produces a token-specific recovery message. Every pinned file
download catches `GatedRepoError` and reports the exact access URL, the need to
accept conditions with the token's account, and the required gated-repository
read permission. Numerically, approvals for two of three repositories yield an
incomplete candidate panel and zero router training; approvals for three of
three allow up to 33,300 aligned outcomes to proceed to the existing audits.

## 2026-08-26 — Missing secret now falls back to hidden token entry

The actionable missing-secret error still interrupted Colab `Run all`. Notebook
03 now treats the Colab secret as the convenient path, not the only path. When
`userdata.get("HF_TOKEN")` raises `SecretNotFoundError`, `NotebookAccessError`,
or `TimeoutException`, the cell reports only the exception class and opens a
`getpass` prompt. Unexpected exceptions still surface normally.

In plain language, a user with no saved Colab secret can paste the Hugging Face
read token once without exposing it on screen. The token lives only in memory
for that runtime.

Technically, credential precedence is environment variable, enabled Colab
secret, then hidden interactive input. An empty interactive value still fails
closed because gated evidence cannot be downloaded anonymously. Numerically,
zero configured secrets plus one pasted nonempty token yields one in-memory
credential reused for three pinned evidence repositories; zero secrets plus an
empty prompt yields zero credentials and zero network requests. No token is
written to notebook output, evidence artifacts, or the report ZIP.

## 2026-08-26 — Missing Colab Hugging Face secret made actionable

After the branch fix succeeded, the next notebook-03 run imported the router
package from `poc` and then stopped in configuration cell 4 with
`SecretNotFoundError: Secret HF_TOKEN does not exist`. This is an access setup
failure, not a Qwen inference, evidence-quality, or ModernBERT training failure.
Zero dataset rows were downloaded and zero models were run.

In plain language, the gated answer files require a key, but the Colab notebook
did not have a key named `HF_TOKEN`. The notebook cannot and should not bypass
the dataset owner's access controls.

Technically, token discovery now first accepts an `HF_TOKEN` environment
variable for non-Colab execution, then requests the Colab secret. Colab secret
lookup exceptions are converted into one actionable `RuntimeError` explaining
the exact case-sensitive name, Secrets-panel key icon, and notebook-access
toggle. For example, a secret named `hf_token` or a disabled `HF_TOKEN` yields
zero usable credentials; an enabled secret named exactly `HF_TOKEN` yields one
credential passed to every pinned Hugging Face dataset request. The README,
runbook, notebook Markdown, and notebook contract test now preserve those steps.

## 2026-08-26 — Colab branch mismatch fixed before evidence loading

An executed notebook-03 run stopped in configuration cell 4 with
`ModuleNotFoundError: No module named 'llm_router.qwen_evidence'`. The setup cell
had cloned and installed `develop`, while the executed notebook came from `poc`,
where the new correctness-audit module exists. No Qwen evidence was downloaded,
no Qwen candidate was run, and ModernBERT training never started.

In plain language, the notebook and its supporting code came from different
versions. This is like using chapter 3 instructions with a chapter 2 toolkit:
the requested tool is absent before the experiment begins.

Technically, notebook 03 now freezes `REPOSITORY_BRANCH = "poc"`, fetches and
switches an existing Colab checkout to that branch, pulls it with `--ff-only`,
installs the selected checkout, and preflights
`src/llm_router/qwen_evidence.py`. The README Colab URL now opens the same
branch. For a numerical execution example, the failed run completed 2 setup/
configuration cells and 0 evidence, training, validation, or test cells; after
the fix, a missing companion file stops in setup with an actionable version-
mismatch message.

## 2026-08-25 — Fail-closed published-correctness audit for Qwen v3

The canonical notebook already downloaded recorded Open LLM Leaderboard answers
and per-example metrics rather than running the three Qwen candidates. This
update made that quality contract executable and inspectable instead of relying
on a permissive `0 <= score <= 1` assertion.

In plain language, every downloaded answer now has an explicit yes/no
`is_correct` field. The notebook refuses to train the router if an outcome is
not clearly correct or incorrect, if a question is missing one of the small,
middle, or large-tier results, or if aligned models disagree about the prompt or
grader. It exports `qwen_quality_audit.csv`, so a Colab user can directly inspect
the arithmetic. For example, 240 correct rows and 60 incorrect rows give
$240/(240+60)=80\%$ quality.

Technically, `llm_router.qwen_evidence` now freezes the ordered panel as
Qwen2.5-1.5B (1.54B), Qwen2.5-3B (3.09B), and Qwen2.5-7B (7.61B, the roughly
8B-class tier). Metric extraction prefers `prompt_level_strict_acc,none`,
`exact_match,none`, `acc_norm,none`, and `acc,none`; conservative fallback
metrics must be binary and mutually agree. The aligned-panel audit requires
exact three-model coverage, unique key/model pairs, nonempty prompts, consistent
task/document/prompt/metric identities, finite binary scores, and the identity
`score == is_correct.astype(float)`. Unit and notebook-contract tests cover the
new failure modes. The correctness policy is now included in the hashed evidence
contract, so this stricter loader produces a new `EVIDENCE_TAG` even when the
pinned dataset revisions and sampling seed stay unchanged.

No Qwen generation was added. The only trainable/inference model in notebook 03
remains the ModernBERT router. Candidate latency remains analytical, and the
routing loss, calibration method, threshold gates, and sealed-test policy did
not change.

## 2026-08-21 — Seed-44 failure and three-tier Qwen v3

### What the completed seed-44 run showed

The executed notebook-02 random seed-44 run trained all three schema-v5 setups.
Validation selected rank-4 hybrid training at threshold `0.895`; it had a
contiguous feasible block of two thresholds. The sealed test contained 2,805
prompts. The router sent 623 prompts (22.21%) to Fin-R1, gained 43 correct
answers, lost 52, and ended nine answers behind Qwen3-8B.

The point-estimate quality retention was 99.55% and its one-sided 95% lower bound
was 98.73%. The harm-rate upper bound was 2.32%, within the 2.5% gate. However,
routed-safety precision was 91.65% with an 89.65% lower bound, below the 90%
gate. The guarded worst-dataset retention lower bound was 88.36%, also below its
90% floor, and macro-dataset retention missed its gate. Therefore the explicit
result was `single_run_passed=False`.

Task mix was materially unsafe. MBPP contributed +17 net answers and HumanEval
+7, while FinQA contributed -9, MMLU-Pro -7, WinoGrande -5, KORBench -5, and
ARC-Challenge -4. Numerically, $43-52=-9$. The positive code gains did not cancel
the knowledge and reasoning losses under the predeclared subgroup policy.

Analytical latency savings were 2.52% at 4 ms overhead and 1.68% at 20 ms. The
policy break-even overhead was 52.14 ms. Measured end-to-end ModernBERT overhead
was 42.55 ms at p50 and 67.23 ms at p95, so median economics were positive while
tail economics were negative.

### Why notebook v3 changed the candidate evidence

The historical panel compared 7.0B and 8.2B models, a parameter ratio of
$7.0/8.2=85.4\%$. That narrow separation limited savings after router overhead
and exposed only one replacement-safety head. LLMRouterBench's public lightweight
pool contains roughly 7B–9B candidates, so it cannot supply honest 1B/3B/8B
quality outcomes.

Notebook 03 now loads pinned per-example detail datasets for official
Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B evaluations. Their official sizes are
1.54B, 3.09B, and 7.61B. The three Open LLM Leaderboard repositories expose the
same 39 task files. V3 excludes overlapping GPQA main and extended variants and
keeps at most 300 aligned prompts from each of the remaining 37 tasks: at most
$11{,}100\times3=33{,}300$ published outcomes. Dataset revisions, evaluation run
IDs, exclusions, cap, and sampling seed are hashed into an evidence tag.

The detail repositories are auto-gated and require an accepted Hugging Face read
token. No Qwen weights are loaded and no candidate answers are generated. A
pinned Qwen tokenizer is downloaded only for token counts. Candidate latency
remains analytical under a BF16 scenario; using 4-bit latency would require
matching per-example quantized quality evidence. The temporary
`qwen-evaluation` dependency extra was removed because `bitsandbytes` and
`datasets` are no longer needed for candidate generation; the normal notebook
dependencies already include Hugging Face Hub and Transformers.

The v3 router compares rank-4 hybrid, rank-4 safety-only, and rank-8 hybrid
setups on validation. It keeps the schema-v5 calibration, threshold stability,
aggregate, macro, harm, routed-precision, guarded-dataset, and overhead gates.
It exports the scored candidate panel and evidence contract with every run.

The capped published panel must not be presented as a passed router until the
notebook is executed and `single_run_passed=True`; even then,
all random seeds, dataset-OOD runs, and a larger evidence panel remain required
for an investment-grade claim.

## 2026-08-19 — Multi-seed Colab plan, router-only timing, and demo

The seed-42 schema-v5 run was judged sufficient for a technical feasibility POC
but insufficient for production or universal-generalization claims. The next
repository version therefore converts the notebook from a mutable seed/split
cell into six immutable run IDs: random and dataset-OOD modes at seeds 42, 43,
and 44. Every run imports the same three training setups and exports a separate
ZIP. This prevents the loss, gate contract, or setup menu from drifting between
confirmation runs.

The analytical scenario identity date is frozen at `2026-08-19`, matching the
completed seed-42 artifact. It no longer changes according to the day a later
Colab confirmation happens to run.

The notebook now measures only ModernBERT batch-one overhead on deterministic
validation prompts. It exports model-only and end-to-end p50/p95 distributions;
candidate LLM latency remains analytical and no candidate is loaded or timed.
The measurement is diagnostic and is collected after validation policy freeze,
so it cannot retroactively alter the 4 ms nominal or 20 ms conservative gate.
For a numerical example, a measured p95 of 30 ms would exceed the seed-42 policy
break-even value of 26.93 ms even if a 14 ms median remained below it. Both
numbers must be disclosed.

An analytical-latency inference runtime and Colab Gradio demo were added. The
demo shows prompt, safety probability, selected model, fallback use, candidate
latency estimates, measured ModernBERT overhead, and estimated net savings. It
does not generate an LLM answer. Investor documentation now defines the initial
commercial customer, current seed-42 evidence and limitations, and funded
validation milestones with explicit narrow-or-stop outcomes.

No safety label, hybrid-loss equation, candidate profile, quality gate,
threshold-selection rule, or analytical latency equation changed. The canonical
notebook's saved seed-42 outputs were cleared because its code changed; the
completed run remains preserved under
`results/modernbert_hybrid_random_seed_42_v3/`.

The generated `results/` tree is now excluded from Ruff because it contains
immutable run artifacts rather than maintained source.

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

## Schema-v4 seed-42 rerun and validation-comparison redesign

### What the latest rerun showed

The schema-v4 rerun evaluated 2,812 test prompts and routed 315 (11.20%) to
Fin-R1. Of those routes, 298 were safe and 17 harmful, so routed-safety precision
was 94.60% with a one-sided 95% lower bound of 92.10%. The router gained 24
answers and lost 17, finishing seven answers ahead of the Qwen fallback.

Quality retention was 100.35% with a 99.82% lower confidence bound. Nominal
analytical savings were 1.05% after 4 ms overhead, but only 0.20% after 20 ms.
The break-even overhead was 23.83 ms. The router captured 12.82% of oracle
savings and recalled only 13.82% of safe faster opportunities.

The net result was task-mix sensitive. MBPP contributed 14 net answers, while
ARC-Challenge contributed -6 and FinQA -3. Removing MBPP leaves the other
datasets seven answers behind fallback:

$$
(24-17)-14=-7.
$$

Calibration improved validation Brier score from 0.1787 to 0.1510 and ECE from
13.35% to 2.35%, but AUROC was only 0.739. The router was therefore calibrated
yet only moderately discriminative, which explains high precision and low
safe-opportunity recall.

### Why schema v5 was introduced

The chosen validation threshold was `0.910`. Threshold `0.905` missed the
routed-precision gate, while `0.915` produced negative savings at 20 ms overhead.
That made activation depend on one isolated grid point.

Schema v5 made the following predeclared changes:

1. compare rank-4 hybrid, rank-4 safety-only, and dataset-balanced rank-4
   hybrid setups using validation outcomes only, with rank 8 available as an
   optional capacity ablation;
2. choose one setup before opening sealed-test outcomes exactly once;
3. report a boolean result for every threshold gate plus the contiguous feasible
   block size;
4. require at least two neighboring feasible thresholds by default;
5. export setup leaderboards and complete setup threshold frontiers;
6. record benchmark fingerprint, Git commit, Python, packages, GPU, and CUDA;
   and
7. call the run result `single_run_passed`, retaining `poc_passed` only as a
   schema-v4 compatibility alias.

For a numerical example, one passing threshold has block size one and cannot
activate schema v5. Two adjacent passing thresholds have block size two and meet
the default stability rule. This deliberately trades some routing rate for a
policy less likely to flip after a 0.005 threshold perturbation.

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
