"""Build notebook 06: Qwen final-token router with versioned corrected scores.

Notebook 05 supplies candidate facts, analytical latency, split implementation,
and safety gates. The evidence loader now regrades MATH with the published
Math-Verify procedure and audits full-task source evidence before sampling.
Historical V5/V6 scores cannot serve as architecture comparison baselines.
The builder emits an output-free notebook under new scoring-v2 run IDs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "05_train_modernbert_input_representation_ood_v5.ipynb"
TARGET = ROOT / "notebooks" / "06_train_qwen15_last_token_router_ood_v6.ipynb"


def source_lines(text: str) -> list[str]:
    """Return nbformat-compatible source lines with stable trailing newlines."""

    normalized = text.strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

v5_contract_source = "".join(notebook["cells"][4]["source"])
candidate_start = v5_contract_source.index("CANDIDATES = {")
evidence_tag_start = v5_contract_source.index("EVIDENCE_TAG =")
candidate_and_auth = v5_contract_source[candidate_start:evidence_tag_start]

notebook["cells"][0]["source"] = source_lines(
    r"""
# V6 Qwen2.5-1.5B final-token router OOD experiment

This notebook reruns the **causal decoder** safety router after correcting the
MATH scoring contract. Pinned per-example MATH rows contain older scores that
can reject a correct boxed answer, while the published aggregates were updated
with Math-Verify. For example, a response ending in `\\boxed{968}` can have an
old score of 0 even though its target is 968. The corrected grader restores the
appropriate label before sampling or training. Counts and non-math source means
must reconcile. Math source means are reported separately because their upstream
rescoring environment is not pinned and some 7B totals do not reproduce exactly.
Original scores and raw evidence remain exported; no labels are invented to
match an aggregate.

A separate
`Qwen/Qwen2.5-1.5B-Instruct` instance reads the prompt, its existing one-token
`<|endoftext|>` sentinel is appended, and two independent safety logits are
computed from the sentinel's final hidden state.  The router never invokes text
generation, does not observe candidate answers, and does not directly emit a
model name.  The analytical selector still chooses the fastest candidate whose
calibrated safety probability clears the frozen threshold.

The deterministic split, seed, class-balanced BCE, calibration, threshold grid,
latency scenario, and quality gates remain fixed. Corrected labels change what
the router learns: if fallback quality changes from 0 to 1 while a smaller
candidate stays at 0, its safety label changes from 1 to 0.

The new `scoring_v2` run IDs preserve historical results. Earlier V5/V6 numbers
use a different scoring contract and are not a controlled architecture
comparison. Seed 42 remains development evidence because its prompts have been
inspected; use untouched task families for independent confirmation.
"""
)

notebook["cells"][1]["source"] = source_lines(
    """
## 1. Set up Colab

Choose a GPU runtime.  A Tesla T4 is the minimum practical target; an A100 is
strongly preferred because five epochs over 1,024-token prompts are expensive.
The notebook downloads Qwen2.5-1.5B once as the router.  It never generates new
candidate answers—the benchmark outcomes remain the pinned published records.

For an immediate run, upload `llm_router_colab_scoring_v2.zip` through Colab's
Files sidebar so it exists at `/content/llm_router_colab_scoring_v2.zip`.
Setup safely extracts this source bundle to `/content/LLM_Router` and skips Git.
Without that bundle, setup fetches the `poc` branch; both the corrected notebook
and companion Python source must have reached that branch. A scoring-version
check stops stale source before downloads or training. Restart the runtime and
run every cell from setup; old checkpoints contain the old labels.

The existing FP16/BF16 head-dtype fix is retained. For example, FP16 `[4, 4]`
is promoted to FP32 before a unit-weight head computes the same logit `8`.
"""
)

notebook["cells"][2]["source"] = source_lines(
    r'''
%cd /content

import hashlib
import importlib
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY_URL = "https://github.com/BrunoVitti96/LLM-router.git"
REPOSITORY_BRANCH = "poc"
PROJECT_ROOT = Path("/content/LLM_Router")
SOURCE_BUNDLE = Path("/content/llm_router_colab_scoring_v2.zip")
EXPECTED_SCORING_VERSION = "qwen-math-verify-v2"


def extract_source_bundle(bundle_path, project_root):
    """Extract a source ZIP only after validating all paths and file types."""
    root = project_root.resolve()
    with zipfile.ZipFile(bundle_path) as archive:
        members = archive.infolist()
        for member in members:
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            target = (root / member.filename).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or ":" in member.filename
                or not target.is_relative_to(root)
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"Unsafe source-bundle path: {member.filename!r}")
        root.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)


if SOURCE_BUNDLE.is_file():
    extract_source_bundle(SOURCE_BUNDLE, PROJECT_ROOT)
    SOURCE_BUNDLE_SHA256 = hashlib.sha256(SOURCE_BUNDLE.read_bytes()).hexdigest()
    print(f"Using corrected local source bundle: {SOURCE_BUNDLE}")
else:
    SOURCE_BUNDLE_SHA256 = None
    if not PROJECT_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--branch", REPOSITORY_BRANCH,
             REPOSITORY_URL, str(PROJECT_ROOT)],
            check=True,
        )
    for arguments in (
        ["fetch", "origin", REPOSITORY_BRANCH],
        ["switch", REPOSITORY_BRANCH],
        ["pull", "--ff-only", "origin", REPOSITORY_BRANCH],
    ):
        subprocess.run(["git", "-C", str(PROJECT_ROOT), *arguments], check=True)

REQUIRED_EVIDENCE_HELPER = PROJECT_ROOT / "src/llm_router/qwen_evidence.py"
REQUIRED_QWEN_ROUTER = PROJECT_ROOT / "src/llm_router/models/qwen_last_token_router.py"
if not REQUIRED_EVIDENCE_HELPER.is_file() or not REQUIRED_QWEN_ROUTER.is_file():
    raise RuntimeError(
        "Repository/source mismatch: notebook 06 requires the corrected evidence "
        "helper and Qwen last-token router. Upload the matching source ZIP."
    )
os.chdir(PROJECT_ROOT)
%pip install -q -U ".[notebook]"

SOURCE_ROOT = str(PROJECT_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)
for module_name in list(sys.modules):
    if module_name == "llm_router" or module_name.startswith("llm_router."):
        del sys.modules[module_name]
importlib.invalidate_caches()
llm_router = importlib.import_module("llm_router")
evidence_helper = importlib.import_module("llm_router.qwen_evidence")
if getattr(evidence_helper, "SCORING_CONTRACT_VERSION", None) != EXPECTED_SCORING_VERSION:
    raise RuntimeError(
        "Repository/source mismatch: this notebook requires qwen-math-verify-v2. "
        "Upload llm_router_colab_scoring_v2.zip to /content, restart the runtime, "
        "and rerun setup, or sync the matching source to the poc branch."
    )
math_helper = importlib.import_module("llm_router.math_scoring")
math_helper.validate_math_scoring_runtime()
print(
    f"Router package ready from {Path(llm_router.__file__).resolve()} "
    f"with scoring contract {evidence_helper.SCORING_CONTRACT_VERSION}"
)
'''
)

notebook["cells"][3]["source"] = source_lines(
    r"""
## 2. Freeze the v6 causal-router contract

The scoring contract is now `qwen-math-verify-v2`; the original policy, split,
and router architecture are retained. Rank-4 LoRA is trained for
all five epochs, and the minimum OOD-validation safety loss checkpoint is
restored.

Memory-safe micro-batches contain one prompt.  Four micro-batches are
accumulated before each optimizer update, preserving V5's effective batch size
of four.  For example, four prompts with individual losses `0.10`, `0.20`,
`0.30`, and `0.40` contribute the averaged update loss
$(0.10+0.20+0.30+0.40)/4=0.25$.

The frozen 4 ms and conservative 20 ms overhead assumptions are retained solely
as frozen evaluation assumptions. Measured Qwen-router p50/p95 are reported separately;
candidate answer generation is still analytical-only for latency.
"""
)

contract_prefix = r'''
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from getpass import getpass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from IPython.display import display
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from llm_router.config import DEFAULT_CONFIG
from llm_router.experiment_comparison import (
    build_setup_comparison,
    combine_threshold_searches,
)
from llm_router.hybrid_inference import (
    HybridModernBERTRouterRuntime,
    create_gradio_demo,
)
from llm_router.models.qwen_last_token_router import (
    QWEN_LAST_TOKEN_POOLING,
    QWEN_ROUTER_TEXT_PREFIX,
    QWEN_ROUTER_TEXT_SUFFIX,
    build_qwen_last_token_router,
    format_qwen_router_text,
)
from llm_router.modernbert_poc import (
    export_modernbert_hybrid_poc,
    train_modernbert_hybrid_poc,
)
from llm_router.oracle import oracle_choices, replacement_safety_targets
from llm_router.public_benchmark import (
    EconomicsScenario,
    ModelProfile,
    export_public_benchmark,
    make_complete_panel,
    run_public_benchmark,
    select_validation_policy,
    simulate_economics,
    split_benchmark,
)
from llm_router.qwen_evidence import (
    BINARY_METRIC_PRIORITY,
    SCORING_CONTRACT_VERSION,
    audit_aligned_outcomes,
    reconcile_published_results,
    score_published_sample,
    scoring_contract,
    validate_qwen_tier_contract,
)
from llm_router.router_overhead import benchmark_modernbert_overhead
from llm_router.utils.training import seed_everything

RUN_SPECS = {
    "qwen25_v6_scoring_v2_ood_seed_42": ("dataset_ood", 42),
    "qwen25_v6_scoring_v2_ood_seed_43": ("dataset_ood", 43),
    "qwen25_v6_scoring_v2_ood_seed_44": ("dataset_ood", 44),
    "qwen25_v6_scoring_v2_random_seed_42": ("random", 42),
    "qwen25_v6_scoring_v2_random_seed_43": ("random", 43),
    "qwen25_v6_scoring_v2_random_seed_44": ("random", 44),
}
RUN_ID = "qwen25_v6_scoring_v2_ood_seed_42"
SPLIT_MODE, SEED = RUN_SPECS[RUN_ID]

MAX_PROMPTS_PER_TASK = 300
EVIDENCE_SAMPLE_SEED = 20260821
EPOCHS = 5
MINIMUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = None
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
STEP_LOG_EVERY = 25
MAX_INPUT_TOKENS = 1024
MEASURE_ROUTER_OVERHEAD = True
LAUNCH_INTERACTIVE_DEMO = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Choose Runtime > Change runtime type > GPU."
'''

contract_suffix = r'''
evidence_contract["correctness_policy"] = scoring_contract()
QWEN_ROUTER_REPO = CANDIDATES["Qwen2.5-1.5B"]["model_repo"]
QWEN_ROUTER_REVISION = CANDIDATES["Qwen2.5-1.5B"]["model_revision"]
SETUP_NAME = "qwen15_last_token_1024"
V6_QWEN_ROUTER_CONTRACT = {
    "version": "v6-qwen15-last-token-scoring-v2",
    "scoring_version": SCORING_CONTRACT_VERSION,
    "historical_architecture_comparison_valid": False,
    "development_comparison": True,
    "router_repo": QWEN_ROUTER_REPO,
    "router_revision": QWEN_ROUTER_REVISION,
    "pooling_strategy": QWEN_LAST_TOKEN_POOLING,
    "text_prefix": QWEN_ROUTER_TEXT_PREFIX,
    "text_suffix": QWEN_ROUTER_TEXT_SUFFIX,
    "max_input_tokens": MAX_INPUT_TOKENS,
    "input_truncation_strategy": "prefix_with_last",
    "loss": "class-balanced fallback-relative safety BCE",
    "oracle_auxiliary_weight": 0.0,
    "lora_rank": 4,
    "lora_alpha": 8,
    "epochs": EPOCHS,
    "checkpoint_rule": "minimum validation safety loss",
    "micro_batch_size": MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": (
        MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    ),
    "candidate_generation_run": False,
    "kv_cache_reuse_assumed": False,
}
assert EPOCHS == MINIMUM_EPOCHS == 5
assert EARLY_STOPPING_PATIENCE is None
assert V6_QWEN_ROUTER_CONTRACT["effective_batch_size"] == 4

EVIDENCE_TAG = hashlib.sha256(
    json.dumps(evidence_contract, sort_keys=True).encode("utf-8")
).hexdigest()[:16]
RUN_CONTRACT_TAG = hashlib.sha256(
    json.dumps(
        {
            "run_id": RUN_ID,
            "evidence_tag": EVIDENCE_TAG,
            "training": V6_QWEN_ROUTER_CONTRACT,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()[:16]
OUTPUT_DIR = PROJECT_ROOT / "reports_benchmark" / RUN_ID


def log_stage(stage, **values):
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    details = " | ".join(f"{key}={value}" for key, value in values.items())
    print(f"[{timestamp}] {stage}" + (f" | {details}" if details else ""))


seed_everything(SEED)
log_stage(
    "contracts frozen",
    run_id=RUN_ID,
    split=SPLIT_MODE,
    evidence_tag=EVIDENCE_TAG,
    run_contract_tag=RUN_CONTRACT_TAG,
    router="Qwen2.5-1.5B final-token classifier",
    effective_batch=V6_QWEN_ROUTER_CONTRACT["effective_batch_size"],
    epochs=EPOCHS,
    gpu=torch.cuda.get_device_name(0),
)
display(pd.DataFrame(CANDIDATES).T)
display(pd.Series(V6_QWEN_ROUTER_CONTRACT, name="V6 router contract"))
'''
notebook["cells"][4]["source"] = source_lines(
    contract_prefix + "\n" + candidate_and_auth + "\n" + contract_suffix
)

notebook["cells"][5]["source"] = source_lines(
    r'''
## 3. Download, correct, and reconcile published Qwen evidence

Accept access to each of these auto-gated Hugging Face datasets:

- `open-llm-leaderboard/Qwen__Qwen2.5-1.5B-Instruct-details`
- `open-llm-leaderboard/Qwen__Qwen2.5-3B-Instruct-details`
- `open-llm-leaderboard/Qwen__Qwen2.5-7B-Instruct-details`

The same pinned evaluation runs and 37 non-overlapping tasks remain selected.
This cell downloads JSONL evidence, not model weights. MATH rows are regraded
from the full response and the document's worked solution using the published
Math-Verify procedure. Other tasks retain their published binary metric,
including strict prompt-level IFEval accuracy. For example, a strict IFEval
score of 0 remains 0 even if loose accuracy is 1.

Original metrics and scores, corrected scores, grader version, raw document,
and response payloads are kept. Before alignment or the 300-prompt task cap,
every task/model sample count and non-math mean must match the source metadata.
Math labels use the pinned local scorer; the upstream math rescoring environment
is not versioned, and some 7B aggregate totals do not reproduce. Those differences
are displayed and exported as diagnostics, never hidden or used to invent labels.
For example, local counting quality may be 55/123 while the source reports 57/123.
The audit distinguishes source detail means, local means, and published leaf means.
'''
)

loader_source = "".join(notebook["cells"][6]["source"])
loader_source = loader_source.replace(
    "metric_name, score = published_binary_score(payload)",
    "scored = score_published_sample(payload, task)",
).replace(
    '"score": score,\n                        "published_metric": metric_name,',
    '**scored,\n'
    '                        "raw_doc_json": json.dumps(\n'
    '                            payload.get("doc"), ensure_ascii=False, sort_keys=True\n'
    '                        ),\n'
    '                        "raw_resps_json": json.dumps(\n'
    '                            payload.get("resps"), ensure_ascii=False\n'
    '                        ),\n'
    '                        "filtered_resps_json": json.dumps(\n'
    '                            payload.get("filtered_resps"), ensure_ascii=False\n'
    '                        ),',
).replace(
    'raw_records = pd.DataFrame(raw_rows)',
    'raw_records = pd.DataFrame(raw_rows)\n'
    'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n'
    'try:\n'
    '    reconciliation_audit = reconcile_published_results(\n'
    '        raw_records, published_run_metadata\n'
    '    )\n'
    'except ValueError as error:\n'
    '    if hasattr(error, "reconciliation_audit"):\n'
    '        error.reconciliation_audit.to_csv(\n'
    '            OUTPUT_DIR / "qwen_scoring_reconciliation.csv", index=False\n'
    '        )\n'
    '    raise\n'
    'reconciliation_audit.to_csv(\n'
    '    OUTPUT_DIR / "qwen_scoring_reconciliation.csv", index=False\n'
    ')\n'
    'raw_records.loc[raw_records.task.str.startswith("leaderboard_math_")].to_parquet(\n'
    '    OUTPUT_DIR / "qwen_math_rescored_records.parquet", index=False\n'
    ')\n'
    'log_stage("full-task count and nonmath checks passed; math locally regraded",\n'
    '          groups=len(reconciliation_audit))\n'
    'display(reconciliation_audit)\n'
    'math_differences = reconciliation_audit.loc[\n'
    '    ~reconciliation_audit.aggregate_required & ~reconciliation_audit.mean_matches\n'
    ']\n'
    'if not math_differences.empty:\n'
    '    print("Math source aggregate differences (local pinned scores are used):")\n'
    '    display(math_differences)',
)
notebook["cells"][6]["source"] = source_lines(loader_source)

notebook["cells"][7]["source"] = source_lines(
    r'''
## 4. Align, sample, and audit the corrected outcomes

Full-task counts and non-math reconciliation have passed; math has been regraded
under the pinned contract and its source differences reported. The loader aligns all
three candidate models, verifies matching document hashes and rendered prompts,
and keeps at most 300 prompts per task using the unchanged deterministic seed.
Duplicate document hashes are kept once globally to prevent repeated questions
crossing dataset-OOD partitions.

The corrected binary `score` becomes `is_correct`. Quality remains
`correct / outcomes`: 240 correct rows among 300 produce $240/300=80\%$.
The task cap changes the subset, so this sampled mean can differ from the
full-task mean above. The original published score and
the corrected score remain available side by side in exported records.

The pinned Qwen tokenizer counts input/output tokens without model inference.
Realized completion length remains excluded from analytical latency.
'''
)

notebook["cells"][13]["source"] = source_lines(
    r"""
## 7. Understand the causal last-token experiment

For candidate $m$, the Qwen router receives prompt tokens followed by the fixed
sentinel.  Under causal attention, the sentinel state $h_T$ can attend to every
earlier input token.  The deployed head is

$$s_m=w_m^\top h_T+b_m,\qquad p_m=\sigma(s_m).$$

The two candidates retain independent sigmoid outputs: both 1.5B and 3B may be
safe on one prompt.  This is deliberately different from training the language
model head to generate one mutually exclusive token such as `1.5B`.

The target remains

$$y_m(x)=\mathbf 1[Q_m(x)\ge Q_{7B}(x)],$$

and latency remains outside the neural model.  For example, probabilities 0.93
and 0.96 at a threshold of 0.90 make both alternatives eligible; the analytical
selector chooses 1.5B because it is faster.
"""
)

notebook["cells"][14]["source"] = source_lines(
    """
## 8. Train the Qwen2.5-1.5B final-token router

The router uses FP16 on a T4 and BF16 on Ampere-or-newer GPUs.  Gradient
checkpointing is enabled in the model builder.  If a T4 still runs out of
memory, restart the runtime before retrying; do not silently shorten the input
or change the dataset because that would change the frozen run contract.
"""
)

notebook["cells"][15]["source"] = source_lines(
    r'''
def report_step(row):
    step = int(row["step_in_epoch"])
    final_step = int(row["steps_per_epoch"])
    if step != 1 and step != final_step and step % STEP_LOG_EVERY:
        return
    print(
        f"[qwen15] epoch={int(row['epoch'])}/{int(row['epochs'])} "
        f"micro_batch={step}/{final_step} "
        f"optimizer_updates={int(row['completed_optimizer_steps'])} "
        f"loss={row['step_total_loss']:.6f} "
        f"running={row['running_train_total_loss']:.6f}",
        flush=True,
    )


def report_epoch(row):
    marker = "BEST" if row["is_best_epoch"] else "    "
    print(
        f"[qwen15] epoch={int(row['epoch'])}/{EPOCHS} {marker} "
        f"train={row['train_total_loss']:.4f} "
        f"validation={row['validation_total_loss']:.4f} "
        f"best_epoch={int(row['best_epoch_so_far'])} "
        f"seconds={row['epoch_seconds']:.1f}"
    )


router_config = replace(
    DEFAULT_CONFIG,
    seed=SEED,
    encoder_repo=QWEN_ROUTER_REPO,
    encoder_revision=QWEN_ROUTER_REVISION,
    lora_r=4,
    lora_alpha=8,
    max_input_tokens=MAX_INPUT_TOKENS,
    input_truncation_strategy="prefix_with_last",
)

torch.cuda.empty_cache()
gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info()
display(
    pd.Series(
        {
            "gpu": torch.cuda.get_device_name(0),
            "total_gib": gpu_total_bytes / 2**30,
            "free_gib_before_training": gpu_free_bytes / 2**30,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": (
                MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
        },
        name="Qwen router preflight",
    )
)

log_stage("Qwen router training started", setup=SETUP_NAME)
try:
    selected_training = train_modernbert_hybrid_poc(
        panel,
        split,
        config=router_config,
        epochs=EPOCHS,
        batch_size=MICRO_BATCH_SIZE,
        learning_rate=1e-4,
        head_learning_rate=2e-4,
        minimum_epochs=MINIMUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        quality_epsilon=0.0,
        safety_loss_weight=1.0,
        oracle_auxiliary_weight=0.0,
        dataset_balanced_sampling=False,
        device=DEVICE,
        progress_callback=report_epoch,
        step_progress_callback=report_step,
        router_builder=build_qwen_last_token_router,
        router_text_formatter=format_qwen_router_text,
        router_display_name="Qwen2.5-1.5B final-token router",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
except RuntimeError as error:
    if "out of memory" in str(error).lower():
        raise RuntimeError(
            "Qwen router training exhausted GPU memory. Restart the runtime "
            "and use an A100; changing the frozen token budget would create a "
            "different experiment."
        ) from error
    raise

assert selected_training.epochs_completed == EPOCHS
assert not selected_training.stopped_early
assert selected_training.gradient_accumulation_steps == 4
log_stage(
    "Qwen router training finished",
    best_epoch=selected_training.best_epoch,
    truncation=f"{selected_training.input_diagnostics['truncation_rate']:.2%}",
)
display(selected_training.history)
display(selected_training.calibration_diagnostics)
'''
)

notebook["cells"][16]["source"] = source_lines(
    """
## 9. Select the threshold using validation only

There is one architecture, so validation selects only its calibrated safety
threshold. The corrected evidence is fixed before splitting; neither test
outcomes nor historical run metrics participate in policy selection.
"""
)

notebook["cells"][17]["source"] = source_lines(
    r'''
policy_kwargs = {
    "objective": "latency",
    "minimum_quality_retention": DEFAULT_CONFIG.minimum_quality_retention,
    "confidence": DEFAULT_CONFIG.quality_confidence,
    "validation_quality_margin": DEFAULT_CONFIG.validation_quality_margin,
    "minimum_predicted_savings": DEFAULT_CONFIG.minimum_predicted_speedup,
    "router_overhead_s": scenario.router_overhead_s,
    "conservative_router_overhead_s": DEFAULT_CONFIG.conservative_router_overhead_s,
    "minimum_macro_quality_retention": DEFAULT_CONFIG.minimum_macro_quality_retention,
    "maximum_quality_loss_rate_ucl": DEFAULT_CONFIG.maximum_quality_loss_rate_ucl,
    "minimum_routed_safety_precision_lcb": DEFAULT_CONFIG.minimum_routed_safety_precision_lcb,
    "minimum_guarded_dataset_quality_retention_lcb": (
        DEFAULT_CONFIG.minimum_guarded_dataset_quality_retention_lcb
    ),
    "minimum_guarded_dataset_prompts": DEFAULT_CONFIG.minimum_guarded_dataset_prompts,
    "minimum_consecutive_feasible_thresholds": (
        DEFAULT_CONFIG.minimum_consecutive_feasible_thresholds
    ),
    "seed": SEED,
}
frozen_validation_policy = select_validation_policy(
    panel,
    split,
    routing_probabilities=selected_training.safety_probabilities,
    router_name="qwen15_last_token_router",
    **policy_kwargs,
)
trainings = {SETUP_NAME: selected_training}
selections = {SETUP_NAME: frozen_validation_policy}
setup_comparison = build_setup_comparison(trainings, selections)
setup_comparison["max_input_tokens"] = MAX_INPUT_TOKENS
setup_comparison["input_truncation_strategy"] = "prefix_with_last"
setup_comparison["pooling_strategy"] = QWEN_LAST_TOKEN_POOLING
setup_comparison["truncation_rate"] = (
    selected_training.input_diagnostics["truncation_rate"]
)
setup_comparison["selected_for_test"] = True
setup_threshold_search = combine_threshold_searches(selections)
display(setup_comparison)
display(frozen_validation_policy.threshold_search)
log_stage(
    "POLICY FROZEN — development test comparison may now open",
    active=frozen_validation_policy.router_active,
    threshold=frozen_validation_policy.selected_threshold,
    reason="; ".join(frozen_validation_policy.failure_reasons) or "all gates passed",
)
'''
)

notebook["cells"][18]["source"] = source_lines(
    """
## 10. Measure Qwen-router overhead only

This times tokenization, transfer, and one final-token classification forward
pass.  It does not generate an answer.  No KV-cache reuse is assumed when the
1.5B candidate is selected, making this a conservative standalone-router
measurement.
"""
)

notebook["cells"][19]["source"] = source_lines(
    r'''
router_overhead_benchmark = None
if MEASURE_ROUTER_OVERHEAD:
    validation_examples = panel.examples.iloc[split.validation]
    router_overhead_benchmark = benchmark_modernbert_overhead(
        selected_training.model,
        selected_training.tokenizer,
        validation_examples.prompt.to_numpy(),
        validation_examples.prompt_tokens.to_numpy(),
        device=DEVICE,
        max_input_tokens=router_config.max_input_tokens,
        input_truncation_strategy=router_config.input_truncation_strategy,
        timed_requests=min(100, len(validation_examples)),
        warmup_requests=10,
        seed=SEED,
        router_text_prefix=QWEN_ROUTER_TEXT_PREFIX,
        router_text_suffix=QWEN_ROUTER_TEXT_SUFFIX,
        router_display_name="Qwen2.5-1.5B final-token router",
    )
    measured = router_overhead_benchmark.summary["end_to_end_ms"]
    log_stage(
        "Qwen router overhead measured",
        p50_ms=f"{measured['p50']:.2f}",
        p95_ms=f"{measured['p95']:.2f}",
        candidate_generation="not run",
        policy_changed=False,
    )
    display(pd.DataFrame(router_overhead_benchmark.summary).loc[
        ["mean", "p50", "p95", "maximum"],
        ["end_to_end_ms", "model_only_ms"],
    ])
'''
)

notebook["cells"][20]["source"] = source_lines(
    """
## 11. Evaluate the frozen development test

The default seed-42 prompts were inspected in earlier experiments. This is a
development evaluation under corrected labels, not new independent confirmation
or a controlled comparison with earlier V5/V6 metrics. The selected run's
deterministic partition and every original policy gate remain unchanged.
"""
)

test_source = "".join(notebook["cells"][21]["source"])
test_source = test_source.replace(
    'router_name="modernbert_qwen_tier_router"',
    'router_name="qwen15_last_token_router"',
)
test_source = test_source.replace("selected_setup=SELECTED_SETUP", "selected_setup=SETUP_NAME")
test_source = test_source.replace("selected_config", "router_config")
test_source = test_source.replace(
    'result.summary.loc["modernbert_qwen_tier_router"]',
    'result.summary.loc["qwen15_last_token_router"]',
)
test_source = test_source.replace("sealed test complete", "development test comparison complete")
test_source = test_source.replace("measured ModernBERT p50", "measured Qwen-router p50")
test_source = test_source.replace("measured ModernBERT p95", "measured Qwen-router p95")
notebook["cells"][21]["source"] = source_lines(test_source)

notebook["cells"][22]["source"] = source_lines(
    """
## 12. Diagnose OOD task mix and within-dataset discrimination

Global AUC can improve merely because a router recognizes task templates.  The
tables below therefore report probability ranges and per-candidate ROC-AUC
inside each test dataset whenever both safe and unsafe labels exist.  A useful
causal router should separate individual prompts, not only assign one nearly
constant score to every member of a dataset.
"""
)

notebook["cells"][23]["source"] = source_lines(
    r'''
decisions = result.decisions.assign(gained=gained, lost=lost, routed=routed)
dataset_outcomes = decisions.groupby("dataset").agg(
    prompts=("dataset", "size"),
    routed=("routed", "sum"),
    gains=("gained", "sum"),
    losses=("lost", "sum"),
    truncated=("router_was_truncated", "sum"),
)
dataset_outcomes["net_answers"] = dataset_outcomes.gains - dataset_outcomes.losses
dataset_outcomes["routed_fraction"] = (
    dataset_outcomes.routed / dataset_outcomes.prompts
)
router_dataset_metrics = result.per_dataset_metrics.loc[
    result.per_dataset_metrics.strategy.eq("qwen15_last_token_router")
].set_index("dataset")
dataset_outcomes = dataset_outcomes.join(
    router_dataset_metrics[
        [
            "quality_retention",
            "quality_retention_lcb",
            "quality_loss_rate_ucl",
            "routed_safety_precision_lcb",
            "resource_savings",
            "conservative_resource_savings",
        ]
    ]
)

test_targets = replacement_safety_targets(
    panel.score[split.test],
    selected_training.fallback_index,
    selected_training.nonfallback_indices,
    quality_epsilon=0.0,
)
within_dataset_rows = []
test_examples = panel.examples.iloc[split.test].reset_index(drop=True)
for candidate_position, candidate_index in enumerate(
    selected_training.nonfallback_indices
):
    candidate = panel.models[candidate_index]
    probabilities = selected_training.safety_probabilities[
        split.test, candidate_index
    ]
    for dataset, positions in test_examples.groupby("dataset").indices.items():
        positions = np.asarray(positions, dtype=int)
        labels = test_targets[positions, candidate_position]
        scores = probabilities[positions]
        within_dataset_rows.append(
            {
                "dataset": dataset,
                "candidate": candidate,
                "prompts": len(positions),
                "safe_prevalence": labels.mean(),
                "probability_mean": scores.mean(),
                "probability_min": scores.min(),
                "probability_max": scores.max(),
                "probability_span": scores.max() - scores.min(),
                "within_dataset_roc_auc": (
                    roc_auc_score(labels, scores)
                    if np.unique(labels).size == 2
                    else np.nan
                ),
            }
        )
within_dataset_discrimination = pd.DataFrame(within_dataset_rows)

display(dataset_outcomes.sort_values("net_answers"))
display(within_dataset_discrimination)
display(
    decisions.groupby("selected_model").agg(
        prompts=("selected_model", "size"),
        gains=("gained", "sum"),
        losses=("lost", "sum"),
    )
)
'''
)

notebook["cells"][24]["source"] = source_lines(
    """
## 13. Corrected-scoring V6 diagnostic dashboard

These plots describe this run's corrected labels, learning curve, validation
ranking, answer gains/losses, and within-dataset discrimination. For example,
10 gains and 4 losses yield six additional correct answers. Conservative
quality, harm, subgroup, stability, and latency-overhead gates still decide
whether that outcome passes. Historical scores use a different contract and
are deliberately absent from this dashboard.
"""
)

notebook["cells"][25]["source"] = source_lines(
    r'''
best_row = selected_training.history.loc[
    selected_training.history.epoch.eq(selected_training.best_epoch)
].iloc[0]
validation_row = setup_comparison.iloc[0]
current_validation_summary = pd.DataFrame(
    [
        {
            "router": "V6 Qwen1.5B final token, scoring v2",
            "scoring_version": SCORING_CONTRACT_VERSION,
            "validation_loss": best_row.validation_safety_loss,
            "safe_roc_auc": validation_row.safe_roc_auc,
            "unsafe_average_precision": validation_row.unsafe_average_precision,
            "routed_fraction": validation_row.routed_fraction,
            "conservative_savings": validation_row.conservative_resource_savings,
        },
    ]
)
current_test_summary = pd.DataFrame(
    [
        {
            "router": "V6 Qwen1.5B final token, scoring v2",
            "scoring_version": SCORING_CONTRACT_VERSION,
            "quality_retention_lcb": router_metrics.quality_retention_lcb,
            "quality_loss_rate_ucl": router_metrics.quality_loss_rate_ucl,
            "guarded_dataset_retention_lcb": (
                router_metrics.guarded_dataset_quality_retention_lcb
            ),
            "routed_fraction": router_metrics.routed_fraction,
            "conservative_savings": router_metrics.conservative_resource_savings,
            "gains": int(gained.sum()),
            "losses": int(lost.sum()),
        },
    ]
)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].plot(
    selected_training.history.epoch,
    selected_training.history.train_safety_loss,
    marker="o",
    label="train",
)
axes[0, 0].plot(
    selected_training.history.epoch,
    selected_training.history.validation_safety_loss,
    marker="o",
    label="OOD validation",
)
axes[0, 0].set(
    title="1. Qwen router learning curve",
    xlabel="Epoch",
    ylabel="Class-balanced BCE",
)
axes[0, 0].legend()

axes[0, 1].bar(
    ["Safety ROC-AUC", "Unsafe average precision"],
    [validation_row.safe_roc_auc, validation_row.unsafe_average_precision],
    color=["tab:green", "tab:orange"],
)
axes[0, 1].set(title="2. Validation discrimination", ylim=(0, 1.0))
axes[0, 1].tick_params(axis="x", rotation=15)

axes[1, 0].bar(
    ["Gains", "Losses"],
    [int(gained.sum()), int(lost.sum())],
    color=["tab:green", "tab:red"],
)
axes[1, 0].axhline(0, color="black", linewidth=0.8)
axes[1, 0].set(title="3. Test answers gained/lost versus fallback", ylabel="Prompts")
axes[1, 0].tick_params(axis="x", rotation=15)

plot_rows = within_dataset_discrimination.dropna(
    subset=["within_dataset_roc_auc"]
)
axes[1, 1].barh(
    plot_rows.dataset + " / " + plot_rows.candidate,
    plot_rows.within_dataset_roc_auc,
    color="tab:purple",
)
axes[1, 1].axvline(0.5, color="black", linewidth=0.8)
axes[1, 1].set(
    title="4. Test within-dataset ROC-AUC",
    xlabel="AUC (only datasets with both labels)",
    xlim=(0, 1),
)

fig.suptitle("V6 Qwen1.5B final-token router — corrected scoring v2", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
v6_diagnostic_figure = fig
display(current_validation_summary)
display(current_test_summary)
'''
)

notebook["cells"][26]["source"] = source_lines(
    """
## 14. Export the reconstructable V6 artifact

The ZIP contains the Qwen LoRA adapter, final-token safety/oracle heads,
tokenizer, calibration, threshold frontier, prompt-level decisions,
within-dataset diagnostics, measured standalone router timing, original and
corrected Qwen scores, raw grading evidence, and full-task reconciliation.
`qwen_math_rescored_records.parquet` preserves every math row before the cap,
including the original worked solutions and responses used by the grader.
"""
)

notebook["cells"][27]["source"] = source_lines(
    r'''
report_dir = export_public_benchmark(result, scenario, OUTPUT_DIR)
sensitivity.to_csv(report_dir / "validation_sensitivity.csv", index=False)
setup_comparison.to_csv(report_dir / "setup_comparison.csv", index=False)
setup_threshold_search.to_csv(
    report_dir / "setup_threshold_search.csv", index=False
)
records.to_parquet(report_dir / "qwen_candidate_records.parquet", index=False)
quality_audit.to_csv(report_dir / "qwen_quality_audit.csv", index=False)
reconciliation_audit.to_csv(
    report_dir / "qwen_scoring_reconciliation.csv", index=False
)
panel_summary.to_csv(report_dir / "qwen_candidate_panel_summary.csv")
within_dataset_discrimination.to_csv(
    report_dir / "within_dataset_discrimination.csv", index=False
)
current_validation_summary.to_csv(
    report_dir / "v6_scoring_v2_validation_summary.csv", index=False
)
current_test_summary.to_csv(
    report_dir / "v6_scoring_v2_test_summary.csv", index=False
)
(report_dir / "qwen_evidence_contract.json").write_text(
    json.dumps(
        {**evidence_contract, "evidence_tag": EVIDENCE_TAG},
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
(report_dir / "published_evaluation_metadata.json").write_text(
    json.dumps(published_run_metadata, indent=2, sort_keys=True),
    encoding="utf-8",
)

artifact_dir = export_modernbert_hybrid_poc(
    selected_training,
    panel.models,
    report_dir / "qwen15_last_token_router",
    selected_threshold=result.selected_threshold,
    router_active=result.router_active,
    poc_passed=result.single_run_passed,
    failure_reasons=result.failure_reasons,
    minimum_predicted_savings=DEFAULT_CONFIG.minimum_predicted_speedup,
    validation_quality_margin=DEFAULT_CONFIG.validation_quality_margin,
    minimum_macro_quality_retention=DEFAULT_CONFIG.minimum_macro_quality_retention,
    maximum_quality_loss_rate_ucl=DEFAULT_CONFIG.maximum_quality_loss_rate_ucl,
    minimum_routed_safety_precision_lcb=(
        DEFAULT_CONFIG.minimum_routed_safety_precision_lcb
    ),
    minimum_guarded_dataset_quality_retention_lcb=(
        DEFAULT_CONFIG.minimum_guarded_dataset_quality_retention_lcb
    ),
    minimum_guarded_dataset_prompts=DEFAULT_CONFIG.minimum_guarded_dataset_prompts,
    conservative_router_overhead_s=DEFAULT_CONFIG.conservative_router_overhead_s,
    minimum_consecutive_feasible_thresholds=(
        DEFAULT_CONFIG.minimum_consecutive_feasible_thresholds
    ),
    benchmark_fingerprint=result.benchmark_fingerprint,
    setup_name=SETUP_NAME,
    oracle_auxiliary_weight=0.0,
    router_overhead_benchmark=(
        router_overhead_benchmark.summary
        if router_overhead_benchmark is not None
        else None
    ),
    router_display_name="Qwen2.5-1.5B final-token safety router",
    pooling_strategy=QWEN_LAST_TOKEN_POOLING,
    router_text_prefix=QWEN_ROUTER_TEXT_PREFIX,
    router_text_suffix=QWEN_ROUTER_TEXT_SUFFIX,
    encoder_reference_compile=None,
    config=router_config,
)
for manifest_path in (
    report_dir / "experiment_manifest.json", artifact_dir / "manifest.json"
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scoring_contract"] = scoring_contract()
    manifest["source_bundle_sha256"] = SOURCE_BUNDLE_SHA256
    manifest["math_source_aggregate_comparison"] = "diagnostic_only"
    manifest["historical_architecture_comparison_valid"] = False
    if SOURCE_BUNDLE_SHA256:
        # A pre-existing checkout can contain a different Git commit; the ZIP
        # digest identifies the actual source used for this portable run.
        manifest.setdefault("reproducibility", {})["repository_commit"] = None
        manifest["reproducibility"]["source_bundle_sha256"] = SOURCE_BUNDLE_SHA256
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
overhead_comparison.to_csv(
    report_dir / "qwen_router_overhead_comparison.csv", index=False
)
dataset_outcomes.reset_index().to_csv(
    report_dir / "investor_ood_dataset_summary.csv", index=False
)
v6_diagnostic_figure.savefig(
    report_dir / "v6_qwen_router_dashboard.png", dpi=180, bbox_inches="tight"
)
(report_dir / "v6_qwen_last_token_contract.json").write_text(
    json.dumps(
        {
            **V6_QWEN_ROUTER_CONTRACT,
            "run_id": RUN_ID,
            "run_contract_tag": RUN_CONTRACT_TAG,
            "best_epoch": selected_training.best_epoch,
            "epochs_completed": selected_training.epochs_completed,
            "selected_threshold": result.selected_threshold,
            "validation_router_active": result.router_active,
            "single_run_passed": result.single_run_passed,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
if router_overhead_benchmark is not None:
    router_overhead_benchmark.export(
        report_dir, file_stem="qwen_router_overhead"
    )

bundle_path = shutil.make_archive(str(OUTPUT_DIR.resolve()), "zip", root_dir=OUTPUT_DIR)
print("Reports:", report_dir)
print("Router artifact:", artifact_dir)
print("ZIP:", bundle_path)
try:
    from google.colab import files

    files.download(bundle_path)
except ImportError:
    pass
'''
)

notebook["cells"][28]["source"] = source_lines(
    """
## 15. Interpretation checklist and interactive showcase

- Confirm full-task counts and non-math checks passed, and inspect the displayed
  math reference differences. These do not override local pinned labels. Inspect
  corrected validation ROC-AUC and unsafe average precision alongside loss.
- Inspect probability spans and within-dataset AUC.  A higher global AUC with
  nearly constant per-dataset probabilities is still a domain shortcut.
- Require every original quality, harm, subgroup, calibration, threshold, and
  savings gate to pass.
- Compare measured Qwen-router p95 with break-even overhead.  The frozen 4/20 ms
  assumptions are frozen evaluation settings, not a production claim.
- Remember that the Qwen router is a separate inference pass and this notebook
  assumes no KV-cache reuse if the 1.5B candidate is selected.
- Treat seed 42 as development evidence.  Confirm on untouched seeds or task
  families before changing the project recommendation.
"""
)

notebook["cells"][29]["source"] = source_lines(
    r'''
# Interactive diagnostic showcase — intentionally the final notebook cell.
demo_runtime = HybridModernBERTRouterRuntime.from_training_result(
    selected_training,
    model_names=panel.models,
    fallback_model=result.fallback_model,
    selected_threshold=result.selected_threshold,
    router_active=result.router_active,
    minimum_predicted_savings=DEFAULT_CONFIG.minimum_predicted_speedup,
    scenario=scenario,
    config=router_config,
    device=DEVICE,
    router_text_prefix=QWEN_ROUTER_TEXT_PREFIX,
    router_text_suffix=QWEN_ROUTER_TEXT_SUFFIX,
    router_display_name="Qwen2.5-1.5B final-token router",
)
demo = create_gradio_demo(demo_runtime)
if LAUNCH_INTERACTIVE_DEMO:
    demo.launch(share=True, debug=False, prevent_thread_lock=True)
demo
'''
)

notebook["metadata"].pop("widgets", None)
notebook["metadata"].pop("v5_contract", None)
notebook["metadata"]["v6_contract"] = {
    "experiment": "Qwen2.5-1.5B causal final-token router ablation",
    "scoring_version": "qwen-math-verify-v2",
    "historical_architecture_comparison_valid": False,
    "router_repo": "Qwen/Qwen2.5-1.5B-Instruct",
    "pooling_strategy": "last_nonpadding_token",
    "max_input_tokens": 1024,
    "loss": "safety-only",
    "epochs": 5,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "default_split": "dataset_ood",
    "development_comparison": True,
}

TARGET.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(TARGET)
