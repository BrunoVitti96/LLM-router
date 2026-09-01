"""Build notebook 05: parallel input-representation OOD experiment.

Notebook 04 is used as the workflow source because it contains the current
published-Qwen evidence, safety-only loss, policy gates, exports, and demo. The
builder clears every output and changes only the versioned run identity, input
representations, parallel training orchestration, diagnostics, and artifact
contract.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "04_train_modernbert_qwen_tiers_safety_only_v4.ipynb"
TARGET = ROOT / "notebooks" / "05_train_modernbert_input_representation_ood_v5.ipynb"


def source_lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def find_cell(cells: list[dict], marker: str) -> int:
    matches = [
        index
        for index, cell in enumerate(cells)
        if marker in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}; found {matches}.")
    return matches[0]


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]
for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    else:
        markdown = "".join(cell.get("source", []))
        markdown = markdown.replace("V4", "V5").replace("v4", "v5")
        markdown = markdown.replace("Notebook 04", "Notebook 05")
        cell["source"] = source_lines(markdown)

cells[find_cell(cells, "# V5 safety-only ModernBERT router")]["source"] = source_lines(
    r"""
# V5 ModernBERT input-representation OOD experiment

Notebook 04 showed a widening train/validation loss gap and routed only
truncated OOD test prompts. Notebook 05 tests whether missing context is a
cause, while keeping the evidence, split, safety loss, LoRA rank, seed,
calibration, gates, and five-epoch checkpoint rule fixed.

Three variants train against exactly the same prompt rows:

| Variant | Token budget | Retained context | Controlled question |
|---|---:|---|---|
| `prefix_512` | 512 | first 512 tokens | reproduces notebook-04 input |
| `prefix_1024` | 1,024 | first 1,024 tokens | does a larger budget help? |
| `head_tail_1024` | 1,024 | beginning + ending | does the discarded tail matter? |

The three models are launched concurrently on one GPU when the memory preflight
passes. Model initialization is serialized to avoid Hugging Face cache races;
training then overlaps. GPU RAM capacity enables concurrency but does not imply
a 3× speedup because all workers still share one compute device.

The active objective remains:

$$
\boxed{\mathcal L_{v5}=\mathcal L_{safety}+0.0\mathcal L_{oracle}}.
$$

Only the validation-selected representation opens the sealed test. Notebook 04
results are exploratory history and are not used to tune notebook 05 after its
test is opened.
"""
)

cells[find_cell(cells, "## 2. Freeze the v5 run")]["source"] = source_lines(
    """
## 2. Freeze the v5 input-representation contract

Change only `RUN_ID` between v5 experiments. The three representations share
one evidence fingerprint, split, loss, optimizer settings, five epochs, and
validation gates. The default is dataset-OOD seed 42.

The parallel preflight requires a CUDA GPU with at least 14 GiB total memory
and 80% free before model construction. Otherwise the same variants run
sequentially. This fallback changes wall-clock execution, not the experiment.
"""
)

config_index = find_cell(cells, "RUN_SPECS = {")
config_source = "".join(cells[config_index]["source"])
config_source = config_source.replace(
    "from dataclasses import replace\n",
    "import threading\n"
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n"
    "from dataclasses import replace\n",
)
config_source = config_source.replace(
    "from llm_router.modernbert_poc import (\n",
    "from llm_router.models.modernbert_router import (\n"
    "    MODERNBERT_REFERENCE_COMPILE,\n"
    ")\n"
    "from llm_router.modernbert_poc import (\n",
)
config_source = replace_between(
    config_source,
    "RUN_SPECS = {",
    "MAX_PROMPTS_PER_TASK = 300",
    '''RUN_SPECS = {
    "qwen25_v5_context_random_seed_42": ("random", 42),
    "qwen25_v5_context_random_seed_43": ("random", 43),
    "qwen25_v5_context_random_seed_44": ("random", 44),
    "qwen25_v5_context_ood_seed_42": ("dataset_ood", 42),
    "qwen25_v5_context_ood_seed_43": ("dataset_ood", 43),
    "qwen25_v5_context_ood_seed_44": ("dataset_ood", 44),
}
RUN_ID = "qwen25_v5_context_ood_seed_42"
SPLIT_MODE, SEED = RUN_SPECS[RUN_ID]

''',
)
config_source = config_source.replace(
    'LAUNCH_INTERACTIVE_DEMO = True\nDEVICE =',
    'LAUNCH_INTERACTIVE_DEMO = True\n'
    'PARALLEL_TRAINING_REQUESTED = True\n'
    'PARALLEL_MIN_TOTAL_GPU_GIB = 14.0\n'
    'PARALLEL_MIN_FREE_FRACTION = 0.80\n'
    'PARALLEL_BATCH_SIZE = 4\n'
    'STEP_LOG_EVERY = 1\n'
    'DEVICE =',
)
config_source = replace_between(
    config_source,
    "SETUP_SPECS = {",
    "CANDIDATES = {",
    '''SETUP_SPECS = {
    "prefix_512": {
        "max_input_tokens": 512,
        "input_truncation_strategy": "prefix",
    },
    "prefix_1024": {
        "max_input_tokens": 1024,
        "input_truncation_strategy": "prefix",
    },
    "head_tail_1024": {
        "max_input_tokens": 1024,
        "input_truncation_strategy": "head_tail",
    },
}
V5_INPUT_CONTRACT = {
    "version": "v5-input-representation-parallel",
    "loss": "class-balanced fallback-relative safety BCE",
    "oracle_auxiliary_weight": 0.0,
    "lora_rank": 4,
    "lora_alpha": 8,
    "epochs": EPOCHS,
    "checkpoint_rule": "minimum validation safety loss",
    "parallel_training_requested": PARALLEL_TRAINING_REQUESTED,
    "modernbert_reference_compile": MODERNBERT_REFERENCE_COMPILE,
    "batch_size_per_variant": PARALLEL_BATCH_SIZE,
    "variants": SETUP_SPECS,
}
assert tuple(SETUP_SPECS) == (
    "prefix_512", "prefix_1024", "head_tail_1024"
)
assert EPOCHS == MINIMUM_EPOCHS == 5
assert EARLY_STOPPING_PATIENCE is None
assert MODERNBERT_REFERENCE_COMPILE is False

''',
)
config_source = config_source.replace("V4_TRAINING_CONTRACT", "V5_INPUT_CONTRACT")
config_source = config_source.replace(
    'loss="safety-only",\n    epochs=EPOCHS,',
    'loss="safety-only",\n'
    '    input_experiment="512-prefix vs 1024-prefix vs 1024-head-tail",\n'
    '    parallel_requested=PARALLEL_TRAINING_REQUESTED,\n'
    '    epochs=EPOCHS,',
)
cells[config_index]["source"] = source_lines(config_source)

cells[find_cell(cells, "## 7. Understand the v5")]["source"] = source_lines(
    r"""
## 7. Understand the controlled input experiment

The safety target and loss do not change. Only the encoded prompt changes.
`prefix` reproduces tokenizer truncation from the right. `head_tail` tokenizes
without truncation, then retains half of the budget from the beginning and half
from the end, including normal special tokens.

For a 1,600-token router input:

- `prefix_512` discards 1,088 tokens;
- `prefix_1024` discards 576 tokens;
- `head_tail_1024` keeps roughly 512 beginning and 512 ending tokens.

All three use the same safety-only objective, rank-4 LoRA, learning rates, split,
seed, and five epochs. Each restores its own minimum validation safety-loss
checkpoint before validation policy comparison.
"""
)

cells[find_cell(cells, "## 8. Train the safety-only")]["source"] = source_lines(
    """
## 8. Train three input representations concurrently

The memory preflight records total and free GPU memory. A T4 with sufficient
headroom launches three worker threads, each with batch size 4. Model downloads
and construction are serialized; optimization overlaps. If the preflight fails,
the notebook runs the exact same configurations sequentially.

ModernBERT's optional internal `torch.compile` reference path is explicitly
disabled. PyTorch's Dynamo/FX compiler state is not safe for these concurrent
worker threads; eager execution computes the same encoder equations without the
compile race. For example, three variants still train $3 \times 5=15$ total
model-epochs, but none can enter a shared Dynamo graph-compilation state.

Step callbacks fire and print after every optimizer step. Each line includes the
variant name so the three interleaved parallel logs remain attributable. Epoch
summaries always print. Concurrency mode and memory facts are exported.
"""
)

training_index = find_cell(cells, "def make_step_logger")
cells[training_index]["source"] = source_lines(
    '''
PRINT_LOCK = threading.Lock()
MODEL_INITIALIZATION_LOCK = threading.Lock()


def make_step_logger(setup_name):
    def report(row):
        step = int(row["step_in_epoch"])
        final_step = int(row["steps_per_epoch"])
        if step != 1 and step != final_step and step % STEP_LOG_EVERY:
            return
        with PRINT_LOCK:
            print(
                f"[{setup_name}] epoch={int(row['epoch'])}/{int(row['epochs'])} "
                f"step={step}/{final_step} "
                f"loss={row['step_total_loss']:.6f} "
                f"running={row['running_train_total_loss']:.6f}",
                flush=True,
            )
    return report


def make_epoch_logger(setup_name):
    def report(row):
        marker = "BEST" if row["is_best_epoch"] else "    "
        with PRINT_LOCK:
            print(
                f"[{setup_name}] epoch={int(row['epoch'])}/{EPOCHS} {marker} "
                f"train={row['train_total_loss']:.4f} "
                f"validation={row['validation_total_loss']:.4f} "
                f"best_epoch={int(row['best_epoch_so_far'])} "
                f"seconds={row['epoch_seconds']:.1f}"
            )
    return report


gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info()
gpu_total_gib = gpu_total_bytes / 2**30
gpu_free_gib = gpu_free_bytes / 2**30
gpu_free_fraction = gpu_free_bytes / gpu_total_bytes
PARALLEL_TRAINING_ENABLED = bool(
    PARALLEL_TRAINING_REQUESTED
    and gpu_total_gib >= PARALLEL_MIN_TOTAL_GPU_GIB
    and gpu_free_fraction >= PARALLEL_MIN_FREE_FRACTION
)
parallel_training_facts = {
    "requested": PARALLEL_TRAINING_REQUESTED,
    "enabled": PARALLEL_TRAINING_ENABLED,
    "gpu": torch.cuda.get_device_name(0),
    "total_gib": gpu_total_gib,
    "free_gib_before_training": gpu_free_gib,
    "free_fraction_before_training": gpu_free_fraction,
    "workers": len(SETUP_SPECS) if PARALLEL_TRAINING_ENABLED else 1,
    "batch_size_per_variant": PARALLEL_BATCH_SIZE,
}
display(pd.Series(parallel_training_facts, name="parallel training preflight"))

setup_configs = {
    setup_name: replace(
        DEFAULT_CONFIG,
        seed=SEED,
        lora_r=4,
        lora_alpha=8,
        max_input_tokens=spec["max_input_tokens"],
        input_truncation_strategy=spec["input_truncation_strategy"],
    )
    for setup_name, spec in SETUP_SPECS.items()
}


def train_variant(setup_name):
    spec = SETUP_SPECS[setup_name]
    setup_config = setup_configs[setup_name]
    log_stage("router training started", setup=setup_name, **spec)
    training = train_modernbert_hybrid_poc(
        panel,
        split,
        config=setup_config,
        epochs=EPOCHS,
        batch_size=PARALLEL_BATCH_SIZE,
        learning_rate=1e-4,
        head_learning_rate=2e-4,
        minimum_epochs=MINIMUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        quality_epsilon=0.0,
        safety_loss_weight=1.0,
        oracle_auxiliary_weight=0.0,
        dataset_balanced_sampling=False,
        device=DEVICE,
        progress_callback=make_epoch_logger(setup_name),
        step_progress_callback=make_step_logger(setup_name),
        initialization_lock=MODEL_INITIALIZATION_LOCK,
    )
    assert training.epochs_completed == EPOCHS
    assert not training.stopped_early
    training.model.to("cpu")
    log_stage(
        "router training finished",
        setup=setup_name,
        best_epoch=training.best_epoch,
        truncation=f"{training.input_diagnostics['truncation_rate']:.2%}",
    )
    return training


trainings = {}
if PARALLEL_TRAINING_ENABLED:
    with ThreadPoolExecutor(max_workers=len(SETUP_SPECS)) as executor:
        futures = {
            executor.submit(train_variant, setup_name): setup_name
            for setup_name in SETUP_SPECS
        }
        for future in as_completed(futures):
            setup_name = futures[future]
            try:
                trainings[setup_name] = future.result()
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    raise RuntimeError(
                        "Parallel training exhausted GPU memory. Restart the runtime, "
                        "set PARALLEL_TRAINING_REQUESTED=False, and rerun all cells."
                    ) from error
                raise
else:
    for setup_name in SETUP_SPECS:
        trainings[setup_name] = train_variant(setup_name)

trainings = {name: trainings[name] for name in SETUP_SPECS}
torch.cuda.empty_cache()
assert tuple(trainings) == tuple(SETUP_SPECS)
'''
)

cells[find_cell(cells, "## 9. Freeze the safety-only")]["source"] = source_lines(
    """
## 9. Select one representation using validation only

Every representation receives independent calibration and threshold search on
the same validation prompts. The existing gate-aware selector chooses one setup;
the sealed test remains unopened. The table adds token budget, strategy, and
truncation rate so a lower loss cannot hide a more expensive input.
"""
)

selection_index = find_cell(cells, "policy_kwargs = {")
selection_source = "".join(cells[selection_index]["source"])
selection_source = selection_source.replace(
    "setup_comparison = build_setup_comparison(trainings, selections)\n",
    "setup_comparison = build_setup_comparison(trainings, selections)\n"
    "setup_comparison['max_input_tokens'] = setup_comparison.setup.map(\n"
    "    lambda name: SETUP_SPECS[name]['max_input_tokens']\n"
    ")\n"
    "setup_comparison['input_truncation_strategy'] = setup_comparison.setup.map(\n"
    "    lambda name: SETUP_SPECS[name]['input_truncation_strategy']\n"
    ")\n"
    "setup_comparison['truncation_rate'] = setup_comparison.setup.map(\n"
    "    lambda name: trainings[name].input_diagnostics['truncation_rate']\n"
    ")\n",
)
selection_source = selection_source.replace(
    'assert SELECTED_SETUP == "safety_only_r4"\n', ""
)
cells[selection_index]["source"] = source_lines(selection_source)

overhead_index = find_cell(cells, "router_overhead_benchmark = None")
overhead_source = "".join(cells[overhead_index]["source"])
overhead_source = overhead_source.replace(
    "        max_input_tokens=selected_config.max_input_tokens,\n",
    "        max_input_tokens=selected_config.max_input_tokens,\n"
    "        input_truncation_strategy=(\n"
    "            selected_config.input_truncation_strategy\n"
    "        ),\n",
)
cells[overhead_index]["source"] = source_lines(overhead_source)

test_index = find_cell(cells, "result = run_public_benchmark(")
test_source = "".join(cells[test_index]["source"])
test_source = test_source.replace(
    '        "router_was_truncated": selected_training.router_was_truncated,\n',
    '        "router_was_truncated": selected_training.router_was_truncated,\n'
    '        "router_input_max_tokens": np.full(\n'
    '            len(panel.examples), selected_config.max_input_tokens\n'
    '        ),\n'
    '        "router_input_truncation_strategy": np.full(\n'
    '            len(panel.examples), selected_config.input_truncation_strategy\n'
    '        ),\n',
)
cells[test_index]["source"] = source_lines(test_source)

cells[find_cell(cells, "## 13. Investor OOD dashboard")]["source"] = source_lines(
    """
## 13. Input-representation and selected-policy dashboard

The upper panels compare validation evidence before test selection: learning
curves, generalization gaps, and truncation. The lower panels show validation
routing economics and the selected setup's sealed OOD domain outcomes. Test
charts are descriptive and must not be used to retune notebook 05.
"""
)

dashboard_index = find_cell(cells, "history = selected_training.history.copy()")
cells[dashboard_index]["source"] = source_lines(
    '''
representation_rows = []
for setup_name, training in trainings.items():
    best_row = training.history.loc[
        training.history.epoch.eq(training.best_epoch)
    ].iloc[0]
    comparison_row = setup_comparison.set_index("setup").loc[setup_name]
    representation_rows.append(
        {
            "setup": setup_name,
            "max_input_tokens": SETUP_SPECS[setup_name]["max_input_tokens"],
            "input_truncation_strategy": SETUP_SPECS[setup_name][
                "input_truncation_strategy"
            ],
            "best_epoch": training.best_epoch,
            "best_train_loss": best_row.train_safety_loss,
            "best_validation_loss": best_row.validation_safety_loss,
            "generalization_gap": (
                best_row.validation_safety_loss - best_row.train_safety_loss
            ),
            "truncation_rate": training.input_diagnostics["truncation_rate"],
            "safe_roc_auc": comparison_row.safe_roc_auc,
            "unsafe_average_precision": comparison_row.unsafe_average_precision,
            "validation_routed_fraction": comparison_row.routed_fraction,
            "validation_conservative_savings": (
                comparison_row.conservative_resource_savings
            ),
            "selected_for_test": setup_name == SELECTED_SETUP,
        }
    )
input_representation_comparison = pd.DataFrame(representation_rows)

investor_ood_dataset_summary = dataset_outcomes.reset_index().copy()
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for setup_name, training in trainings.items():
    axes[0, 0].plot(
        training.history.epoch,
        training.history.validation_safety_loss,
        marker="o",
        label=f"{setup_name} validation",
    )
    axes[0, 0].plot(
        training.history.epoch,
        training.history.train_safety_loss,
        linestyle="--",
        alpha=0.65,
        label=f"{setup_name} train",
    )
axes[0, 0].set(
    title="1. Train versus OOD-validation safety loss",
    xlabel="Epoch",
    ylabel="Class-balanced BCE",
)
axes[0, 0].legend(fontsize=8)

colors = np.where(
    input_representation_comparison.selected_for_test, "tab:green", "tab:blue"
)
axes[0, 1].scatter(
    100 * input_representation_comparison.truncation_rate,
    input_representation_comparison.best_validation_loss,
    s=130,
    c=colors,
    edgecolor="black",
)
for row in input_representation_comparison.itertuples():
    axes[0, 1].annotate(
        row.setup,
        (100 * row.truncation_rate, row.best_validation_loss),
        xytext=(5, 4),
        textcoords="offset points",
    )
axes[0, 1].set(
    title="2. Does less truncation improve validation?",
    xlabel="Inputs exceeding the token budget (%)",
    ylabel="Best validation safety loss",
)

axes[1, 0].scatter(
    100 * input_representation_comparison.validation_routed_fraction,
    100 * input_representation_comparison.validation_conservative_savings,
    s=130,
    c=colors,
    edgecolor="black",
)
for row in input_representation_comparison.itertuples():
    axes[1, 0].annotate(
        row.setup,
        (
            100 * row.validation_routed_fraction,
            100 * row.validation_conservative_savings,
        ),
        xytext=(5, 4),
        textcoords="offset points",
    )
axes[1, 0].axhline(0, color="black", linewidth=0.8)
axes[1, 0].set(
    title="3. Validation routing and savings",
    xlabel="Prompts routed (%)",
    ylabel="Analytical savings at 20 ms (%)",
)

domain_plot = investor_ood_dataset_summary.replace([np.inf, -np.inf], np.nan)
domain_plot = domain_plot.dropna(subset=["quality_retention"])
axes[1, 1].scatter(
    100 * domain_plot.routed_fraction,
    100 * domain_plot.quality_retention,
    s=40 + 3 * domain_plot.prompts,
    c=np.where(
        domain_plot.quality_retention_lcb
        >= DEFAULT_CONFIG.minimum_macro_quality_retention,
        "tab:green",
        "tab:red",
    ),
    alpha=0.7,
    edgecolor="black",
)
axes[1, 1].axhline(100, color="black", linewidth=0.8)
axes[1, 1].axhline(98, color="tab:red", linestyle="--")
for row in domain_plot.itertuples():
    axes[1, 1].annotate(
        row.dataset,
        (100 * row.routed_fraction, 100 * row.quality_retention),
        fontsize=7,
        xytext=(4, 3),
        textcoords="offset points",
    )
axes[1, 1].set(
    title=f"4. Selected OOD policy: {SELECTED_SETUP}",
    xlabel="Prompts routed (%)",
    ylabel="Quality retention (%)",
)

fig.suptitle("V5 input-representation OOD dashboard", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
investor_ood_figure = fig
display(input_representation_comparison.sort_values("best_validation_loss"))
display(investor_ood_dataset_summary.sort_values("quality_retention_lcb"))
'''
)

export_markdown_index = find_cell(cells, "## 14. Export the reconstructable")
cells[export_markdown_index]["source"] = source_lines(
    """
## 14. Export the reconstructable v5 artifact

The ZIP records all three input representations, concurrency preflight, per-
variant histories and input diagnostics, validation comparison, selected sealed
test, OOD dashboard, selected adapter, and interactive runtime. The selected
artifact stores both token budget and truncation strategy.
"""
)

export_index = find_cell(cells, "report_dir = export_public_benchmark")
export_source = "".join(cells[export_index]["source"])
export_source = export_source.replace(
    '    training.calibration_diagnostics.to_csv(\n'
    '        setup_dir / "calibration_diagnostics.csv", index=False\n'
    '    )\n',
    '    training.calibration_diagnostics.to_csv(\n'
    '        setup_dir / "calibration_diagnostics.csv", index=False\n'
    '    )\n'
    '    (setup_dir / "input_diagnostics.json").write_text(\n'
    '        json.dumps(training.input_diagnostics, indent=2), encoding="utf-8"\n'
    '    )\n',
)
export_source = export_source.replace(
    '(report_dir / "v4_training_contract.json").write_text(',
    '(report_dir / "v5_input_representation_contract.json").write_text(',
)
export_source = export_source.replace("**V4_TRAINING_CONTRACT", "**V5_INPUT_CONTRACT")
export_source = export_source.replace(
    'overhead_comparison.to_csv(\n    report_dir / "modernbert_overhead_comparison.csv", index=False\n)\n',
    'overhead_comparison.to_csv(\n'
    '    report_dir / "modernbert_overhead_comparison.csv", index=False\n'
    ')\n'
    'input_representation_comparison.to_csv(\n'
    '    report_dir / "input_representation_comparison.csv", index=False\n'
    ')\n'
    '(report_dir / "parallel_training_facts.json").write_text(\n'
    '    json.dumps(parallel_training_facts, indent=2), encoding="utf-8"\n'
    ')\n',
)
export_source = export_source.replace(
    '            "epochs_completed": selected_training.epochs_completed,\n',
    '            "epochs_completed": selected_training.epochs_completed,\n'
    '            "selected_setup": SELECTED_SETUP,\n'
    '            "parallel_training": parallel_training_facts,\n',
)
cells[export_index]["source"] = source_lines(export_source)

checklist_index = find_cell(cells, "## 15. Final interpretation checklist")
cells[checklist_index]["source"] = source_lines(
    """
## 15. Interpretation checklist and interactive showcase

Before attributing improvement to truncation:

- confirm the three variants used identical evidence, split, loss, seed, LoRA,
  optimizer rates, batch size, and five epochs;
- confirm concurrency mode and GPU memory facts were exported;
- compare both truncation rate and validation loss—not training loss alone;
- require improved validation ROC-AUC/unsafe precision, not only calibration;
- open the sealed test for the validation-selected representation only;
- inspect selected test routing by truncation status and dataset;
- treat a parallel single-seed winner as exploratory until repeated; and
- never claim measured Qwen latency or production savings.

The final cell launches the selected representation using the same prefix or
head-tail encoder used during training.
"""
)

notebook["metadata"].pop("v4_contract", None)
# The executed v4 source may contain megabytes of widget state. It is output
# state, not part of the v5 experiment contract, so a clean build removes it.
notebook["metadata"].pop("widgets", None)
notebook["metadata"]["v5_contract"] = {
    "experiment": "input representation OOD ablation",
    "variants": {
        "prefix_512": {"max_input_tokens": 512, "strategy": "prefix"},
        "prefix_1024": {"max_input_tokens": 1024, "strategy": "prefix"},
        "head_tail_1024": {"max_input_tokens": 1024, "strategy": "head_tail"},
    },
    "parallel_training_requested": True,
    "modernbert_reference_compile": False,
    "loss": "safety-only",
    "epochs": 5,
    "default_split": "dataset_ood",
}

TARGET.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(TARGET)
