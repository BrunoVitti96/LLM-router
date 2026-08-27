"""Build the clean safety-only v4 Colab notebook from notebook 03.

The transformation deliberately reuses notebook 03's pinned Qwen evidence,
analytical latency, calibration, policy gates, sealed-test evaluation, export,
and Gradio runtime. V4 changes only the versioned run identity, the training
contract, the investor diagnostics, and the final-cell layout.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "03_train_modernbert_qwen_tiers_poc.ipynb"
TARGET = ROOT / "notebooks" / "04_train_modernbert_qwen_tiers_safety_only_v4.ipynb"


def source_lines(text: str) -> list[str]:
    """Return notebook-compatible source lines with stable trailing newlines."""

    return text.strip("\n").splitlines(keepends=True)


def find_cell(cells: list[dict], marker: str) -> int:
    """Find exactly one cell containing ``marker``."""

    matches = [
        index
        for index, cell in enumerate(cells)
        if marker in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}; found {matches}.")
    return matches[0]


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    """Replace one inclusive/exclusive source block."""

    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

cells[find_cell(cells, "# Three-tier Qwen quality-preserving")]["source"] = source_lines(
    r"""
# V4 safety-only ModernBERT router for three Qwen tiers

This is the clean v4 Colab notebook. It keeps notebook 03's pinned published
quality evidence, analytical latency model, calibration, threshold gates,
sealed-test policy, artifact export, and interactive routing demonstration.

The training contract changes in one explicit way:

$$
\boxed{\mathcal L_{train}=\mathcal L_{safety}+0.0\mathcal L_{oracle}
=\mathcal L_{safety}}.
$$

The oracle head and oracle-loss implementation remain present for artifact and
code compatibility, but their coefficient is exactly zero, so they contribute
no training gradient. V4 trains rank-4 LoRA for all 15 epochs and restores the
checkpoint with the lowest validation safety loss.

| Tier | Candidate | Exact parameters | Quality evidence |
|---|---|---:|---|
| small | `Qwen2.5-1.5B` | 1.54B | Open LLM Leaderboard details |
| middle | `Qwen2.5-3B` | 3.09B | Open LLM Leaderboard details |
| large (~8B class) | `Qwen2.5-7B` | 7.61B | Open LLM Leaderboard details |

**No Qwen model weights are loaded or run.** Candidate latency remains an
analytical BF16 estimate. Only the ModernBERT routing path is timed.

The default run is the dataset-OOD seed-42 stress test. Entire published tasks
are held out from training, so its investor dashboard tests transfer to unseen
domains rather than only new prompts from familiar domains.
"""
)

cells[find_cell(cells, "## 2. Freeze the run")]["source"] = source_lines(
    """
## 2. Freeze the v4 run and published-evidence contracts

Change only `RUN_ID` between v4 experiments. Every seed reuses the same pinned
detail-dataset revisions, evaluation runs, safety-only loss, 15-epoch schedule,
checkpoint rule, calibration, threshold grid, and quality gates.

`random` asks prompt-level feasibility. `dataset_ood` holds out complete tasks
and is the investor-facing generalization stress test. V4 keeps at most 300
aligned prompts per task: at most 11,100 prompts and 33,300 recorded outcomes.
"""
)

config_index = find_cell(cells, "RUN_SPECS = {")
config_source = "".join(cells[config_index]["source"])
config_source = replace_between(
    config_source,
    "RUN_SPECS = {",
    "MAX_PROMPTS_PER_TASK = 300",
    '''RUN_SPECS = {
    "qwen25_v4_random_seed_42": ("random", 42),
    "qwen25_v4_random_seed_43": ("random", 43),
    "qwen25_v4_random_seed_44": ("random", 44),
    "qwen25_v4_dataset_ood_seed_42": ("dataset_ood", 42),
    "qwen25_v4_dataset_ood_seed_43": ("dataset_ood", 43),
    "qwen25_v4_dataset_ood_seed_44": ("dataset_ood", 44),
}
RUN_ID = "qwen25_v4_dataset_ood_seed_42"
SPLIT_MODE, SEED = RUN_SPECS[RUN_ID]

''',
)
config_source = config_source.replace(
    "EPOCHS = 8\nMINIMUM_EPOCHS = 2\nEARLY_STOPPING_PATIENCE = 2",
    "EPOCHS = 15\nMINIMUM_EPOCHS = 15\nEARLY_STOPPING_PATIENCE = None",
)
config_source = replace_between(
    config_source,
    "SETUP_SPECS = {",
    "CANDIDATES = {",
    '''SETUP_SPECS = {
    "safety_only_r4": {
        "lora_r": 4,
        "lora_alpha": 8,
        "oracle_auxiliary_weight": 0.0,
        "dataset_balanced_sampling": False,
    },
}
V4_TRAINING_CONTRACT = {
    "version": "v4-safety-only-15-epoch",
    "loss": "class-balanced fallback-relative safety BCE",
    "safety_loss_weight": 1.0,
    "oracle_head_present": True,
    "oracle_auxiliary_weight": 0.0,
    "epochs": EPOCHS,
    "early_stopping": False,
    "checkpoint_rule": "minimum validation safety loss across all 15 epochs",
    "lora_rank": 4,
    "lora_alpha": 8,
}
assert tuple(SETUP_SPECS) == ("safety_only_r4",)
assert SETUP_SPECS["safety_only_r4"]["oracle_auxiliary_weight"] == 0.0
assert EPOCHS == MINIMUM_EPOCHS == 15
assert EARLY_STOPPING_PATIENCE is None

''',
)
config_source = config_source.replace(
    'EVIDENCE_TAG = hashlib.sha256(\n'
    '    json.dumps(evidence_contract, sort_keys=True).encode("utf-8")\n'
    ').hexdigest()[:16]\nOUTPUT_DIR',
    'EVIDENCE_TAG = hashlib.sha256(\n'
    '    json.dumps(evidence_contract, sort_keys=True).encode("utf-8")\n'
    ').hexdigest()[:16]\n'
    'RUN_CONTRACT_TAG = hashlib.sha256(\n'
    '    json.dumps(\n'
    '        {\n'
    '            "run_id": RUN_ID,\n'
    '            "evidence_tag": EVIDENCE_TAG,\n'
    '            "training": V4_TRAINING_CONTRACT,\n'
    '        },\n'
    '        sort_keys=True,\n'
    '    ).encode("utf-8")\n'
    ').hexdigest()[:16]\n'
    'OUTPUT_DIR',
)
config_source = config_source.replace(
    '    evidence_tag=EVIDENCE_TAG,\n    gpu=torch.cuda.get_device_name(0),',
    '    evidence_tag=EVIDENCE_TAG,\n'
    '    run_contract_tag=RUN_CONTRACT_TAG,\n'
    '    loss="safety-only",\n'
    '    epochs=EPOCHS,\n'
    '    gpu=torch.cuda.get_device_name(0),',
)
cells[config_index]["source"] = source_lines(config_source)

cells[find_cell(cells, "## 6. Verify validation-only oracle")]["source"] = source_lines(
    """
## 6. Verify validation-only outcome-oracle headroom

The outcome oracle remains a diagnostic upper bound, not a training loss. It
uses recorded outcomes to ask whether safe faster choices exist under several
analytical scenarios. If oracle savings are non-positive, no prompt-only safety
classifier can create routing opportunity.

This diagnostic does not activate the oracle head and does not change
`oracle_auxiliary_weight=0.0`.
"""
)

cells[find_cell(cells, "## 7. Understand the objective")]["source"] = source_lines(
    r"""
## 7. Understand the v4 safety-only objective and calibration

For every non-fallback candidate, ModernBERT predicts:

$$
\widehat P_m(\text{safe}\mid x),\qquad
y_m(x)=\mathbf 1[Q_m(x)\ge Q_f(x)-\epsilon_q].
$$

With default $\epsilon_q=0$, matching the fallback is safe. Independent,
class-balanced binary cross-entropy is the only active training objective:

$$
\boxed{\mathcal L_{v4}=\mathcal L_{safety}}.
$$

The model still instantiates `oracle_logits`, and the shared loss function still
calculates its diagnostic value, but multiplication by zero gives:

$$
\mathcal L_{train}=1.0\mathcal L_{safety}
+0.0\mathcal L_{oracle}=\mathcal L_{safety}.
$$

For example, if a safe 1.5B example receives probability 0.8 and its class
weight is 2.33, its contribution is
$2.33[-\log(0.8)]\approx0.52$. The oracle contribution is zero regardless of
its diagnostic value.

Each candidate receives a Platt scaler. Threshold search uses out-of-fold
validation probabilities. Training runs all 15 epochs, and the model restored
for calibration is the epoch with minimum validation safety loss.
"""
)

cells[find_cell(cells, "## 8. Train every declared")]["source"] = source_lines(
    """
## 8. Train the safety-only router for all 15 epochs

V4 trains one rank-4 setup. Early stopping is disabled so every run completes
15 epochs. After the last epoch, the training function restores the checkpoint
with the lowest validation safety loss; the sealed test is still unopened.

The oracle head remains in the model and artifact schema, while its coefficient
stays exactly `0.0`.
"""
)

training_index = find_cell(cells, "def make_epoch_logger")
training_source = "".join(cells[training_index]["source"])
training_source = training_source.replace(
    '    trainings[setup_name] = training\n    log_stage(',
    '    assert training.epochs_completed == EPOCHS\n'
    '    assert not training.stopped_early\n'
    '    assert spec["oracle_auxiliary_weight"] == 0.0\n'
    '    trainings[setup_name] = training\n'
    '    log_stage(',
)
training_source = training_source.replace(
    '        epochs=training.epochs_completed,\n        truncation=',
    '        epochs=training.epochs_completed,\n'
    '        checkpoint_rule="minimum validation safety loss",\n'
    '        oracle_weight=spec["oracle_auxiliary_weight"],\n'
    '        truncation=',
)
cells[training_index]["source"] = source_lines(training_source)

cells[find_cell(cells, "## 9. Compare setups")]["source"] = source_lines(
    """
## 9. Freeze the safety-only validation policy

The one-row setup table preserves the existing report schema. Validation still
chooses the calibrated threshold and requires aggregate and macro quality,
harm-rate control, routed safety precision, subgroup protection when applicable,
positive savings at 4 ms and 20 ms overhead, and at least two adjacent feasible
thresholds. An isolated lucky threshold fails closed.
"""
)

selection_index = find_cell(cells, "policy_kwargs = {")
selection_source = "".join(cells[selection_index]["source"])
selection_source = selection_source.replace(
    "SELECTED_SETUP = choose_validation_setup(setup_comparison)\n",
    "SELECTED_SETUP = choose_validation_setup(setup_comparison)\n"
    'assert SELECTED_SETUP == "safety_only_r4"\n',
)
cells[selection_index]["source"] = source_lines(selection_source)

cells[find_cell(cells, "## 12. Diagnose task mix")]["source"] = source_lines(
    """
## 12. Diagnose OOD task mix, tier usage, and truncation

For dataset-OOD runs, every displayed test task was absent from training. The
detailed diagnostics expose net answers, quality retention, routing rate,
conservative analytical savings, selected-tier counts, and truncation. Random
runs still generate the same tables but must not be labeled OOD evidence.
"""
)

diagnostics_index = find_cell(cells, "dataset_outcomes = decisions.groupby")
diagnostics_source = "".join(cells[diagnostics_index]["source"])
diagnostics_source = diagnostics_source.replace(
    '["resource_savings", "conservative_resource_savings"]',
    '[\n'
    '            "quality_retention",\n'
    '            "quality_retention_lcb",\n'
    '            "quality_loss_rate_ucl",\n'
    '            "routed_safety_precision_lcb",\n'
    '            "resource_savings",\n'
    '            "conservative_resource_savings",\n'
    '        ]',
)
cells[diagnostics_index]["source"] = source_lines(diagnostics_source)

dashboard_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": source_lines(
        """
## 13. Investor OOD dashboard

These four charts answer four investor questions without hiding the safety
failure modes:

1. **Did training converge, and which epoch was retained?**
2. **Is there a stable validation trade-off between routing and savings?**
3. **Do held-out domains preserve fallback quality as routing increases?**
4. **Which capacity tier receives traffic, and where do harmful routes occur?**

The validation frontier is the only chart used for policy selection. The OOD
test charts are descriptive reports created after the sealed test opens; they
must never be used to retune the threshold.
"""
    ),
}

dashboard_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_lines(
        '''
history = selected_training.history.copy()
frontier = frozen_validation_policy.threshold_search.copy()
investor_ood_dataset_summary = dataset_outcomes.reset_index().copy()
investor_mode_label = (
    "Dataset-OOD: complete held-out tasks"
    if SPLIT_MODE == "dataset_ood"
    else "Random split: not an OOD claim"
)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Learning curve and best validation checkpoint.
axes[0, 0].plot(
    history.epoch,
    history.train_safety_loss,
    marker="o",
    label="Train safety loss",
)
axes[0, 0].plot(
    history.epoch,
    history.validation_safety_loss,
    marker="o",
    label="Validation safety loss",
)
best_row = history.loc[history.epoch.eq(selected_training.best_epoch)].iloc[0]
axes[0, 0].axvline(
    selected_training.best_epoch,
    color="tab:green",
    linestyle="--",
    label=f"Retained epoch {selected_training.best_epoch}",
)
axes[0, 0].scatter(
    [selected_training.best_epoch],
    [best_row.validation_safety_loss],
    color="tab:green",
    s=90,
    zorder=5,
)
axes[0, 0].set(
    title="1. Fifteen-epoch learning curve",
    xlabel="Epoch",
    ylabel="Class-balanced BCE",
)
axes[0, 0].legend()

# 2. Validation threshold frontier; green points passed every frozen gate.
point_colors = np.where(frontier.is_feasible, "tab:green", "lightgray")
axes[0, 1].scatter(
    100 * frontier.routed_fraction,
    100 * frontier.conservative_resource_savings,
    c=point_colors,
    edgecolor="black",
    linewidth=0.4,
)
diagnostic_threshold = frozen_validation_policy.diagnostic_threshold
diagnostic_row = frontier.loc[
    frontier.threshold.eq(diagnostic_threshold)
].iloc[0]
axes[0, 1].scatter(
    [100 * diagnostic_row.routed_fraction],
    [100 * diagnostic_row.conservative_resource_savings],
    marker="*",
    s=220,
    color="gold",
    edgecolor="black",
    label=f"Frozen/diagnostic threshold {diagnostic_threshold:.3f}",
)
axes[0, 1].axhline(0, color="black", linewidth=0.8)
axes[0, 1].set(
    title="2. Validation routing-savings frontier",
    xlabel="Prompts routed to smaller tiers (%)",
    ylabel="Analytical savings at 20 ms (%)",
)
axes[0, 1].legend()

# 3. Domain safety map. Bubble area represents held-out test prompts.
domain_plot = investor_ood_dataset_summary.replace([np.inf, -np.inf], np.nan)
domain_plot = domain_plot.dropna(subset=["quality_retention"])
domain_colors = np.where(
    domain_plot.quality_retention_lcb >= DEFAULT_CONFIG.minimum_macro_quality_retention,
    "tab:green",
    "tab:red",
)
axes[1, 0].scatter(
    100 * domain_plot.routed_fraction,
    100 * domain_plot.quality_retention,
    s=40 + 3 * domain_plot.prompts,
    c=domain_colors,
    alpha=0.70,
    edgecolor="black",
    linewidth=0.5,
)
axes[1, 0].axhline(100, color="black", linewidth=0.8, label="Fallback parity")
axes[1, 0].axhline(
    100 * DEFAULT_CONFIG.minimum_macro_quality_retention,
    color="tab:red",
    linestyle="--",
    linewidth=1.0,
    label="98% retention reference",
)
for _, row in domain_plot.nsmallest(min(8, len(domain_plot)), "quality_retention").iterrows():
    axes[1, 0].annotate(
        row["dataset"],
        (100 * row.routed_fraction, 100 * row.quality_retention),
        fontsize=8,
        xytext=(4, 3),
        textcoords="offset points",
    )
axes[1, 0].set(
    title="3. Held-out-domain safety map",
    xlabel="Prompts routed to smaller tiers (%)",
    ylabel="Quality retention (%)",
)
axes[1, 0].legend()

# 4. Traffic allocation with harmful routes visible inside each tier.
model_allocation = decisions.groupby("selected_model").agg(
    prompts=("selected_model", "size"),
    harmful=("quality_lost", "sum"),
).reindex(panel.models, fill_value=0)
model_allocation["non_harmful"] = (
    model_allocation.prompts - model_allocation.harmful
)
axes[1, 1].bar(
    model_allocation.index,
    model_allocation.non_harmful,
    color="tab:blue",
    label="Non-harmful decisions",
)
axes[1, 1].bar(
    model_allocation.index,
    model_allocation.harmful,
    bottom=model_allocation.non_harmful,
    color="tab:red",
    label="Harmful decisions",
)
axes[1, 1].set(
    title="4. Selected-tier allocation and harm",
    xlabel="Chosen model",
    ylabel="Test prompts",
)
axes[1, 1].tick_params(axis="x", rotation=15)
axes[1, 1].legend()

fig.suptitle(
    f"V4 safety-only investor dashboard — {investor_mode_label}",
    fontsize=16,
)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
investor_ood_figure = fig

display(
    investor_ood_dataset_summary.sort_values("quality_retention_lcb")[[
        "dataset",
        "prompts",
        "routed_fraction",
        "quality_retention",
        "quality_retention_lcb",
        "quality_loss_rate_ucl",
        "conservative_resource_savings",
        "net_answers",
    ]]
)
'''
    ),
}

insert_after = diagnostics_index + 1
cells[insert_after:insert_after] = [dashboard_markdown, dashboard_code]

export_markdown_index = find_cell(cells, "## 13. Export the reconstructable")
cells[export_markdown_index]["source"] = source_lines(
    """
## 14. Export the reconstructable v4 artifact

The ZIP contains the pinned Qwen evidence, safety-only 15-epoch contract,
training history, retained checkpoint, calibration, threshold frontier,
sealed-test decisions, OOD investor dashboard, ModernBERT adapter and heads,
analytical scenario, and router-only timing diagnostics.

The oracle head remains in the compatible artifact, but the exported training
contract records `oracle_auxiliary_weight=0.0`.
"""
)

export_index = find_cell(cells, "report_dir = export_public_benchmark")
export_source = "".join(cells[export_index]["source"])
demo_start = export_source.index(
    "demo_runtime = HybridModernBERTRouterRuntime.from_training_result("
)
bundle_start = export_source.index("bundle_path = shutil.make_archive", demo_start)
export_source = export_source[:demo_start] + export_source[bundle_start:]
export_source = export_source.replace(
    'overhead_comparison.to_csv(\n    report_dir / "modernbert_overhead_comparison.csv", index=False\n)\n',
    'overhead_comparison.to_csv(\n'
    '    report_dir / "modernbert_overhead_comparison.csv", index=False\n'
    ')\n'
    'investor_ood_dataset_summary.to_csv(\n'
    '    report_dir / "investor_ood_dataset_summary.csv", index=False\n'
    ')\n'
    'investor_ood_figure.savefig(\n'
    '    report_dir / "investor_ood_dashboard.png", dpi=180, bbox_inches="tight"\n'
    ')\n'
    '(report_dir / "v4_training_contract.json").write_text(\n'
    '    json.dumps(\n'
    '        {\n'
    '            **V4_TRAINING_CONTRACT,\n'
    '            "run_id": RUN_ID,\n'
    '            "run_contract_tag": RUN_CONTRACT_TAG,\n'
    '            "best_epoch": selected_training.best_epoch,\n'
    '            "epochs_completed": selected_training.epochs_completed,\n'
    '        },\n'
    '        indent=2,\n'
    '        sort_keys=True,\n'
    '    ),\n'
    '    encoding="utf-8",\n'
    ')\n',
)
cells[export_index]["source"] = source_lines(export_source)

checklist_index = find_cell(cells, "## 14. Final interpretation checklist")
cells[checklist_index]["source"] = source_lines(
    """
## 15. Final interpretation checklist and interactive showcase

Before an investor claim, confirm:

- all 15 epochs completed and `best_epoch` is the minimum validation safety loss;
- `oracle_auxiliary_weight=0.0` and the oracle head is present only for compatibility;
- the ZIP contains evidence, training, calibration, threshold, decision, timing,
  and OOD dashboard artifacts;
- `single_run_passed` and every failure reason are explicit;
- random and dataset-OOD runs are never pooled into one claim;
- candidate latency is described as analytical BF16, not measured serving;
- no held-out domain supplies all gains or absorbs hidden harm; and
- analytical savings are never converted to guaranteed dollars.

The final cell launches the interactive investor showcase. A user enters a
prompt and sees each candidate's calibrated safety probability, analytical
latency, eligibility, chosen model, fallback status, and estimated savings. It
does not generate a Qwen answer and does not load Qwen weights.
"""
)

interactive_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_lines(
        '''
# Interactive investor showcase — intentionally the final notebook cell.
demo_runtime = HybridModernBERTRouterRuntime.from_training_result(
    selected_training,
    model_names=panel.models,
    fallback_model=result.fallback_model,
    selected_threshold=result.selected_threshold,
    router_active=result.router_active,
    minimum_predicted_savings=DEFAULT_CONFIG.minimum_predicted_speedup,
    scenario=scenario,
    config=selected_config,
    device=DEVICE,
)
demo = create_gradio_demo(demo_runtime)
if LAUNCH_INTERACTIVE_DEMO:
    demo.launch(share=True, debug=False, prevent_thread_lock=True)
demo
'''
    ),
}
cells.append(interactive_code)

notebook["metadata"]["v4_contract"] = {
    "loss": "safety-only",
    "oracle_auxiliary_weight": 0.0,
    "epochs": 15,
    "checkpoint_rule": "minimum validation safety loss",
    "default_split": "dataset_ood",
}

TARGET.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(TARGET)
