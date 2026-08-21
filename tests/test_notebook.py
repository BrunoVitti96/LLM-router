import ast
import json
from pathlib import Path


def test_training_notebook_is_clean_and_short():
    path = Path("notebooks/01_train_modernbert_router.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) <= 14
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)


def test_executed_hybrid_poc_notebook_is_didactic_and_syntactically_valid():
    path = Path("notebooks/02_train_modernbert_hybrid_poc.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_lessons = (
        "Leakage check",
        "hindsight oracle",
        "Understand the objective, loss, and calibration",
        "replacement safety",
        "training-only head",
        "class-balanced",
        "out-of-fold",
        "scenario sensitivity",
        "validation_sensitivity.csv",
        'RUN_ID = "random_seed_',
        "FROZEN_EXPERIMENT_PLAN",
        "get_experiment_run",
        "per_dataset_metrics",
        "SETUP_SPECS",
        "setup_comparison.csv",
        "setup_threshold_search.csv",
        "select_validation_policy",
        "choose_validation_setup",
        "MINIMUM_CONSECUTIVE_FEASIBLE_THRESHOLDS",
        "feasible_block_size",
        "prompt_hash",
        "input_diagnostics",
        "router_was_truncated",
        "early_stopping_patience",
        "router_overhead_sensitivity",
        "benchmark_modernbert_overhead",
        "candidate_latency=\"analytical-only\"",
        "router_overhead_benchmark.export",
        "create_gradio_demo",
        "LAUNCH_INTERACTIVE_DEMO",
        "single_run_passed",
        "poc_passed",
        "Train every declared setup",
        "sealed test",
        "Final interpretation checklist",
        "git clone --branch develop",
        "hf_hub_download",
        "archive_preview",
        "train_modernbert_hybrid_poc",
        "files.download",
    )
    assert all(lesson in full_text for lesson in required_lessons)
    assert "train_modernbert_oracle_poc" not in full_text
    assert 'probability_kind="oracle"' not in full_text
    assert '-e ".[notebook]"' not in full_text
    assert "sys.path.insert(0, SOURCE_ROOT)" in full_text
    assert 'module_name.startswith("llm_router.")' in full_text
    assert 'importlib.import_module("llm_router")' in full_text
    assert full_text.count("%pip install") == 1
    assert "replace-with" not in full_text
    assert 'rglob("bench")' not in full_text
    assert full_text.count("run_public_benchmark(") == 1

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)


def test_qwen_tier_v3_notebook_is_clean_didactic_and_syntactically_valid():
    path = Path("notebooks/03_train_modernbert_qwen_tiers_poc.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_lessons = (
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "1.54B",
        "3.09B",
        "7.61B",
        "EVIDENCE_TAG",
        "DATASET_REVISIONS",
        "qwen25_random_seed_42",
        "qwen25_dataset_ood_seed_42",
        "Google Drive",
        "2,700",
        "NF4 4-bit",
        "Leakage check",
        "hindsight oracle",
        "replacement-safety",
        "class-balanced",
        "out-of-fold",
        "SETUP_SPECS",
        "hybrid_r8",
        "select_validation_policy",
        "choose_validation_setup",
        "POLICY FROZEN",
        "run_public_benchmark(",
        "single_run_passed",
        "benchmark_modernbert_overhead",
        'candidate_latency="analytical-only"',
        "qwen_candidate_records.parquet",
        "qwen_evidence_contract.json",
        "create_gradio_demo",
        "files.download",
        "Final interpretation checklist",
    )
    assert all(lesson in full_text for lesson in required_lessons)
    assert full_text.count("run_public_benchmark(") == 1
    assert full_text.count("%pip install") == 1

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)
