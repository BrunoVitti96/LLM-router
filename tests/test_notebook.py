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


def test_qwen_tier_v3_notebook_is_didactic_and_syntactically_valid():
    path = Path("notebooks/03_train_modernbert_qwen_tiers_poc.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_lessons = (
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "1.54B",
        "3.09B",
        "7.61B",
        "EVIDENCE_TAG",
        "evidence_revision",
        "evaluation_run",
        "qwen25_random_seed_42",
        "qwen25_dataset_ood_seed_42",
        'REPOSITORY_BRANCH = "poc"',
        "REQUIRED_EVIDENCE_HELPER",
        "Repository/source mismatch",
        "Open LLM Leaderboard",
        "HF_TOKEN",
        "key icon",
        "enable notebook access",
        'os.environ.get("HF_TOKEN")',
        "getpass",
        "hidden session-only token prompt",
        "Hugging Face token accepted",
        "accept the dataset access conditions",
        "Read access to gated repositories",
        "33,300",
        "No Qwen model weights",
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
        "qwen_quality_audit.csv",
        "qwen_evidence_contract.json",
        "published_evaluation_metadata.json",
        "audit_aligned_outcomes",
        "published_binary_score",
        "prompt_level_strict_acc",
        "correctness_policy",
        "is_correct",
        "correct / outcomes",
        "create_gradio_demo",
        "files.download",
        "Final interpretation checklist",
    )
    assert all(lesson in full_text for lesson in required_lessons)
    assert full_text.count("run_public_benchmark(") == 1
    assert full_text.count("%pip install") == 1
    assert "AutoModelForCausalLM" not in full_text
    assert "BitsAndBytesConfig" not in full_text
    assert ".generate(" not in full_text
    assert "pipeline(" not in full_text

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)


def test_qwen_tier_v4_notebook_is_clean_safety_only_and_investor_ready():
    path = Path("notebooks/04_train_modernbert_qwen_tiers_safety_only_v4.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_contract = (
        "V4 safety-only ModernBERT router",
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "qwen25_v4_random_seed_42",
        "qwen25_v4_dataset_ood_seed_42",
        'RUN_ID = "qwen25_v4_dataset_ood_seed_42"',
        "EPOCHS = 15",
        "MINIMUM_EPOCHS = 15",
        "EARLY_STOPPING_PATIENCE = None",
        '"oracle_head_present": True',
        '"oracle_auxiliary_weight": 0.0',
        "oracle_logits",
        "minimum validation safety loss",
        "assert training.epochs_completed == EPOCHS",
        "assert not training.stopped_early",
        "select_validation_policy",
        "POLICY FROZEN",
        "run_public_benchmark(",
        "single_run_passed",
        "Investor OOD dashboard",
        "Fifteen-epoch learning curve",
        "Validation routing-savings frontier",
        "Held-out-domain safety map",
        "Selected-tier allocation and harm",
        "investor_ood_dashboard.png",
        "investor_ood_dataset_summary.csv",
        "v4_training_contract.json",
        "create_gradio_demo",
        "demo.launch",
        "Interactive investor showcase",
    )
    assert all(item in full_text for item in required_contract)
    assert '"hybrid_r4": {' not in full_text
    assert '"hybrid_r8": {' not in full_text
    assert full_text.count("run_public_benchmark(") == 1
    assert full_text.count("%pip install") == 1
    assert "AutoModelForCausalLM" not in full_text
    assert ".generate(" not in full_text

    assert notebook["cells"][-1]["cell_type"] == "code"
    final_source = "".join(notebook["cells"][-1]["source"])
    assert "create_gradio_demo" in final_source
    assert "demo.launch" in final_source

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)
