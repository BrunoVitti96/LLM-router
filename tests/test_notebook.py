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


def test_qwen_tier_v4_notebook_is_safety_only_and_investor_ready():
    path = Path("notebooks/04_train_modernbert_qwen_tiers_safety_only_v4.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_contract = (
        "V4 safety-only ModernBERT router",
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "qwen25_v4_random_seed_42",
        "qwen25_v4_dataset_ood_seed_42",
        'RUN_ID = "qwen25_v4_dataset_ood_seed_42"',
        "EPOCHS = 5",
        "MINIMUM_EPOCHS = 5",
        "EARLY_STOPPING_PATIENCE = None",
        '"oracle_head_present": True',
        '"oracle_auxiliary_weight": 0.0',
        "oracle_logits",
        "minimum validation safety loss",
        "assert training.epochs_completed == EPOCHS",
        "assert not training.stopped_early",
        "def make_step_logger",
        "step_progress_callback=make_step_logger(setup_name)",
        "step_total_loss",
        "running_train_total_loss",
        "select_validation_policy",
        "POLICY FROZEN",
        "run_public_benchmark(",
        "single_run_passed",
        "Investor OOD dashboard",
        "Five-epoch learning curve",
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


def test_qwen_tier_v5_notebook_is_valid_parallel_input_experiment():
    path = Path("notebooks/05_train_modernbert_input_representation_ood_v5.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    has_retained_results = any(cell["outputs"] for cell in code_cells)
    if not has_retained_results:
        assert "widgets" not in notebook["metadata"]
        assert all(cell["execution_count"] is None for cell in code_cells)

    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_contract = (
        "V5 ModernBERT input-representation OOD experiment",
        'RUN_ID = "qwen25_v5_context_ood_seed_42"',
        '"prefix_512": {',
        '"prefix_1024": {',
        '"head_tail_1024": {',
        '"max_input_tokens": 512',
        '"max_input_tokens": 1024',
        '"input_truncation_strategy": "head_tail"',
        "ThreadPoolExecutor",
        "PARALLEL_TRAINING_REQUESTED = True",
        "PARALLEL_TRAINING_ENABLED",
        "PARALLEL_BATCH_SIZE = 4",
        "STEP_LOG_EVERY = 1",
        '"modernbert_reference_compile": MODERNBERT_REFERENCE_COMPILE',
        "assert MODERNBERT_REFERENCE_COMPILE is False",
        "ModernBERT's optional internal `torch.compile` reference path",
        "initialization_lock=MODEL_INITIALIZATION_LOCK",
        "minimum validation safety loss",
        "input_representation_comparison.csv",
        "parallel_training_facts.json",
        "v5_input_representation_contract.json",
        'V5_INPUT_CONTRACT["oracle_auxiliary_weight"]',
        "Train versus OOD-validation safety loss",
        "Does less truncation improve validation?",
        "create_gradio_demo",
        "demo.launch",
    )
    assert all(item in full_text for item in required_contract)
    assert full_text.count("run_public_benchmark(") == 1
    assert full_text.count("train_modernbert_hybrid_poc(") == 1
    assert "AutoModelForCausalLM" not in full_text
    assert ".generate(" not in full_text
    assert 'selected_spec["oracle_auxiliary_weight"]' not in full_text
    assert notebook["metadata"]["v5_contract"][
        "modernbert_reference_compile"
    ] is False

    assert notebook["cells"][-1]["cell_type"] == "code"
    assert "demo.launch" in "".join(notebook["cells"][-1]["source"])

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)


def test_qwen_tier_v6_notebook_is_clean_last_token_ablation():
    path = Path("notebooks/06_train_qwen15_last_token_router_ood_v6.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    # Preserve historical execution evidence, including failed runs. The
    # builder still emits a clean notebook when no results are retained.
    if not any(cell["outputs"] for cell in code_cells):
        assert "widgets" not in notebook["metadata"]
        assert all(cell["execution_count"] is None for cell in code_cells)

    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_contract = (
        "V6 Qwen2.5-1.5B final-token router OOD experiment",
        'RUN_ID = "qwen25_v6_qwen_router_ood_seed_42"',
        'QWEN_ROUTER_REPO = CANDIDATES["Qwen2.5-1.5B"]',
        '"pooling_strategy": QWEN_LAST_TOKEN_POOLING',
        "QWEN_ROUTER_TEXT_SUFFIX,",
        '"max_input_tokens": MAX_INPUT_TOKENS',
        "MICRO_BATCH_SIZE = 1",
        "GRADIENT_ACCUMULATION_STEPS = 4",
        '"effective_batch_size":',
        '"oracle_auxiliary_weight": 0.0',
        "minimum validation safety loss",
        "build_qwen_last_token_router",
        "format_qwen_router_text",
        "gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS",
        "replacement_safety_targets",
        "within_dataset_roc_auc",
        "V5_SEED42_REFERENCE",
        'router_name="qwen15_last_token_router"',
        "qwen_router_overhead_comparison.csv",
        "v6_qwen_last_token_contract.json",
        "development comparison",
        "never generates",
        "create_gradio_demo",
        "demo.launch",
    )
    assert all(item in full_text for item in required_contract)
    assert full_text.count("run_public_benchmark(") == 1
    assert full_text.count("train_modernbert_hybrid_poc(") == 1
    assert ".generate(" not in full_text
    assert "ThreadPoolExecutor" not in full_text
    assert notebook["metadata"]["v6_contract"]["development_comparison"] is True

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)
