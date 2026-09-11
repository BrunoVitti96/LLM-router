import ast
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest


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
        'RUN_ID = "qwen25_v6_scoring_v2_ood_seed_42"',
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
        "score_published_sample",
        "reconcile_published_results",
        'evidence_contract["correctness_policy"] = scoring_contract()',
        "qwen_scoring_reconciliation.csv",
        "qwen_math_rescored_records.parquet",
        "validate_math_scoring_runtime",
        '"math_source_aggregate_comparison"] = "diagnostic_only"',
        "SOURCE_BUNDLE_SHA256",
        "EXPECTED_SCORING_VERSION",
        "llm_router_colab_scoring_v2.zip",
        'router_name="qwen15_last_token_router"',
        "qwen_router_overhead_comparison.csv",
        "v6_qwen_last_token_contract.json",
        "development evidence",
        "never generates",
        "create_gradio_demo",
        "demo.launch",
    )
    assert all(item in full_text for item in required_contract)
    assert full_text.count("run_public_benchmark(") == 1
    assert full_text.count("train_modernbert_hybrid_poc(") == 1
    assert ".generate(" not in full_text
    assert "ThreadPoolExecutor" not in full_text
    assert "V5_SEED42_REFERENCE" not in full_text
    assert "v5_v6_validation_comparison" not in full_text
    assert "v5_v6_test_comparison" not in full_text
    assert "qwen25_v6_qwen_router_ood_seed_42" not in full_text
    assert notebook["metadata"]["v6_contract"]["development_comparison"] is True
    assert notebook["metadata"]["v6_contract"]["scoring_version"] == (
        "qwen-math-verify-v2"
    )

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)


def _v6_cell(index):
    notebook = json.loads(
        Path("notebooks/06_train_qwen15_last_token_router_ood_v6.ipynb").read_text(
            encoding="utf-8"
        )
    )
    return "".join(notebook["cells"][index]["source"])


def test_v6_loader_reconciles_full_rows_and_preserves_grading_evidence(tmp_path):
    """Six source rows are checked before a one-prompt cap can keep only three.

    The scorer stub isolates notebook wiring; evidence tests exercise the real
    grader. For a target of 968, this wiring must preserve old score 0 beside
    corrected score 1 and retain the exact raw solution and model response.
    """
    task = "leaderboard_math_hard_counting_and_prob"
    sample_name = f"run/samples_{task}_frozen.jsonl"
    payloads = [
        {
            "doc_id": index,
            "doc_hash": f"doc-{index}",
            "arguments": [[f"Question {index}"]],
            "doc": {"solution": r"The answer is $\boxed{968}$."},
            "resps": [[r"I calculate 1024 - 56 = 968. Final: \boxed{968}"]],
            "filtered_resps": [r"\boxed{968}"],
            "target": "968",
            "exact_match,none": 0.0,
        }
        for index in range(2)
    ]
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text("\n".join(map(json.dumps, payloads)), encoding="utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_text('{"fixture": true}', encoding="utf-8")
    candidates = {
        name: {
            "evidence_repo": f"fixture/{name}",
            "evidence_revision": "frozen-revision",
            "evaluation_run": "frozen",
        }
        for name in ("Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B")
    }
    calls = []

    class FixtureHfApi:
        def list_repo_files(self, *args, **kwargs):
            return [sample_name]

    def fixture_download(*, filename, **kwargs):
        return str(sample_path if filename.endswith(".jsonl") else results_path)

    def fixture_score(payload, task_name):
        calls.append((payload["doc_id"], task_name))
        return {
            "published_metric": "exact_match,none",
            "published_score": 0.0,
            "score": 1.0,
            "score_source": "fixture-math-verify",
            "scoring_version": "qwen-math-verify-v2",
        }

    def fixture_reconcile(rows, metadata):
        assert len(rows) == 6
        assert set(metadata) == set(candidates)
        assert len(calls) == 6
        assert rows.published_score.eq(0).all()
        assert rows.score.eq(1).all()
        assert rows.raw_doc_json.map(json.loads).tolist() == [
            payload["doc"] for _ in candidates for payload in payloads
        ]
        return pd.DataFrame({
            "reconciled": [True], "aggregate_required": [False],
            "mean_matches": [False],
        })

    namespace = {
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
        "pd": pd,
        "CANDIDATES": candidates,
        "EXCLUDED_OVERLAPPING_TASKS": set(),
        "HF_TOKEN": "fixture-token",
        "MAX_PROMPTS_PER_TASK": 1,
        "OUTPUT_DIR": tmp_path / "report",
        "HfApi": FixtureHfApi,
        "hf_hub_download": fixture_download,
        "GatedRepoError": RuntimeError,
        "score_published_sample": fixture_score,
        "reconcile_published_results": fixture_reconcile,
        "log_stage": lambda *args, **kwargs: None,
        "display": lambda *args, **kwargs: None,
    }
    exec(compile(_v6_cell(6), "v6 evidence loader", "exec"), namespace)  # noqa: S102 - trusted local notebook fixture
    rows = namespace["raw_records"]
    assert rows.raw_resps_json.map(json.loads).iloc[0] == payloads[0]["resps"]
    assert rows.filtered_resps_json.map(json.loads).iloc[0] == (
        payloads[0]["filtered_resps"]
    )
    assert namespace["reconciliation_audit"].reconciled.all()
    assert (tmp_path / "report" / "qwen_scoring_reconciliation.csv").is_file()
    assert len(pd.read_parquet(
        tmp_path / "report" / "qwen_math_rescored_records.parquet"
    )) == 6
    assert "selected_keys" not in namespace

    def reject_reconciliation(rows, metadata):
        raise ValueError("Full-task aggregate mismatch")

    namespace["reconcile_published_results"] = reject_reconciliation
    with pytest.raises(ValueError, match="Full-task aggregate mismatch"):
        exec(compile(_v6_cell(6), "v6 evidence loader", "exec"), namespace)  # noqa: S102 - trusted local notebook fixture


def _v6_source_extractor():
    setup_tree = ast.parse(
        "\n".join(
            line for line in _v6_cell(2).splitlines() if not line.startswith("%")
        )
    )
    extract_function = next(
        node for node in setup_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_source_bundle"
    )
    namespace = {
        "zipfile": zipfile,
        "PurePosixPath": PurePosixPath,
        "stat": stat,
        "shutil": shutil,
    }
    exec(  # noqa: S102 - trusted local notebook function
        compile(ast.Module(body=[extract_function], type_ignores=[]), "bundle", "exec"),
        namespace,
    )
    return namespace["extract_source_bundle"]


@pytest.mark.parametrize(
    "unsafe_name", ["../escape.py", "/escape.py", "C:/escape.py", r"..\escape.py"]
)
def test_v6_source_bundle_rejects_unsafe_paths_before_writing(tmp_path, unsafe_name):
    """An unsafe second member must not leave even the first safe file written."""
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("src/valid.py", "VALID = True\n")
        archive.writestr(unsafe_name, "ESCAPED = True\n")
    output_root = tmp_path / "project"
    with pytest.raises(ValueError, match="Unsafe source-bundle path"):
        _v6_source_extractor()(bundle, output_root)
    assert not output_root.exists()


def test_v6_source_bundle_extracts_matching_project_files(tmp_path):
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("src/llm_router/qwen_evidence.py", "SCORING = 'v2'\n")
        archive.writestr("pyproject.toml", "[project]\n")
    output_root = tmp_path / "project"
    _v6_source_extractor()(bundle, output_root)
    extracted_source = output_root / "src/llm_router/qwen_evidence.py"
    assert extracted_source.read_text() == "SCORING = 'v2'\n"


def test_portable_bundle_is_reproducible_and_contains_matching_source(tmp_path):
    """The distributed notebook and grader must match the hashed ZIP sources."""
    import runpy

    builder = runpy.run_path("scripts/build_colab_bundle.py")
    project = tmp_path / "source"
    for name in (
        "pyproject.toml", "README.md", "audits.md", "docs/COLAB_RUNBOOK.md",
        builder["NOTEBOOK"], "src/llm_router/qwen_evidence.py",
    ):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n", encoding="utf-8")
    (project / "configs").mkdir()
    build = builder["build_bundle"]
    build.__globals__["ROOT"] = project
    archive_path = build()
    original = archive_path.read_bytes()
    assert build().read_bytes() == original
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("colab_source_manifest.json"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert not any("results/" in name or ".venv/" in name for name in archive.namelist())
        assert archive.read(builder["NOTEBOOK"]) == (
            archive_path.parent / Path(builder["NOTEBOOK"]).name
        ).read_bytes()
