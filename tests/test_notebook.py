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


def test_hybrid_poc_notebook_is_clean_didactic_and_syntactically_valid():
    path = Path("notebooks/02_train_modernbert_hybrid_poc.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    full_text = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
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
        'SPLIT_MODE = "random"',
        "candidate_diagnostics",
        "per_dataset_metrics",
        "VALIDATION_QUALITY_MARGIN",
        "MINIMUM_ROUTED_SAFETY_PRECISION_LCB",
        "MINIMUM_GUARDED_DATASET_QUALITY_LCB",
        "CONSERVATIVE_ROUTER_OVERHEAD_S",
        "prompt_hash",
        "input_diagnostics",
        "router_was_truncated",
        "early_stopping_patience",
        "break_even_router_overhead_ms",
        "poc_passed",
        "Train ModernBERT",
        "sealed test",
        "Final POC checklist",
        "git clone --branch develop",
        "hf_hub_download",
        "archive_preview",
        "train_modernbert_hybrid_poc",
        "files.download",
    )
    assert all(lesson in full_text for lesson in required_lessons)
    assert "train_modernbert_oracle_poc" not in full_text
    assert "probability_kind=\"oracle\"" not in full_text
    assert '-e \".[notebook]\"' not in full_text
    assert 'sys.path.insert(0, SOURCE_ROOT)' in full_text
    assert 'module_name.startswith("llm_router.")' in full_text
    assert 'import llm_router' in full_text
    assert full_text.count("%pip install") == 1
    assert "replace-with" not in full_text
    assert 'rglob("bench")' not in full_text

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)
