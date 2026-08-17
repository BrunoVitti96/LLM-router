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


def test_oracle_poc_notebook_is_clean_didactic_and_syntactically_valid():
    path = Path("notebooks/02_train_modernbert_oracle_poc.ipynb")
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
        "Understand the training loss",
        "Train ModernBERT",
        "sealed test",
        "Final POC checklist",
    )
    assert all(lesson in full_text for lesson in required_lessons)

    for cell in code_cells:
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)
