"""Package the corrected notebook and source for a Colab run before Git push.

The ZIP contains only declared project files and a per-file SHA-256 manifest.
For example, changing the grader's handling of a boxed 968 changes its file
hash and the bundle hash; old results, credentials and model weights are absent.
Run from any directory with ``python scripts/build_colab_bundle.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/06_train_qwen15_last_token_router_ood_v6.ipynb"


def build_bundle() -> Path:
    """Create a deterministic source ZIP and matching ready-to-upload notebook."""

    files = [ROOT / name for name in (
        "pyproject.toml", "README.md", "audits.md", "docs/COLAB_RUNBOOK.md", NOTEBOOK,
    )]
    files.extend(sorted((ROOT / "src").rglob("*.py")))
    files.extend(sorted((ROOT / "configs").glob("*.json")))
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {
        "scoring_version": "qwen-math-verify-v2",
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(contents.items())
        },
    }
    contents["colab_source_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output = ROOT / "dist" / "colab"
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "llm_router_colab_scoring_v2.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(contents.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content)
    shutil.copyfile(ROOT / NOTEBOOK, output / Path(NOTEBOOK).name)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    print(archive_path)
    print(f"SHA-256: {digest}")
    return archive_path


if __name__ == "__main__":
    build_bundle()
