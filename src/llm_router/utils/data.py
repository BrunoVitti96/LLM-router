"""Evidence auditing and fallback-relative training target construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_router.config import EXPECTED_MODEL_REPOS, RouterConfig
from llm_router.utils.text import format_router_input


@dataclass(frozen=True)
class Evidence:
    root: Path
    manifest: dict
    prompts: pd.DataFrame
    measurements: pd.DataFrame
    fingerprint: str


@dataclass(frozen=True)
class RouterData:
    table: pd.DataFrame
    quality: np.ndarray
    latency: np.ndarray
    tokens: np.ndarray
    masks: dict[str, np.ndarray]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    strongest_idx: int
    fallback_name: str
    fastest_idx: int
    nonfallback_indices: np.ndarray
    nonfallback_names: tuple[str, ...]
    replacement_safe: np.ndarray
    opportunity_gain: np.ndarray
    safety_example_weight: np.ndarray
    task_latency_baseline: np.ndarray
    task_model_median: pd.DataFrame
    text: np.ndarray
    oracle_idx: np.ndarray


def load_and_audit_evidence(
    root: str | Path,
    config: RouterConfig,
    current_gpu: str,
) -> Evidence:
    """Load v3's immutable panel and fail loudly on any contract drift."""
    root = Path(root)
    manifest_path = root / "run_manifest_v3.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Complete notebook 03 first; run_manifest_v3.json was not found."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == config.required_v3_schema_version
    assert manifest["prompt_template_version"] == config.required_v3_prompt_template
    assert manifest["seed"] == config.seed
    assert int(manifest["n_per_task"]) == config.n_per_task
    assert tuple(manifest["tasks"]) == config.tasks
    assert set(manifest["models"]) == set(config.model_names)
    assert current_gpu == manifest["gpu"], (
        "Router overhead must use the same GPU as candidate timings. "
        f"Evidence used {manifest['gpu']!r}; current GPU is {current_gpu!r}."
    )
    for name, repo in EXPECTED_MODEL_REPOS.items():
        assert manifest["models"][name]["repo"] == repo

    prompts_path = root / "data" / "prompts_v3.parquet"
    measurements_path = root / "data" / "measurements_v3.parquet"
    if not prompts_path.exists() or not measurements_path.exists():
        raise FileNotFoundError("Notebook 03 must finish scoring before v4 can run.")

    prompts = pd.read_parquet(prompts_path)
    measurements = pd.read_parquet(measurements_path)
    prompt_columns = {
        "prompt_id",
        "task",
        "subject",
        "num_choices",
        "prompt_words",
        "length_bin",
        "prompt",
        "reference",
    }
    measurement_columns = {
        "prompt_id",
        "task",
        "model",
        "model_repo",
        "run_fingerprint",
        "generation_s",
        "output_tokens",
        "quality",
        "parsed",
        "strict_format",
    }
    assert not (prompt_columns - set(prompts.columns))
    assert not (measurement_columns - set(measurements.columns))
    assert prompts.prompt_id.is_unique
    assert len(prompts) == config.n_per_task * len(config.tasks)
    assert (
        prompts.groupby("task").size().reindex(config.tasks).eq(config.n_per_task).all()
    )

    fingerprint = manifest["run_fingerprint"]
    measurements = measurements.drop_duplicates(["prompt_id", "model"], keep="last")
    assert set(measurements.model) == set(config.model_names)
    assert measurements.run_fingerprint.eq(fingerprint).all()
    assert len(measurements) == len(prompts) * len(config.model_names)
    assert (
        measurements.groupby("prompt_id")
        .model.nunique()
        .eq(len(config.model_names))
        .all()
    )
    assert measurements.generation_s.gt(0).all()
    assert measurements.output_tokens.ge(0).all()
    assert measurements.quality.isin([0.0, 1.0]).all()
    numeric = measurements[["generation_s", "output_tokens", "quality"]]
    assert np.isfinite(numeric).all().all()
    repositories = (
        measurements[["model", "model_repo"]]
        .drop_duplicates()
        .set_index("model")
        .model_repo.to_dict()
    )
    assert repositories == EXPECTED_MODEL_REPOS
    return Evidence(root, manifest, prompts, measurements, fingerprint)


def evidence_summary(evidence: Evidence) -> pd.DataFrame:
    return evidence.measurements.groupby(["model", "task"]).agg(
        prompts=("prompt_id", "size"),
        accuracy=("quality", "mean"),
        mean_latency_s=("generation_s", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        parse_rate=("parsed", "mean"),
        strict_format_rate=("strict_format", "mean"),
    )


def prepare_router_data(evidence: Evidence, config: RouterConfig) -> RouterData:
    """Create deterministic splits, labels, weights, and latency baselines."""

    def wide(column: str) -> pd.DataFrame:
        return evidence.measurements.pivot(
            index="prompt_id", columns="model", values=column
        ).reindex(columns=config.model_names)

    table = evidence.prompts.set_index("prompt_id").join(
        wide("quality").add_prefix("q__")
    )
    table = table.join(wide("generation_s").add_prefix("l__"))
    table = table.join(wide("output_tokens").add_prefix("t__")).reset_index()

    train_val, test = train_test_split(
        table.index,
        test_size=0.20,
        random_state=config.seed,
        stratify=table.task,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=config.seed + 1,
        stratify=table.loc[train_val, "task"],
    )
    masks = {
        "train": table.index.isin(train),
        "validation": table.index.isin(validation),
        "test": table.index.isin(test),
    }

    quality = table[[f"q__{name}" for name in config.model_names]].to_numpy(float)
    latency = table[[f"l__{name}" for name in config.model_names]].to_numpy(float)
    tokens = table[[f"t__{name}" for name in config.model_names]].to_numpy(float)
    strongest_idx = int(quality[masks["train"]].mean(axis=0).argmax())
    fallback_name = config.model_names[strongest_idx]
    nonfallback_indices = np.array(
        [index for index in range(len(config.model_names)) if index != strongest_idx]
    )
    nonfallback_names = tuple(
        config.model_names[index] for index in nonfallback_indices
    )

    fallback_quality = quality[:, strongest_idx][:, None]
    fallback_latency = latency[:, strongest_idx][:, None]
    alternative_quality = quality[:, nonfallback_indices]
    alternative_latency = latency[:, nonfallback_indices]
    replacement_safe = (
        alternative_quality >= fallback_quality - config.quality_safety_epsilon
    )
    normalized_gain = np.maximum(
        0.0,
        (fallback_latency - alternative_latency) / np.maximum(fallback_latency, 1e-9),
    )
    opportunity_gain = replacement_safe.astype(float) * normalized_gain
    quality_drop = np.maximum(0.0, fallback_quality - alternative_quality)
    safety_example_weight = np.where(
        replacement_safe,
        1.0 + config.missed_opportunity_weight * opportunity_gain,
        1.0 + config.unsafe_selection_weight * quality_drop,
    )

    train_prompt_ids = set(table.loc[masks["train"], "prompt_id"])
    train_long = evidence.measurements.loc[
        evidence.measurements.prompt_id.isin(train_prompt_ids)
    ]
    task_model_median = train_long.pivot_table(
        index="task", columns="model", values="generation_s", aggfunc="median"
    ).reindex(index=config.tasks, columns=config.model_names)
    global_model_median = np.median(latency[masks["train"]], axis=0)
    task_latency_baseline = np.vstack(
        [
            task_model_median.loc[task]
            .fillna(pd.Series(global_model_median, index=config.model_names))
            .to_numpy(float)
            for task in table.task
        ]
    )
    text = np.array(
        [
            format_router_input(
                row.prompt,
                row.task,
                str(row.subject),
                int(row.num_choices),
                str(row.length_bin),
            )
            for row in table.itertuples()
        ],
        dtype=object,
    )
    fastest_idx = int(latency[masks["train"]].mean(axis=0).argmin())
    actual_eligible = quality >= quality[:, strongest_idx][:, None]
    oracle_idx = np.where(actual_eligible, latency, np.inf).argmin(axis=1)

    return RouterData(
        table=table,
        quality=quality,
        latency=latency,
        tokens=tokens,
        masks=masks,
        train_indices=np.flatnonzero(masks["train"]),
        validation_indices=np.flatnonzero(masks["validation"]),
        test_indices=np.flatnonzero(masks["test"]),
        strongest_idx=strongest_idx,
        fallback_name=fallback_name,
        fastest_idx=fastest_idx,
        nonfallback_indices=nonfallback_indices,
        nonfallback_names=nonfallback_names,
        replacement_safe=replacement_safe,
        opportunity_gain=opportunity_gain,
        safety_example_weight=safety_example_weight,
        task_latency_baseline=task_latency_baseline,
        task_model_median=task_model_median,
        text=text,
        oracle_idx=oracle_idx,
    )
