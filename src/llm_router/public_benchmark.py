"""Public routing-benchmark ingestion, simulation, and lightweight baselines.

This module intentionally does not import torch.  It provides a cheap first-stage
experiment that can establish routing headroom before training ModernBERT.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from llm_router.analytical_latency import (
    AnalyticalModelProfile,
    HardwareProfile,
    OutputLengthPolicy,
    estimate_latency_seconds,
)

REQUIRED_RECORD_COLUMNS = {
    "example_id",
    "dataset",
    "model",
    "prompt",
    "score",
    "prompt_tokens",
    "completion_tokens",
}

DEFAULT_THRESHOLD_GRID = tuple(
    np.unique(
        np.round(
            np.concatenate(
                (
                    np.array([0.50, 0.60, 0.70, 0.75, 0.80]),
                    np.arange(0.85, 0.951, 0.005),
                    np.array([0.975, 0.99]),
                )
            ),
            3,
        )
    )
)


@dataclass(frozen=True)
class ModelProfile:
    """Economics and serving assumptions for one candidate model."""

    name: str
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    ttft_s: float | None = None
    output_tokens_per_second: float | None = None
    queue_s: float = 0.0
    parameters_billions: float | None = None
    active_parameters_billions: float | None = None
    architecture: str | None = None
    weight_bits: int = 16
    diffusion_steps: int = 1
    diffusion_block_size: int = 1
    architecture_factor: float = 1.0

    def validate(self) -> None:
        numeric = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.ttft_s,
            self.output_tokens_per_second,
            self.queue_s,
            self.parameters_billions,
            self.active_parameters_billions,
            self.architecture_factor,
        )
        if any(value is not None and value < 0 for value in numeric):
            raise ValueError(f"Negative economics value in profile {self.name!r}.")
        if self.output_tokens_per_second == 0:
            raise ValueError(f"output_tokens_per_second must be positive for {self.name!r}.")
        uses_analytical_latency = self.parameters_billions is not None
        if uses_analytical_latency and self.architecture is None:
            raise ValueError(
                f"Analytical profile {self.name!r} requires an architecture."
            )
        if self.architecture is not None and not uses_analytical_latency:
            raise ValueError(
                f"Profile {self.name!r} has an architecture but no parameter count."
            )
        if self.weight_bits <= 0 or self.diffusion_steps <= 0:
            raise ValueError("Precision and diffusion steps must be positive.")


@dataclass(frozen=True)
class EconomicsScenario:
    """A dated, explicit cost and latency scenario.

    Latency is simulated as network + queue + TTFT + completion_tokens / TPS.
    The simulator is suitable for sensitivity analysis, not a claim of measured
    production latency.
    """

    name: str
    as_of: str
    profiles: tuple[ModelProfile, ...]
    network_s: float = 0.0
    router_overhead_s: float = 0.0
    notes: str = ""
    latency_method: str = "analytical"
    effective_tflops: float = 60.0
    memory_bandwidth_gbps: float = 900.0
    fixed_model_overhead_s: float = 0.015
    output_base_tokens: float = 24.0
    output_tokens_per_prompt_token: float = 0.20
    output_min_tokens: int = 16
    output_max_tokens: int = 256

    @classmethod
    def from_json(cls, path: str | Path) -> EconomicsScenario:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_profiles = payload.pop("models")
        profiles = tuple(
            ModelProfile(name=name, **settings)
            for name, settings in raw_profiles.items()
        )
        scenario = cls(profiles=profiles, **payload)
        scenario.validate()
        return scenario

    def validate(self) -> None:
        if not self.name or not self.as_of:
            raise ValueError("A scenario requires non-empty name and as_of fields.")
        if self.network_s < 0 or self.router_overhead_s < 0:
            raise ValueError("Scenario overhead values cannot be negative.")
        if self.latency_method not in {"analytical", "throughput"}:
            raise ValueError("latency_method must be 'analytical' or 'throughput'.")
        if len({profile.name for profile in self.profiles}) != len(self.profiles):
            raise ValueError("Scenario model names must be unique.")
        for profile in self.profiles:
            profile.validate()

    @property
    def by_model(self) -> dict[str, ModelProfile]:
        return {profile.name: profile for profile in self.profiles}


@dataclass(frozen=True)
class BenchmarkPanel:
    """A complete prompt-by-model outcome panel."""

    examples: pd.DataFrame
    models: tuple[str, ...]
    score: np.ndarray
    cost: np.ndarray
    latency: np.ndarray


@dataclass(frozen=True)
class BenchmarkSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mode: str
    train_datasets: tuple[str, ...]
    validation_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]


@dataclass(frozen=True)
class PublicBenchmarkResult:
    """Reports produced by the lightweight public-benchmark experiment."""

    objective: str
    fallback_model: str
    selected_threshold: float
    router_active: bool
    split: BenchmarkSplit
    summary: pd.DataFrame
    threshold_search: pd.DataFrame
    decisions: pd.DataFrame
    candidate_diagnostics: pd.DataFrame
    per_dataset_metrics: pd.DataFrame
    router_overhead_sensitivity: pd.DataFrame
    poc_passed: bool
    failure_reasons: tuple[str, ...]
    minimum_quality_retention: float
    confidence: float
    validation_quality_margin: float = 0.0
    router_name: str = "tfidf_safety_router"


def normalized_prompt_hash(prompt: object) -> str:
    """Return a stable content identity while preserving meaningful code layout.

    Unicode representation, line endings, and trailing whitespace are normalized.
    Case and leading indentation are deliberately preserved because both can change
    the meaning of code and natural-language prompts.
    """

    normalized = unicodedata.normalize("NFKC", str(prompt or ""))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _bench_root(root: str | Path) -> Path:
    root = Path(root)
    conventional_roots = (root / "results" / "bench", root / "bench", root)
    for candidate in conventional_roots:
        if candidate.is_dir() and next(candidate.glob("*/*/*/*.json"), None):
            return candidate

    # Release archives have used different wrapper directories. Infer the
    # benchmark root from the documented dataset/split/model/file.json shape
    # instead of requiring a directory literally named ``bench``.
    if root.is_dir():
        for result_file in root.rglob("*.json"):
            if len(result_file.parents) < 4:
                continue
            candidate = result_file.parents[3]
            if next(candidate.glob("*/*/*/*.json"), None):
                return candidate

    nested = root / "results" / "bench"
    return nested if nested.is_dir() else root


def benchmark_inventory(root: str | Path) -> pd.DataFrame:
    """List datasets, source splits, models, and files without loading records."""
    bench_root = _bench_root(root)
    rows = []
    for path in bench_root.glob("*/*/*/*.json"):
        relative = path.relative_to(bench_root).parts
        rows.append(
            {
                "dataset": relative[0],
                "source_split": relative[1],
                "model": relative[2],
                "file": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_llmrouterbench(
    root: str | Path,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load standardized records from an extracted LLMRouterBench release.

    Expected layout is ``results/bench/<dataset>/<split>/<model>/*.json``.
    Filters are applied before JSON contents are loaded so a small model pool does
    not require holding the full benchmark in memory.
    """
    bench_root = _bench_root(root)
    if not bench_root.is_dir():
        raise FileNotFoundError(f"Benchmark root does not exist: {bench_root}")
    model_filter = None if models is None else set(models)
    dataset_filter = None if datasets is None else set(datasets)
    rows: list[dict] = []
    paths = sorted(bench_root.glob("*/*/*/*.json"))
    for path in paths:
        dataset, source_split, model = path.relative_to(bench_root).parts[:3]
        if model_filter is not None and model not in model_filter:
            continue
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            index = record.get("index")
            example_id = f"{dataset}::{source_split}::{index}"
            rows.append(
                {
                    "example_id": example_id,
                    "dataset": dataset,
                    "source_split": source_split,
                    "model": model,
                    "prompt": record.get("origin_query") or record.get("prompt", ""),
                    "formatted_prompt": record.get("prompt", ""),
                    "prediction": record.get("prediction", ""),
                    "ground_truth": record.get("ground_truth"),
                    "score": record.get("score"),
                    "prompt_tokens": record.get("prompt_tokens"),
                    "completion_tokens": record.get("completion_tokens"),
                    "recorded_cost": record.get("cost"),
                    "source_file": str(path),
                    "source_mtime": path.stat().st_mtime,
                }
            )
    if not rows:
        available = benchmark_inventory(bench_root)
        detail = ""
        if not available.empty:
            detail = f" Available models: {sorted(available.model.unique())}"
        raise ValueError(f"No benchmark records matched the requested filters.{detail}")
    frame = pd.DataFrame(rows).sort_values("source_mtime")
    frame = frame.drop_duplicates(["example_id", "model"], keep="last")
    numeric_columns = ["score", "prompt_tokens", "completion_tokens"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric_columns].to_numpy(float)).all():
        raise ValueError("Scores and token counts must be finite.")
    if (frame[["prompt_tokens", "completion_tokens"]] < 0).any().any():
        raise ValueError("Token counts cannot be negative.")
    return frame.reset_index(drop=True)


def simulate_economics(
    records: pd.DataFrame, scenario: EconomicsScenario
) -> pd.DataFrame:
    """Attach cost and a scenario latency proxy to outcome records.

    In the preferred ``analytical`` mode, latency depends only on model-card
    attributes and prompt tokens.  Realized completion tokens and measured
    candidate timing are deliberately excluded from the routing POC.
    """
    missing_columns = REQUIRED_RECORD_COLUMNS - set(records)
    if missing_columns:
        raise ValueError(f"Benchmark records are missing columns: {missing_columns}")
    scenario.validate()
    profiles = scenario.by_model
    missing_profiles = sorted(set(records.model) - set(profiles))
    if missing_profiles:
        raise ValueError(f"Scenario has no profiles for models: {missing_profiles}")

    output = records.copy()
    costs = np.zeros(len(output), dtype=float)
    latencies = np.zeros(len(output), dtype=float)
    cost_sources: list[str] = []
    for position, row in enumerate(output.itertuples(index=False)):
        profile = profiles[row.model]
        has_prices = (
            profile.input_price_per_million is not None
            and profile.output_price_per_million is not None
        )
        if has_prices:
            costs[position] = (
                row.prompt_tokens * profile.input_price_per_million
                + row.completion_tokens * profile.output_price_per_million
            ) / 1_000_000
            cost_sources.append("scenario_token_prices")
        elif pd.notna(getattr(row, "recorded_cost", np.nan)):
            costs[position] = float(row.recorded_cost)
            cost_sources.append("benchmark_record")
        else:
            raise ValueError(
                f"No token prices or recorded cost are available for {row.model!r}."
            )
        if scenario.latency_method == "analytical":
            if profile.parameters_billions is None or profile.architecture is None:
                raise ValueError(
                    "Analytical latency requires parameters_billions and "
                    f"architecture for {row.model!r}."
                )
            analytical_profile = AnalyticalModelProfile(
                name=profile.name,
                parameters_billions=profile.parameters_billions,
                active_parameters_billions=profile.active_parameters_billions,
                architecture=profile.architecture,
                weight_bits=profile.weight_bits,
                diffusion_steps=profile.diffusion_steps,
                diffusion_block_size=profile.diffusion_block_size,
                architecture_factor=profile.architecture_factor,
            )
            model_latency = estimate_latency_seconds(
                [row.prompt_tokens],
                analytical_profile,
                HardwareProfile(
                    effective_tflops=scenario.effective_tflops,
                    memory_bandwidth_gbps=scenario.memory_bandwidth_gbps,
                    fixed_overhead_s=scenario.fixed_model_overhead_s,
                ),
                OutputLengthPolicy(
                    base_tokens=scenario.output_base_tokens,
                    tokens_per_prompt_token=scenario.output_tokens_per_prompt_token,
                    minimum_tokens=scenario.output_min_tokens,
                    maximum_tokens=scenario.output_max_tokens,
                ),
            )[0]
            latencies[position] = scenario.network_s + profile.queue_s + model_latency
        else:
            if profile.ttft_s is None or profile.output_tokens_per_second is None:
                raise ValueError(
                    f"Throughput latency requires ttft_s and "
                    f"output_tokens_per_second for {row.model!r}."
                )
            latencies[position] = (
                scenario.network_s
                + profile.queue_s
                + profile.ttft_s
                + row.completion_tokens / profile.output_tokens_per_second
            )
    output["simulated_cost"] = costs
    output["simulated_latency_s"] = latencies
    output["latency_source"] = scenario.latency_method
    output["cost_source"] = cost_sources
    output["economics_scenario"] = scenario.name
    output["economics_as_of"] = scenario.as_of
    return output


def make_complete_panel(
    records: pd.DataFrame, models: Sequence[str] | None = None
) -> BenchmarkPanel:
    """Create aligned matrices, dropping prompts missing any selected model."""
    required = REQUIRED_RECORD_COLUMNS | {"simulated_cost", "simulated_latency_s"}
    missing = required - set(records)
    if missing:
        raise ValueError(f"Simulated benchmark records are missing columns: {missing}")
    selected_models = tuple(models or sorted(records.model.unique()))
    if len(selected_models) < 2:
        raise ValueError("Routing requires at least two models.")
    filtered = records.loc[records.model.isin(selected_models)].copy()
    counts = filtered.groupby("example_id").model.nunique()
    complete_ids = counts[counts == len(selected_models)].index
    dropped = counts.size - len(complete_ids)
    if dropped:
        warnings.warn(
            f"Dropped {dropped} prompts without a complete selected-model panel.",
            stacklevel=2,
        )
    filtered = filtered.loc[filtered.example_id.isin(complete_ids)]
    if filtered.empty:
        raise ValueError("No complete prompt-by-model panel remains.")
    metadata = (
        filtered.sort_values(["example_id", "model"])
        .groupby("example_id", sort=True)
        .agg(
            dataset=("dataset", "first"),
            source_split=("source_split", "first"),
            prompt=("prompt", "first"),
            prompt_tokens=("prompt_tokens", "median"),
        )
        .reset_index()
    )
    metadata["prompt_hash"] = metadata.prompt.map(normalized_prompt_hash)
    metadata["prompt_group_size"] = (
        metadata.groupby("prompt_hash").prompt_hash.transform("size").astype(int)
    )

    def pivot(column: str) -> np.ndarray:
        wide = filtered.pivot(index="example_id", columns="model", values=column)
        return wide.reindex(index=metadata.example_id, columns=selected_models).to_numpy(float)

    return BenchmarkPanel(
        examples=metadata,
        models=selected_models,
        score=pivot("score"),
        cost=pivot("simulated_cost"),
        latency=pivot("simulated_latency_s"),
    )


def split_benchmark(
    panel: BenchmarkPanel,
    mode: str = "dataset_ood",
    seed: int = 42,
) -> BenchmarkSplit:
    """Create 60/20/20 random or dataset-disjoint splits."""
    rng = np.random.default_rng(seed)
    datasets = np.array(sorted(panel.examples.dataset.unique()), dtype=object)
    if mode == "dataset_ood":
        if len(datasets) < 3:
            raise ValueError("dataset_ood splitting requires at least three datasets.")
        shuffled = rng.permutation(datasets)
        test_count = max(1, round(0.20 * len(shuffled)))
        validation_count = max(1, round(0.20 * len(shuffled)))
        if test_count + validation_count >= len(shuffled):
            test_count = validation_count = 1
        test_datasets = tuple(sorted(shuffled[:test_count]))
        validation_datasets = tuple(
            sorted(shuffled[test_count : test_count + validation_count])
        )
        train_datasets = tuple(sorted(shuffled[test_count + validation_count :]))
        train = np.flatnonzero(panel.examples.dataset.isin(train_datasets))
        validation = np.flatnonzero(panel.examples.dataset.isin(validation_datasets))
        test = np.flatnonzero(panel.examples.dataset.isin(test_datasets))
    elif mode == "random":
        # Five stratified group folds give an approximate 60/20/20 split while
        # ensuring repeated prompt content can never cross split boundaries.
        # This is stronger than checking example IDs because public releases may
        # contain overlapping source subsets with different IDs.
        prompt_hash = (
            panel.examples.prompt_hash.to_numpy()
            if "prompt_hash" in panel.examples
            else panel.examples.prompt.map(normalized_prompt_hash).to_numpy()
        )
        group_counts = (
            pd.DataFrame({"dataset": panel.examples.dataset, "group": prompt_hash})
            .drop_duplicates()
            .groupby("dataset")
            .size()
        )
        too_small = group_counts[group_counts < 5]
        if not too_small.empty:
            raise ValueError(
                "Random grouped splitting requires at least five distinct prompt "
                f"groups per dataset; too small: {too_small.to_dict()}."
            )
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed
        )
        fold_rows = [
            heldout
            for _, heldout in splitter.split(
                np.zeros(len(panel.examples)),
                panel.examples.dataset.to_numpy(),
                groups=prompt_hash,
            )
        ]
        test = np.sort(fold_rows[0])
        validation = np.sort(fold_rows[1])
        train = np.sort(np.concatenate(fold_rows[2:]))
        all_datasets = tuple(sorted(datasets))
        train_datasets = validation_datasets = test_datasets = all_datasets
    else:
        raise ValueError("Split mode must be 'random' or 'dataset_ood'.")
    prompt_hash = (
        panel.examples.prompt_hash.to_numpy()
        if "prompt_hash" in panel.examples
        else panel.examples.prompt.map(normalized_prompt_hash).to_numpy()
    )
    split_by_hash: dict[str, set[str]] = {}
    for split_name, indices in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for value in np.unique(prompt_hash[indices]):
            split_by_hash.setdefault(str(value), set()).add(split_name)
    leaked = [value for value, names in split_by_hash.items() if len(names) > 1]
    if leaked:
        raise ValueError(
            f"{len(leaked)} normalized prompt groups cross split boundaries. "
            "Remove cross-dataset duplicates before using this split."
        )
    return BenchmarkSplit(
        train=train,
        validation=validation,
        test=test,
        mode=mode,
        train_datasets=train_datasets,
        validation_datasets=validation_datasets,
        test_datasets=test_datasets,
    )


def _quality_retention_lcb(
    chosen_score: np.ndarray,
    fallback_score: np.ndarray,
    confidence: float,
) -> float:
    """Normal-approximation one-sided lower bound for paired retention."""
    delta = np.asarray(chosen_score) - np.asarray(fallback_score)
    standard_error = 0.0 if len(delta) < 2 else delta.std(ddof=1) / np.sqrt(len(delta))
    lower_delta = delta.mean() - NormalDist().inv_cdf(confidence) * standard_error
    return float((fallback_score.mean() + lower_delta) / max(fallback_score.mean(), 1e-12))


def _binomial_rate_ucl(events: np.ndarray, confidence: float) -> float:
    """Wilson one-sided upper bound for a binary event rate."""

    events = np.asarray(events, dtype=bool)
    if len(events) == 0:
        return 0.0
    probability = float(events.mean())
    z = NormalDist().inv_cdf(confidence)
    z2_over_n = z * z / len(events)
    center = (probability + z * z / (2 * len(events))) / (1 + z2_over_n)
    margin = (
        z
        * np.sqrt(
            probability * (1 - probability) / len(events)
            + z * z / (4 * len(events) ** 2)
        )
        / (1 + z2_over_n)
    )
    return float(min(1.0, center + margin))


def _macro_dataset_retention(
    indices: np.ndarray,
    selected_score: np.ndarray,
    fallback_score: np.ndarray,
    panel: BenchmarkPanel,
    confidence: float,
) -> tuple[float, float, int]:
    """Macro-average retention so large datasets cannot hide small-dataset harm."""

    labels = panel.examples.dataset.iloc[indices].to_numpy()
    ratios = []
    for dataset in np.unique(labels):
        included = labels == dataset
        fallback_mean = float(fallback_score[included].mean())
        if fallback_mean > 1e-12:
            ratios.append(float(selected_score[included].mean() / fallback_mean))
    if not ratios:
        return float("nan"), float("nan"), 0
    values = np.asarray(ratios)
    standard_error = 0.0 if len(values) < 2 else values.std(ddof=1) / np.sqrt(len(values))
    lower = values.mean() - NormalDist().inv_cdf(confidence) * standard_error
    return float(values.mean()), float(lower), len(values)


def _route_metrics(
    indices: np.ndarray,
    choices: np.ndarray,
    panel: BenchmarkPanel,
    resource: np.ndarray,
    fallback_idx: int,
    confidence: float,
    overhead: float = 0.0,
    quality_epsilon: float = 0.0,
) -> dict[str, float]:
    rows = np.arange(len(indices))
    selected_score = panel.score[indices][rows, choices]
    fallback_score = panel.score[indices, fallback_idx]
    selected_resource = resource[indices][rows, choices] + overhead
    fallback_resource = resource[indices, fallback_idx]
    quality_lost = selected_score < fallback_score - quality_epsilon
    routed = choices != fallback_idx
    routed_safety_precision = (
        float((~quality_lost[routed]).mean()) if routed.any() else 1.0
    )
    macro_retention, macro_lcb, macro_count = _macro_dataset_retention(
        indices, selected_score, fallback_score, panel, confidence
    )
    return {
        "quality": float(selected_score.mean()),
        "fallback_quality": float(fallback_score.mean()),
        "quality_retention": float(
            selected_score.mean() / max(fallback_score.mean(), 1e-12)
        ),
        "quality_retention_lcb": _quality_retention_lcb(
            selected_score, fallback_score, confidence
        ),
        "quality_loss_rate": float(quality_lost.mean()),
        "quality_loss_rate_ucl": _binomial_rate_ucl(quality_lost, confidence),
        "routed_safety_precision": routed_safety_precision,
        "routed_fraction": float(routed.mean()),
        "macro_dataset_quality_retention": macro_retention,
        "macro_dataset_quality_retention_lcb": macro_lcb,
        "macro_dataset_count": float(macro_count),
        "mean_resource": float(selected_resource.mean()),
        "fallback_resource": float(fallback_resource.mean()),
        "resource_savings": float(
            1.0 - selected_resource.mean() / max(fallback_resource.mean(), 1e-12)
        ),
        "fallback_usage": float(np.mean(choices == fallback_idx)),
    }


def _candidate_diagnostics(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    resource: np.ndarray,
    fallback_idx: int,
    quality_epsilon: float,
) -> pd.DataFrame:
    rows = []
    split_indices = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    for split_name, indices in split_indices.items():
        fallback_quality = panel.score[indices, fallback_idx]
        fallback_resource = resource[indices, fallback_idx]
        safe = panel.score[indices] >= (
            fallback_quality[:, None] - quality_epsilon
        )
        safe[:, fallback_idx] = True
        oracle = np.where(safe, resource[indices], np.inf).argmin(axis=1)
        for model_idx, model_name in enumerate(panel.models):
            model_resource = resource[indices, model_idx]
            candidate_score = panel.score[indices, model_idx]
            equal_quality = np.isclose(candidate_score, fallback_quality)
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "is_fallback": model_idx == fallback_idx,
                    "mean_quality": float(panel.score[indices, model_idx].mean()),
                    "mean_resource": float(model_resource.mean()),
                    "relative_resource_vs_fallback": float(
                        model_resource.mean() / max(fallback_resource.mean(), 1e-12)
                    ),
                    "safe_rate": float(safe[:, model_idx].mean()),
                    "candidate_better_rate": float(
                        ((candidate_score > fallback_quality) & ~equal_quality).mean()
                    ),
                    "candidate_equal_rate": float(equal_quality.mean()),
                    "candidate_worse_rate": float(
                        ((candidate_score < fallback_quality) & ~equal_quality).mean()
                    ),
                    "both_zero_rate": float(
                        (
                            np.isclose(candidate_score, 0.0)
                            & np.isclose(fallback_quality, 0.0)
                        ).mean()
                    ),
                    "faster_than_fallback_rate": float(
                        (model_resource < fallback_resource).mean()
                    ),
                    "oracle_selection_rate": float((oracle == model_idx).mean()),
                }
            )
    return pd.DataFrame(rows)


def _fit_safety_probabilities(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    fallback_idx: int,
    epsilon: float,
    seed: int,
) -> tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )
    train_text = panel.examples.prompt.iloc[split.train].fillna("").astype(str)
    train_features = vectorizer.fit_transform(train_text)
    all_features = vectorizer.transform(panel.examples.prompt.fillna("").astype(str))
    probabilities = np.ones_like(panel.score, dtype=float)
    fallback_score = panel.score[split.train, fallback_idx]
    for model_idx in range(len(panel.models)):
        if model_idx == fallback_idx:
            continue
        target = (
            panel.score[split.train, model_idx] >= fallback_score - epsilon
        ).astype(int)
        if np.unique(target).size == 1:
            probabilities[:, model_idx] = float(target[0])
            continue
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1_000,
            class_weight="balanced",
            random_state=seed + model_idx,
        )
        classifier.fit(train_features, target)
        probabilities[:, model_idx] = classifier.predict_proba(all_features)[:, 1]
    return probabilities, vectorizer


def _predicted_resource(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    resource: np.ndarray,
) -> np.ndarray:
    """Use train-only dataset/model medians, backing off to global medians."""
    global_median = np.median(resource[split.train], axis=0)
    prediction = np.tile(global_median, (len(panel.examples), 1))
    train_frame = panel.examples.iloc[split.train]
    for dataset in train_frame.dataset.unique():
        train_rows = split.train[train_frame.dataset.to_numpy() == dataset]
        dataset_rows = np.flatnonzero(panel.examples.dataset.eq(dataset))
        prediction[dataset_rows] = np.median(resource[train_rows], axis=0)
    return prediction


def _route_from_probability(
    probability: np.ndarray,
    predicted_resource: np.ndarray,
    threshold: float,
    fallback_idx: int,
    minimum_predicted_savings: float,
) -> np.ndarray:
    eligible = probability >= threshold
    eligible[:, fallback_idx] = True
    fallback_resource = predicted_resource[:, fallback_idx, None]
    cheap_enough = predicted_resource <= fallback_resource * (
        1.0 - minimum_predicted_savings
    )
    cheap_enough[:, fallback_idx] = True
    return np.where(eligible & cheap_enough, predicted_resource, np.inf).argmin(axis=1)


def _dataset_lookup_choices(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    fallback_idx: int,
) -> np.ndarray:
    """A coarse task-prior baseline, with fallback for unseen datasets."""
    choices = np.full(len(panel.examples), fallback_idx, dtype=int)
    train_frame = panel.examples.iloc[split.train]
    for dataset in train_frame.dataset.unique():
        train_rows = split.train[train_frame.dataset.to_numpy() == dataset]
        best = int(panel.score[train_rows].mean(axis=0).argmax())
        choices[panel.examples.dataset.eq(dataset)] = best
    return choices


def run_public_benchmark(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    objective: str = "cost",
    minimum_quality_retention: float = 0.98,
    confidence: float = 0.95,
    quality_epsilon: float = 0.0,
    minimum_predicted_savings: float = 0.02,
    threshold_grid: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    validation_quality_margin: float = 0.0,
    router_overhead_s: float = 0.0,
    seed: int = 42,
    routing_probabilities: np.ndarray | None = None,
    router_name: str | None = None,
) -> PublicBenchmarkResult:
    """Run headroom analysis and a sealed-test routing experiment.

    Thresholds are selected on validation using a one-sided confidence bound.
    Test outcomes are opened once after the threshold and activation guard freeze.
    """
    if objective not in {"cost", "latency"}:
        raise ValueError("Objective must be 'cost' or 'latency'.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0.")
    if validation_quality_margin < 0:
        raise ValueError("validation_quality_margin cannot be negative.")
    resource = panel.cost if objective == "cost" else panel.latency
    overhead = router_overhead_s if objective == "latency" else 0.0
    fallback_idx = int(panel.score[split.train].mean(axis=0).argmax())
    fallback_model = panel.models[fallback_idx]
    if routing_probabilities is None:
        probability, _ = _fit_safety_probabilities(
            panel, split, fallback_idx, quality_epsilon, seed
        )
        router_name = router_name or "tfidf_safety_router"
    else:
        probability = np.asarray(routing_probabilities, dtype=float)
        if probability.shape != panel.score.shape:
            raise ValueError("Routing probabilities must match panel.score shape.")
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0)
            or np.any(probability > 1)
        ):
            raise ValueError("Routing probabilities must be finite values in [0, 1].")
        if not np.allclose(probability[:, fallback_idx], 1.0, atol=1e-6):
            raise ValueError("The fallback safety probability must always equal one.")
        router_name = router_name or "modernbert_hybrid_router"
    # Cost requires a train-only predictor. Analytical latency is already known
    # from prompt size at route time, so no test outcomes or response lengths leak.
    resource_prediction = (
        panel.latency.copy()
        if objective == "latency"
        else _predicted_resource(panel, split, resource)
    )
    search_rows = []
    for threshold in threshold_grid:
        all_choices = _route_from_probability(
            probability,
            resource_prediction,
            float(threshold),
            fallback_idx,
            minimum_predicted_savings,
        )
        metrics = _route_metrics(
            split.validation,
            all_choices[split.validation],
            panel,
            resource,
            fallback_idx,
            confidence,
            overhead,
            quality_epsilon,
        )
        search_rows.append({"threshold": float(threshold), **metrics})
    threshold_search = pd.DataFrame(search_rows)
    validation_quality_target = minimum_quality_retention + validation_quality_margin
    feasible = threshold_search.loc[
        threshold_search.quality_retention_lcb.ge(validation_quality_target)
        & threshold_search.resource_savings.gt(0)
    ]
    if feasible.empty:
        router_active = False
        selected_threshold = 1.1
        router_choices = np.full(len(panel.examples), fallback_idx, dtype=int)
    else:
        router_active = True
        best = feasible.sort_values(
            ["resource_savings", "quality_loss_rate", "threshold"],
            ascending=[False, True, False],
        ).iloc[0]
        selected_threshold = float(best.threshold)
        router_choices = _route_from_probability(
            probability,
            resource_prediction,
            selected_threshold,
            fallback_idx,
            minimum_predicted_savings,
        )

    cheapest_single_idx = int(resource[split.train].mean(axis=0).argmin())
    lookup_choices = _dataset_lookup_choices(panel, split, fallback_idx)
    fallback_choices = np.full(len(panel.examples), fallback_idx, dtype=int)
    cheapest_choices = np.full(len(panel.examples), cheapest_single_idx, dtype=int)
    test_fallback_score = panel.score[split.test, fallback_idx]
    actual_safe = panel.score[split.test] >= test_fallback_score[:, None] - quality_epsilon
    oracle_choices = np.where(actual_safe, resource[split.test], np.inf).argmin(axis=1)

    strategies = {
        "best_single": (fallback_choices[split.test], 0.0),
        "cheapest_single": (cheapest_choices[split.test], 0.0),
        "dataset_lookup": (lookup_choices[split.test], 0.0),
        "outcome_oracle": (oracle_choices, 0.0),
        router_name: (
            router_choices[split.test],
            overhead if router_active else 0.0,
        ),
    }
    summary = pd.DataFrame(
        {
            name: _route_metrics(
                split.test,
                choices,
                panel,
                resource,
                fallback_idx,
                confidence,
                strategy_overhead,
                quality_epsilon,
            )
            for name, (choices, strategy_overhead) in strategies.items()
        }
    ).T
    summary.insert(0, "objective", objective)
    summary.insert(1, "selected_model", "dynamic")
    summary.loc["best_single", "selected_model"] = fallback_model
    summary.loc["cheapest_single", "selected_model"] = panel.models[cheapest_single_idx]
    oracle_savings = float(summary.loc["outcome_oracle", "resource_savings"])
    summary["oracle_savings_capture"] = (
        summary.resource_savings / oracle_savings if oracle_savings > 0 else np.nan
    )
    per_dataset_rows = []
    test_dataset = panel.examples.dataset.iloc[split.test].to_numpy()
    for strategy_name, (choices, strategy_overhead) in strategies.items():
        for dataset in np.unique(test_dataset):
            included = test_dataset == dataset
            dataset_indices = split.test[included]
            dataset_choices = choices[included]
            per_dataset_rows.append(
                {
                    "strategy": strategy_name,
                    "dataset": dataset,
                    "prompts": int(included.sum()),
                    **_route_metrics(
                        dataset_indices,
                        dataset_choices,
                        panel,
                        resource,
                        fallback_idx,
                        confidence,
                        strategy_overhead,
                        quality_epsilon,
                    ),
                }
            )
    per_dataset_metrics = pd.DataFrame(per_dataset_rows)

    router_metrics = summary.loc[router_name]
    if objective == "latency":
        overhead_values_s = np.unique(
            np.array([overhead, 0.010, 0.020, 0.050], dtype=float)
        )
        if router_active:
            gross_saved_s = float(
                router_metrics.fallback_resource
                - (router_metrics.mean_resource - overhead)
            )
            net_savings = (
                gross_saved_s - overhead_values_s
            ) / router_metrics.fallback_resource
        else:
            gross_saved_s = 0.0
            net_savings = np.zeros_like(overhead_values_s)
        router_overhead_sensitivity = pd.DataFrame(
            {
                "router_overhead_ms": overhead_values_s * 1_000,
                "net_latency_savings": net_savings,
                "break_even_router_overhead_ms": gross_saved_s * 1_000,
            }
        )
    else:
        router_overhead_sensitivity = pd.DataFrame(
            columns=(
                "router_overhead_ms",
                "net_latency_savings",
                "break_even_router_overhead_ms",
            )
        )
    failure_reasons = []
    if not router_active:
        failure_reasons.append(
            "No validation threshold met both the buffered quality LCB target "
            f"({validation_quality_target:.3f}) and net-savings gate."
        )
    if router_metrics.quality_retention_lcb < minimum_quality_retention:
        failure_reasons.append("Sealed-test quality retention missed its LCB gate.")
    if router_metrics.resource_savings <= 0:
        failure_reasons.append("Sealed-test net analytical savings were not positive.")
    if router_metrics.fallback_usage >= 1.0 - 1e-12:
        failure_reasons.append("The sealed-test policy routed every prompt to fallback.")
    poc_passed = not failure_reasons

    selected_test = router_choices[split.test]
    rows = np.arange(len(split.test))
    decisions = panel.examples.iloc[split.test].copy().reset_index(drop=True)
    decisions["fallback_model"] = fallback_model
    decisions["selected_model"] = np.array(panel.models, dtype=object)[selected_test]
    decisions["router_active"] = router_active
    decisions["selected_quality"] = panel.score[split.test][rows, selected_test]
    decisions["fallback_quality"] = panel.score[split.test, fallback_idx]
    decisions["quality_delta"] = (
        decisions.selected_quality - decisions.fallback_quality
    )
    decisions["quality_lost"] = decisions.quality_delta < -quality_epsilon
    decisions["selected_resource"] = resource[split.test][rows, selected_test]
    decisions["fallback_resource"] = resource[split.test, fallback_idx]
    decisions["selected_routing_probability"] = probability[split.test][
        rows, selected_test
    ]
    candidate_diagnostics = _candidate_diagnostics(
        panel, split, resource, fallback_idx, quality_epsilon
    )

    return PublicBenchmarkResult(
        objective=objective,
        fallback_model=fallback_model,
        selected_threshold=selected_threshold,
        router_active=router_active,
        split=split,
        summary=summary,
        threshold_search=threshold_search,
        decisions=decisions,
        candidate_diagnostics=candidate_diagnostics,
        per_dataset_metrics=per_dataset_metrics,
        router_overhead_sensitivity=router_overhead_sensitivity,
        poc_passed=poc_passed,
        failure_reasons=tuple(failure_reasons),
        minimum_quality_retention=minimum_quality_retention,
        confidence=confidence,
        validation_quality_margin=validation_quality_margin,
        router_name=router_name,
    )


def export_public_benchmark(
    result: PublicBenchmarkResult,
    scenario: EconomicsScenario,
    output_dir: str | Path,
) -> Path:
    """Write reproducible reports and the exact scenario assumptions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "strategy_summary.csv")
    result.threshold_search.to_csv(output_dir / "threshold_search.csv", index=False)
    result.decisions.to_parquet(output_dir / "test_decisions.parquet", index=False)
    result.candidate_diagnostics.to_csv(
        output_dir / "candidate_diagnostics.csv", index=False
    )
    result.per_dataset_metrics.to_csv(
        output_dir / "per_dataset_metrics.csv", index=False
    )
    result.router_overhead_sensitivity.to_csv(
        output_dir / "test_router_overhead_sensitivity.csv", index=False
    )
    manifest = {
        "schema_version": 3,
        "poc_passed": result.poc_passed,
        "failure_reasons": result.failure_reasons,
        "success_criteria": {
            "minimum_quality_retention": result.minimum_quality_retention,
            "validation_quality_margin": result.validation_quality_margin,
            "validation_quality_target": (
                result.minimum_quality_retention + result.validation_quality_margin
            ),
            "one_sided_confidence": result.confidence,
            "requires_positive_net_resource_savings": True,
            "requires_nontrivial_routing": True,
            "reports_macro_dataset_retention": True,
            "reports_quality_loss_rate_ucl": True,
        },
        "objective": result.objective,
        "router_name": result.router_name,
        "fallback_model": result.fallback_model,
        "selected_threshold": result.selected_threshold,
        "router_active": result.router_active,
        "split": {
            "mode": result.split.mode,
            "random_grouping": "normalized_prompt_sha256",
            "train_datasets": result.split.train_datasets,
            "validation_datasets": result.split.validation_datasets,
            "test_datasets": result.split.test_datasets,
        },
        "scenario": {
            "name": scenario.name,
            "as_of": scenario.as_of,
            "network_s": scenario.network_s,
            "router_overhead_s": scenario.router_overhead_s,
            "notes": scenario.notes,
            "latency_method": scenario.latency_method,
            "hardware": {
                "effective_tflops": scenario.effective_tflops,
                "memory_bandwidth_gbps": scenario.memory_bandwidth_gbps,
                "fixed_model_overhead_s": scenario.fixed_model_overhead_s,
            },
            "output_length_policy": {
                "base_tokens": scenario.output_base_tokens,
                "tokens_per_prompt_token": scenario.output_tokens_per_prompt_token,
                "minimum_tokens": scenario.output_min_tokens,
                "maximum_tokens": scenario.output_max_tokens,
            },
            "models": {
                profile.name: {
                    "input_price_per_million": profile.input_price_per_million,
                    "output_price_per_million": profile.output_price_per_million,
                    "ttft_s": profile.ttft_s,
                    "output_tokens_per_second": profile.output_tokens_per_second,
                    "queue_s": profile.queue_s,
                    "parameters_billions": profile.parameters_billions,
                    "active_parameters_billions": profile.active_parameters_billions,
                    "architecture": profile.architecture,
                    "weight_bits": profile.weight_bits,
                    "diffusion_steps": profile.diffusion_steps,
                    "diffusion_block_size": profile.diffusion_block_size,
                    "architecture_factor": profile.architecture_factor,
                }
                for profile in scenario.profiles
            },
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
