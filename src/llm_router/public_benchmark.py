"""Public routing-benchmark ingestion, simulation, and lightweight baselines.

This module intentionally does not import torch.  It provides a cheap first-stage
experiment that can establish routing headroom before training ModernBERT.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


REQUIRED_RECORD_COLUMNS = {
    "example_id",
    "dataset",
    "model",
    "prompt",
    "score",
    "prompt_tokens",
    "completion_tokens",
}


@dataclass(frozen=True)
class ModelProfile:
    """Economics and serving assumptions for one candidate model."""

    name: str
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    ttft_s: float | None = None
    output_tokens_per_second: float | None = None
    queue_s: float = 0.0

    def validate(self) -> None:
        numeric = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.ttft_s,
            self.output_tokens_per_second,
            self.queue_s,
        )
        if any(value is not None and value < 0 for value in numeric):
            raise ValueError(f"Negative economics value in profile {self.name!r}.")
        if self.output_tokens_per_second == 0:
            raise ValueError(f"output_tokens_per_second must be positive for {self.name!r}.")


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

    @classmethod
    def from_json(cls, path: str | Path) -> "EconomicsScenario":
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


def _bench_root(root: str | Path) -> Path:
    root = Path(root)
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
    """Attach realized cost and simulated latency to outcome records."""
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
        if profile.ttft_s is None or profile.output_tokens_per_second is None:
            raise ValueError(
                f"Latency simulation requires ttft_s and "
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
        train_parts, validation_parts, test_parts = [], [], []
        for dataset in datasets:
            indices = np.flatnonzero(panel.examples.dataset.eq(dataset))
            indices = rng.permutation(indices)
            n_test = max(1, round(0.20 * len(indices)))
            n_validation = max(1, round(0.20 * len(indices)))
            if n_test + n_validation >= len(indices):
                raise ValueError(
                    f"Dataset {dataset!r} needs at least three complete prompts."
                )
            test_parts.append(indices[:n_test])
            validation_parts.append(indices[n_test : n_test + n_validation])
            train_parts.append(indices[n_test + n_validation :])
        train = np.sort(np.concatenate(train_parts))
        validation = np.sort(np.concatenate(validation_parts))
        test = np.sort(np.concatenate(test_parts))
        all_datasets = tuple(sorted(datasets))
        train_datasets = validation_datasets = test_datasets = all_datasets
    else:
        raise ValueError("Split mode must be 'random' or 'dataset_ood'.")
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


def _route_metrics(
    indices: np.ndarray,
    choices: np.ndarray,
    panel: BenchmarkPanel,
    resource: np.ndarray,
    fallback_idx: int,
    confidence: float,
    overhead: float = 0.0,
) -> dict[str, float]:
    rows = np.arange(len(indices))
    selected_score = panel.score[indices][rows, choices]
    fallback_score = panel.score[indices, fallback_idx]
    selected_resource = resource[indices][rows, choices] + overhead
    fallback_resource = resource[indices, fallback_idx]
    return {
        "quality": float(selected_score.mean()),
        "fallback_quality": float(fallback_score.mean()),
        "quality_retention": float(
            selected_score.mean() / max(fallback_score.mean(), 1e-12)
        ),
        "quality_retention_lcb": _quality_retention_lcb(
            selected_score, fallback_score, confidence
        ),
        "quality_loss_rate": float(np.mean(selected_score < fallback_score)),
        "mean_resource": float(selected_resource.mean()),
        "fallback_resource": float(fallback_resource.mean()),
        "resource_savings": float(
            1.0 - selected_resource.mean() / max(fallback_resource.mean(), 1e-12)
        ),
        "fallback_usage": float(np.mean(choices == fallback_idx)),
    }


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
    threshold_grid: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    router_overhead_s: float = 0.0,
    seed: int = 42,
) -> PublicBenchmarkResult:
    """Run headroom analysis and a TF-IDF safety-router experiment.

    Thresholds are selected on validation using a one-sided confidence bound.
    Test outcomes are opened once after the threshold and activation guard freeze.
    """
    if objective not in {"cost", "latency"}:
        raise ValueError("Objective must be 'cost' or 'latency'.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0.")
    resource = panel.cost if objective == "cost" else panel.latency
    overhead = router_overhead_s if objective == "latency" else 0.0
    fallback_idx = int(panel.score[split.train].mean(axis=0).argmax())
    fallback_model = panel.models[fallback_idx]
    probability, _ = _fit_safety_probabilities(
        panel, split, fallback_idx, quality_epsilon, seed
    )
    resource_prediction = _predicted_resource(panel, split, resource)

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
        )
        search_rows.append({"threshold": float(threshold), **metrics})
    threshold_search = pd.DataFrame(search_rows)
    feasible = threshold_search.loc[
        threshold_search.quality_retention_lcb.ge(minimum_quality_retention)
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
        "tfidf_safety_router": (
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
            )
            for name, (choices, strategy_overhead) in strategies.items()
        }
    ).T
    summary.insert(0, "objective", objective)
    summary.insert(1, "selected_model", "dynamic")
    summary.loc["best_single", "selected_model"] = fallback_model
    summary.loc["cheapest_single", "selected_model"] = panel.models[cheapest_single_idx]

    selected_test = router_choices[split.test]
    rows = np.arange(len(split.test))
    decisions = panel.examples.iloc[split.test].copy().reset_index(drop=True)
    decisions["fallback_model"] = fallback_model
    decisions["selected_model"] = np.array(panel.models, dtype=object)[selected_test]
    decisions["router_active"] = router_active
    decisions["selected_quality"] = panel.score[split.test][rows, selected_test]
    decisions["fallback_quality"] = panel.score[split.test, fallback_idx]
    decisions["selected_resource"] = resource[split.test][rows, selected_test]
    decisions["fallback_resource"] = resource[split.test, fallback_idx]
    decisions["p_safe_selected"] = probability[split.test][rows, selected_test]

    return PublicBenchmarkResult(
        objective=objective,
        fallback_model=fallback_model,
        selected_threshold=selected_threshold,
        router_active=router_active,
        split=split,
        summary=summary,
        threshold_search=threshold_search,
        decisions=decisions,
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
    manifest = {
        "schema_version": 1,
        "objective": result.objective,
        "fallback_model": result.fallback_model,
        "selected_threshold": result.selected_threshold,
        "router_active": result.router_active,
        "split": {
            "mode": result.split.mode,
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
            "models": {
                profile.name: {
                    "input_price_per_million": profile.input_price_per_million,
                    "output_price_per_million": profile.output_price_per_million,
                    "ttft_s": profile.ttft_s,
                    "output_tokens_per_second": profile.output_tokens_per_second,
                    "queue_s": profile.queue_s,
                }
                for profile in scenario.profiles
            },
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
