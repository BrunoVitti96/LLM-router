"""Exact validation selector and fallback-relative metrics."""

from __future__ import annotations

import itertools
from statistics import NormalDist

import numpy as np
import pandas as pd

from llm_router.config import RouterConfig
from llm_router.utils.data import RouterData


def blended_latency(
    neural_latency: np.ndarray,
    indices: np.ndarray,
    blend: float,
    task_latency_baseline: np.ndarray,
) -> np.ndarray:
    return blend * neural_latency + (1.0 - blend) * task_latency_baseline[indices]


def select_routes(
    safety_probability: np.ndarray,
    latency_prediction: np.ndarray,
    thresholds: np.ndarray,
    data: RouterData,
    config: RouterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.zeros_like(latency_prediction, dtype=bool)
    eligible[:, data.strongest_idx] = True
    fallback_prediction = latency_prediction[:, data.strongest_idx]
    for column, model_index in enumerate(data.nonfallback_indices):
        safe_enough = safety_probability[:, column] >= thresholds[column]
        fast_enough = latency_prediction[:, model_index] <= fallback_prediction * (
            1.0 - config.minimum_predicted_speedup
        )
        eligible[:, model_index] = safe_enough & fast_enough
    chosen = np.where(eligible, latency_prediction, np.inf).argmin(axis=1)
    selected_fallback = chosen == data.strongest_idx
    no_eligible_alternative = ~eligible[:, data.nonfallback_indices].any(axis=1)
    return chosen, eligible, selected_fallback, no_eligible_alternative


def route_metrics(
    indices: np.ndarray,
    chosen: np.ndarray,
    data: RouterData,
    config: RouterConfig,
    overhead_s: np.ndarray | None = None,
) -> dict[str, float]:
    indices = np.asarray(indices, dtype=int)
    chosen = np.asarray(chosen, dtype=int)
    row = np.arange(len(indices))
    chosen_q = data.quality[indices][row, chosen]
    chosen_generation = data.latency[indices][row, chosen]
    fallback_q = data.quality[indices, data.strongest_idx]
    fallback_generation = data.latency[indices, data.strongest_idx]
    overhead = (
        np.zeros(len(indices), dtype=float)
        if overhead_s is None
        else np.asarray(overhead_s, dtype=float)
    )
    chosen_latency = chosen_generation + overhead
    actual_safe_faster = (
        data.quality[indices] >= fallback_q[:, None] - config.quality_safety_epsilon
    ) & (data.latency[indices] < fallback_generation[:, None])
    missed_safe_opportunity = actual_safe_faster.any(axis=1) & (
        chosen_generation >= fallback_generation
    )
    regret = np.maximum(0.0, fallback_q - chosen_q)
    paired_delta = chosen_q - fallback_q
    standard_error = (
        0.0
        if len(paired_delta) < 2
        else paired_delta.std(ddof=1) / np.sqrt(len(paired_delta))
    )
    lower_delta = paired_delta.mean() - NormalDist().inv_cdf(
        config.quality_confidence
    ) * standard_error
    quality_retention_lcb = (
        fallback_q.mean() + lower_delta
    ) / max(fallback_q.mean(), 1e-9)
    return {
        "accuracy": float(chosen_q.mean()),
        "fallback_accuracy": float(fallback_q.mean()),
        "accuracy_delta": float(chosen_q.mean() - fallback_q.mean()),
        "quality_retention": float(chosen_q.mean() / max(fallback_q.mean(), 1e-9)),
        "quality_retention_lcb": float(quality_retention_lcb),
        "quality_loss_rate": float(np.mean(chosen_q < fallback_q)),
        "mean_quality_regret": float(regret.mean()),
        "p95_quality_regret": float(np.quantile(regret, 0.95, method="higher")),
        "generation_s": float(chosen_generation.mean()),
        "router_overhead_s": float(overhead.mean()),
        "latency_s": float(chosen_latency.mean()),
        "latency_reduction": float(
            1.0 - chosen_latency.mean() / fallback_generation.mean()
        ),
        "fallback_usage": float(np.mean(chosen == data.strongest_idx)),
        "missed_safe_opportunity_rate": float(missed_safe_opportunity.mean()),
    }


def search_selector(
    prediction: dict[str, np.ndarray],
    calibrated_probability: np.ndarray,
    data: RouterData,
    config: RouterConfig,
) -> tuple[pd.DataFrame, pd.Series | None]:
    indices = prediction["indices"]
    rows = []
    for blend in config.latency_blend_grid:
        latency_prediction = blended_latency(
            prediction["latency"], indices, blend, data.task_latency_baseline
        )
        for thresholds in itertools.product(
            config.safety_threshold_grid, repeat=len(data.nonfallback_names)
        ):
            chosen, eligible, _, no_eligible = select_routes(
                calibrated_probability,
                latency_prediction,
                np.asarray(thresholds),
                data,
                config,
            )
            metrics = route_metrics(
                indices, chosen, data, config, prediction["overhead_s"]
            )
            rows.append(
                {
                    "latency_blend": blend,
                    **{
                        f"threshold__{candidate}": threshold
                        for candidate, threshold in zip(
                            data.nonfallback_names, thresholds
                        )
                    },
                    **metrics,
                    "eligible_alternative_rate": float(
                        eligible[:, data.nonfallback_indices].any(axis=1).mean()
                    ),
                    "no_eligible_alternative_rate": float(no_eligible.mean()),
                }
            )
    search = pd.DataFrame(rows)
    feasible = search.loc[
        search.quality_retention_lcb.ge(config.minimum_quality_retention)
        & search.latency_reduction.gt(0)
    ]
    if feasible.empty:
        return search, None
    threshold_columns = [
        f"threshold__{candidate}" for candidate in data.nonfallback_names
    ]
    best = (
        feasible.assign(threshold_sum=feasible[threshold_columns].sum(axis=1))
        .sort_values(
            ["latency_reduction", "quality_loss_rate", "threshold_sum"],
            ascending=[False, True, False],
        )
        .iloc[0]
    )
    return search, best
