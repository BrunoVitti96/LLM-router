"""Freeze validation choices, open the sealed test, and compute diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score
from transformers import PreTrainedTokenizerBase

from llm_router.config import RouterConfig
from llm_router.models.modernbert_router import DecisionAlignedRouter
from llm_router.utils.calibration import apply_platt, fit_platt, out_of_fold_platt
from llm_router.utils.data import RouterData
from llm_router.utils.routing import (
    blended_latency,
    route_metrics,
    search_selector,
    select_routes,
)
from llm_router.utils.training import make_encoder, predict_indices


@dataclass(frozen=True)
class EvaluationResult:
    router_active: bool
    selected_thresholds: np.ndarray
    selected_latency_blend: float
    platt_parameters: list[tuple[float, float]]
    selector_search: pd.DataFrame
    evaluation: pd.DataFrame
    report: pd.DataFrame
    calibration_diagnostics: pd.DataFrame
    latency_diagnostics: pd.DataFrame
    confidence_intervals: pd.DataFrame
    selection_by_task: pd.DataFrame
    eligibility_by_task: pd.DataFrame
    mean_diagnostic_overhead_ms: float


def evaluate_router(
    model: DecisionAlignedRouter,
    tokenizer: PreTrainedTokenizerBase,
    data: RouterData,
    config: RouterConfig,
    compute_dtype: torch.dtype,
    device: str = "cuda",
    bootstrap_samples: int = 2000,
) -> EvaluationResult:
    """Select on validation once, then evaluate exactly once on sealed test."""
    encode_indices = make_encoder(tokenizer, data.text, config)
    validation_prediction = predict_indices(
        model,
        data.validation_indices,
        encode_indices,
        config,
        compute_dtype,
        measure_batch_one_overhead=True,
        device=device,
    )
    validation_targets = data.replacement_safe[data.validation_indices]
    oof_probability, _ = out_of_fold_platt(
        validation_prediction["safety_logits"],
        validation_targets,
        data.nonfallback_names,
        config.platt_folds,
        config.seed,
    )
    selector_search, selected_validation_row = search_selector(
        validation_prediction, oof_probability, data, config
    )
    platt_parameters = [
        fit_platt(
            validation_prediction["safety_logits"][:, column],
            validation_targets[:, column],
        )
        for column in range(len(data.nonfallback_names))
    ]

    if selected_validation_row is None:
        warnings.warn(
            "No validation configuration met the 98% one-sided retention lower "
            "confidence bound with positive net latency reduction. Deployment "
            "is fallback-only.",
            stacklevel=2,
        )
        router_active = False
        selected_thresholds = np.ones(len(data.nonfallback_names), dtype=float)
        selected_latency_blend = 0.0
    else:
        router_active = True
        selected_thresholds = np.array(
            [
                float(selected_validation_row[f"threshold__{candidate}"])
                for candidate in data.nonfallback_names
            ]
        )
        selected_latency_blend = float(selected_validation_row.latency_blend)

    test_prediction = predict_indices(
        model,
        data.test_indices,
        encode_indices,
        config,
        compute_dtype,
        measure_batch_one_overhead=True,
        device=device,
    )
    test_probability = apply_platt(test_prediction["safety_logits"], platt_parameters)
    test_latency_prediction = blended_latency(
        test_prediction["latency"],
        data.test_indices,
        selected_latency_blend,
        data.task_latency_baseline,
    )
    (
        diagnostic_test_idx,
        diagnostic_eligible,
        diagnostic_selected_fallback,
        diagnostic_no_eligible,
    ) = select_routes(
        test_probability,
        test_latency_prediction,
        selected_thresholds,
        data,
        config,
    )
    if router_active:
        router_test_idx = diagnostic_test_idx
        deployed_overhead = test_prediction["overhead_s"]
    else:
        router_test_idx = np.full(len(data.test_indices), data.strongest_idx)
        deployed_overhead = np.zeros(len(data.test_indices), dtype=float)

    strategy_choices = {
        "strongest": np.full(len(data.test_indices), data.strongest_idx),
        "fastest": np.full(len(data.test_indices), data.fastest_idx),
        "fallback_relative_oracle": data.oracle_idx[data.test_indices],
        "router": router_test_idx,
    }
    evaluation = pd.DataFrame(
        {
            strategy: route_metrics(
                data.test_indices,
                choices,
                data,
                config,
                deployed_overhead if strategy == "router" else None,
            )
            for strategy, choices in strategy_choices.items()
        }
    ).T

    test_targets = data.replacement_safe[data.test_indices]
    calibration_diagnostics = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "platt_slope": platt_parameters[column][0],
                "platt_intercept": platt_parameters[column][1],
                "test_brier_score": brier_score_loss(
                    test_targets[:, column], test_probability[:, column]
                ),
                "mean_predicted_safety": test_probability[:, column].mean(),
                "observed_safety": test_targets[:, column].mean(),
            }
            for column, candidate in enumerate(data.nonfallback_names)
        ]
    ).set_index("candidate")
    latency_diagnostics = pd.DataFrame(
        [
            {
                "model": model_name,
                "test_mae_s": mean_absolute_error(
                    data.latency[data.test_indices, model_index],
                    test_latency_prediction[:, model_index],
                ),
                "test_r2": r2_score(
                    data.latency[data.test_indices, model_index],
                    test_latency_prediction[:, model_index],
                ),
                "token_mae": mean_absolute_error(
                    data.tokens[data.test_indices, model_index],
                    test_prediction["tokens"][:, model_index],
                ),
            }
            for model_index, model_name in enumerate(config.model_names)
        ]
    ).set_index("model")

    row = np.arange(len(data.test_indices))
    router_quality = data.quality[data.test_indices][row, router_test_idx]
    fallback_quality = data.quality[data.test_indices, data.strongest_idx]
    router_latency = (
        data.latency[data.test_indices][row, router_test_idx] + deployed_overhead
    )
    fallback_latency = data.latency[data.test_indices, data.strongest_idx]
    rng = np.random.default_rng(config.seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        sample = rng.integers(0, len(data.test_indices), len(data.test_indices))
        bootstrap.append(
            {
                "accuracy_delta": (
                    router_quality[sample].mean() - fallback_quality[sample].mean()
                ),
                "latency_reduction": 1.0
                - router_latency[sample].mean() / fallback_latency[sample].mean(),
            }
        )
    confidence_intervals = pd.DataFrame(bootstrap).quantile([0.025, 0.5, 0.975])

    selected_models = np.array(config.model_names, dtype=object)[router_test_idx]
    selection_by_task = pd.crosstab(
        data.table.loc[data.test_indices, "task"],
        selected_models,
        normalize="index",
    )
    eligibility_frame = pd.DataFrame(
        {
            "task": data.table.loc[data.test_indices, "task"].to_numpy(),
            **{
                candidate: diagnostic_eligible[:, model_index]
                for candidate, model_index in zip(
                    data.nonfallback_names, data.nonfallback_indices
                )
            },
        }
    )
    eligibility_by_task = eligibility_frame.groupby("task").mean()

    report = (
        data.table.loc[
            data.test_indices,
            ["prompt_id", "task", "subject", "prompt", "reference"],
        ]
        .copy()
        .reset_index(drop=True)
    )
    report["router_active"] = router_active
    report["selected_model"] = np.array(config.model_names, dtype=object)[
        router_test_idx
    ]
    report["diagnostic_selected_model"] = np.array(config.model_names, dtype=object)[
        diagnostic_test_idx
    ]
    report["selected_fallback"] = router_test_idx == data.strongest_idx
    report["diagnostic_selected_fallback"] = diagnostic_selected_fallback
    report["no_eligible_alternative"] = diagnostic_no_eligible
    report["quality"] = data.quality[data.test_indices][row, router_test_idx]
    report["fallback_quality"] = data.quality[data.test_indices, data.strongest_idx]
    report["quality_regret"] = np.maximum(0.0, report.fallback_quality - report.quality)
    report["generation_s"] = data.latency[data.test_indices][row, router_test_idx]
    report["router_overhead_s"] = deployed_overhead
    report["latency_s"] = report.generation_s + report.router_overhead_s
    for column, candidate in enumerate(data.nonfallback_names):
        model_index = data.nonfallback_indices[column]
        report[f"p_safe__{candidate}"] = test_probability[:, column]
        report[f"eligible__{candidate}"] = diagnostic_eligible[:, model_index]
    for model_index, model_name in enumerate(config.model_names):
        report[f"predicted_latency__{model_name}"] = test_latency_prediction[
            :, model_index
        ]
        report[f"measured_latency__{model_name}"] = data.latency[
            data.test_indices, model_index
        ]
        report[f"quality__{model_name}"] = data.quality[data.test_indices, model_index]

    return EvaluationResult(
        router_active=router_active,
        selected_thresholds=selected_thresholds,
        selected_latency_blend=selected_latency_blend,
        platt_parameters=platt_parameters,
        selector_search=selector_search,
        evaluation=evaluation,
        report=report,
        calibration_diagnostics=calibration_diagnostics,
        latency_diagnostics=latency_diagnostics,
        confidence_intervals=confidence_intervals,
        selection_by_task=selection_by_task,
        eligibility_by_task=eligibility_by_task,
        mean_diagnostic_overhead_ms=float(1000 * test_prediction["overhead_s"].mean()),
    )
