"""Validation-only comparison helpers for ModernBERT experiment setups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from llm_router.public_benchmark import ValidationPolicySelection


def summarize_validation_setup(
    setup_name: str,
    training: Any,
    selection: ValidationPolicySelection,
) -> dict[str, Any]:
    """Summarize one setup without consulting sealed-test outcomes.

    For example, a setup may save 0.8% at 20 ms overhead but be rejected because
    only one threshold passes.  Both the savings and the stability failure remain
    visible in the comparison instead of collapsing the result to one boolean.
    """

    selected = selection.threshold_search.loc[
        selection.threshold_search.threshold.eq(selection.diagnostic_threshold)
    ].iloc[0]
    calibration = training.calibration_diagnostics
    best_history = training.history.loc[
        training.history.epoch.eq(training.best_epoch)
    ].iloc[0]
    return {
        "setup": setup_name,
        "router_active": selection.router_active,
        "selected_threshold": (
            selection.selected_threshold if selection.router_active else None
        ),
        "diagnostic_threshold": selection.diagnostic_threshold,
        "validation_total_loss": float(best_history.validation_total_loss),
        "best_epoch": training.best_epoch,
        "epochs_completed": training.epochs_completed,
        "dataset_balanced_sampling": training.dataset_balanced_sampling,
        "training_minutes": training.training_seconds / 60.0,
        "calibrated_brier": float(calibration.calibrated_brier.mean()),
        "calibrated_ece": float(calibration.calibrated_ece.mean()),
        "safe_roc_auc": float(calibration.safe_roc_auc.mean()),
        "unsafe_average_precision": float(calibration.unsafe_average_precision.mean()),
        "quality_retention_lcb": float(selected.quality_retention_lcb),
        "macro_quality_retention_lcb": float(
            selected.macro_dataset_quality_retention_lcb
        ),
        "quality_loss_rate_ucl": float(selected.quality_loss_rate_ucl),
        "routed_safety_precision_lcb": float(selected.routed_safety_precision_lcb),
        "guarded_dataset_retention_lcb": float(
            selected.guarded_dataset_quality_retention_lcb
        ),
        "safe_opportunity_recall": float(selected.safe_opportunity_recall),
        "routed_fraction": float(selected.routed_fraction),
        "nominal_resource_savings": float(selected.resource_savings),
        "conservative_resource_savings": float(selected.conservative_resource_savings),
        "feasible_block_size": int(selected.feasible_block_size),
        "gate_pass_count": int(selected.gate_pass_count),
        "failure_reasons": " | ".join(selection.failure_reasons),
    }


def build_setup_comparison(
    trainings: Mapping[str, Any],
    selections: Mapping[str, ValidationPolicySelection],
) -> pd.DataFrame:
    """Build the validation-only setup leaderboard."""

    if set(trainings) != set(selections):
        raise ValueError("Trainings and selections must contain the same setup names.")
    return pd.DataFrame(
        [
            summarize_validation_setup(name, trainings[name], selections[name])
            for name in trainings
        ]
    )


def choose_validation_setup(comparison: pd.DataFrame) -> str:
    """Choose a setup using validation safety first, then conservative savings.

    If no setup activates, the most informative near-miss is returned so the final
    sealed evaluation still fails closed with that setup's fallback-only policy.
    """

    required = {
        "setup",
        "router_active",
        "conservative_resource_savings",
        "routed_safety_precision_lcb",
        "validation_total_loss",
    }
    missing = required - set(comparison)
    if missing:
        raise ValueError(f"Comparison is missing columns: {sorted(missing)}")
    ordered = comparison.sort_values(
        [
            "router_active",
            "conservative_resource_savings",
            "routed_safety_precision_lcb",
            "validation_total_loss",
        ],
        ascending=[False, False, False, True],
    )
    return str(ordered.iloc[0].setup)


def combine_threshold_searches(
    selections: Mapping[str, ValidationPolicySelection],
) -> pd.DataFrame:
    """Stack setup threshold frontiers for plotting and artifact export."""

    return pd.concat(
        [
            selection.threshold_search.assign(setup=name)
            for name, selection in selections.items()
        ],
        ignore_index=True,
    )
