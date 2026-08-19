from types import SimpleNamespace

import pandas as pd

from llm_router.experiment_comparison import (
    build_setup_comparison,
    choose_validation_setup,
)


def fake_training(loss: float):
    diagnostics = pd.DataFrame(
        [
            {
                "calibrated_brier": 0.15,
                "calibrated_ece": 0.02,
                "safe_roc_auc": 0.75,
                "unsafe_average_precision": 0.55,
            }
        ]
    )
    history = pd.DataFrame([{"epoch": 1, "validation_total_loss": loss}])
    return SimpleNamespace(
        calibration_diagnostics=diagnostics,
        history=history,
        best_epoch=1,
        epochs_completed=2,
        dataset_balanced_sampling=False,
        training_seconds=120.0,
    )


def fake_selection(active: bool, savings: float):
    threshold = pd.DataFrame(
        [
            {
                "threshold": 0.9,
                "quality_retention_lcb": 0.995,
                "macro_dataset_quality_retention_lcb": 0.99,
                "quality_loss_rate_ucl": 0.01,
                "routed_safety_precision_lcb": 0.92,
                "guarded_dataset_quality_retention_lcb": 0.94,
                "safe_opportunity_recall": 0.15,
                "routed_fraction": 0.10,
                "resource_savings": savings + 0.01,
                "conservative_resource_savings": savings,
                "feasible_block_size": 2 if active else 0,
                "gate_pass_count": 7 if active else 6,
            }
        ]
    )
    return SimpleNamespace(
        threshold_search=threshold,
        diagnostic_threshold=0.9,
        selected_threshold=0.9 if active else 1.1,
        router_active=active,
        failure_reasons=() if active else ("failed",),
    )


def test_comparison_prefers_active_setup_then_conservative_savings():
    trainings = {
        "inactive_but_fast": fake_training(0.4),
        "active_small": fake_training(0.5),
        "active_large": fake_training(0.6),
    }
    selections = {
        "inactive_but_fast": fake_selection(False, 0.05),
        "active_small": fake_selection(True, 0.01),
        "active_large": fake_selection(True, 0.02),
    }

    comparison = build_setup_comparison(trainings, selections)

    assert choose_validation_setup(comparison) == "active_large"
    assert set(comparison.setup) == set(trainings)
    assert comparison.failure_reasons.notna().all()
