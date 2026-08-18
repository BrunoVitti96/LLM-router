import numpy as np

from llm_router.utils.calibration import (
    apply_platt,
    calibrate_with_validation,
    fit_platt,
)


def test_constant_platt_target_returns_prevalence():
    parameters = fit_platt(np.array([-2.0, 2.0]), np.array([1.0, 1.0]))
    probabilities = apply_platt(np.array([[-100.0], [100.0]]), [parameters])
    assert parameters[0] == 0.0
    assert np.allclose(probabilities, 0.9999)


def test_apply_platt_is_monotonic():
    probabilities = apply_platt(np.array([[-2.0], [0.0], [2.0]]), [(1.0, 0.0)])[:, 0]
    assert np.all(np.diff(probabilities) > 0)


def test_validation_calibration_is_oof_and_never_reads_test_targets():
    logits = np.linspace(-3, 3, 12)[:, None]
    validation = np.arange(2, 10)
    targets = np.zeros_like(logits)
    targets[validation, 0] = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    changed_test_targets = targets.copy()
    changed_test_targets[[0, 1, 10, 11], 0] = 1

    first = calibrate_with_validation(
        logits, targets, validation, ("candidate",), folds=4, seed=42
    )
    second = calibrate_with_validation(
        logits,
        changed_test_targets,
        validation,
        ("candidate",),
        folds=4,
        seed=42,
    )

    assert np.allclose(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert np.all((first[0] >= 0) & (first[0] <= 1))
    assert first[2][0]["candidate"] == "candidate"
