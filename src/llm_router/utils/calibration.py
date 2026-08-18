"""Per-candidate Platt scaling, including out-of-fold validation estimates."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


def fit_platt(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    prevalence = float(targets.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        clipped = np.clip(prevalence, 1e-4, 1 - 1e-4)
        return 0.0, float(np.log(clipped / (1 - clipped)))

    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    raw_slope = torch.nn.Parameter(torch.tensor(0.5413249))
    intercept = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS(
        [raw_slope, intercept],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        slope = F.softplus(raw_slope) + 1e-4
        loss = F.binary_cross_entropy_with_logits(slope * x + intercept, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        float((F.softplus(raw_slope) + 1e-4).detach()),
        float(intercept.detach()),
    )


def apply_platt(
    logits: np.ndarray, parameters: list[tuple[float, float]]
) -> np.ndarray:
    logits = np.asarray(logits)
    probabilities = np.zeros_like(logits, dtype=float)
    for column, (slope, intercept) in enumerate(parameters):
        calibrated = np.clip(slope * logits[:, column] + intercept, -40, 40)
        probabilities[:, column] = 1 / (1 + np.exp(-calibrated))
    return probabilities


def _expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(targets))
    error = 0.0
    for lower, upper in pairwise(edges):
        included = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if included.any():
            error += included.sum() / total * abs(
                probabilities[included].mean() - targets[included].mean()
            )
    return float(error)


def calibrate_with_validation(
    logits: np.ndarray,
    targets: np.ndarray,
    validation_rows: np.ndarray,
    candidate_names: tuple[str, ...],
    *,
    folds: int = 5,
    seed: int = 42,
) -> tuple[
    np.ndarray,
    dict[str, tuple[float, float]],
    list[dict[str, float | int | str]],
]:
    """Fit deployable Platt scalers and use OOF probabilities on validation.

    Final scalers are fitted on all validation rows and applied to non-validation
    rows. Validation rows receive out-of-fold probabilities, preventing each
    example from calibrating its own confidence before threshold selection.
    """

    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    validation_rows = np.asarray(validation_rows, dtype=int)
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("Calibration logits and targets must be matching matrices.")
    if logits.shape[1] != len(candidate_names):
        raise ValueError("Candidate names do not match calibration columns.")
    if len(validation_rows) == 0:
        raise ValueError("Calibration requires at least one validation example.")
    if (
        np.any(validation_rows < 0)
        or np.any(validation_rows >= len(logits))
        or len(np.unique(validation_rows)) != len(validation_rows)
    ):
        raise ValueError("Validation rows must be unique, in-range indices.")
    if not np.isfinite(logits).all():
        raise ValueError("Calibration logits must be finite.")

    parameters: dict[str, tuple[float, float]] = {}
    diagnostics: list[dict[str, float | int | str]] = []
    final_parameters = []
    validation_logits = logits[validation_rows]
    validation_targets = targets[validation_rows]
    if not np.isin(validation_targets, (0.0, 1.0)).all():
        raise ValueError("Calibration targets must be binary on validation rows.")

    for column, candidate in enumerate(candidate_names):
        target = validation_targets[:, column]
        fitted = fit_platt(validation_logits[:, column], target)
        parameters[candidate] = fitted
        final_parameters.append(fitted)

    probabilities = apply_platt(logits, final_parameters)
    for column, candidate in enumerate(candidate_names):
        target = validation_targets[:, column].astype(int)
        raw = 1 / (1 + np.exp(-np.clip(validation_logits[:, column], -40, 40)))
        calibrated = np.full(len(validation_rows), target.mean(), dtype=float)
        class_counts = np.bincount(target, minlength=2)
        fold_count = min(folds, int(class_counts.min()))
        if fold_count >= 2:
            splitter = StratifiedKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=seed + column,
            )
            for fit_rows, heldout_rows in splitter.split(validation_logits, target):
                fitted = fit_platt(
                    validation_logits[fit_rows, column], target[fit_rows]
                )
                calibrated[heldout_rows] = apply_platt(
                    validation_logits[heldout_rows, column, None], [fitted]
                )[:, 0]
        probabilities[validation_rows, column] = calibrated
        diagnostics.append(
            {
                "candidate": candidate,
                "validation_examples": len(target),
                "safe_prevalence": float(target.mean()),
                "raw_brier": float(np.mean((raw - target) ** 2)),
                "calibrated_brier": float(np.mean((calibrated - target) ** 2)),
                "raw_ece": _expected_calibration_error(raw, target),
                "calibrated_ece": _expected_calibration_error(calibrated, target),
            }
        )
    return probabilities, parameters, diagnostics


def out_of_fold_platt(
    logits: np.ndarray,
    targets: np.ndarray,
    candidate_names: tuple[str, ...],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, list[tuple[float, float]]]]:
    probabilities = np.zeros_like(logits, dtype=float)
    parameters_by_candidate = {}
    for column, candidate in enumerate(candidate_names):
        candidate_target = targets[:, column].astype(int)
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed + column
        )
        fold_parameters = []
        for fit_rows, heldout_rows in splitter.split(logits, candidate_target):
            parameters = fit_platt(logits[fit_rows, column], candidate_target[fit_rows])
            fold_parameters.append(parameters)
            probabilities[heldout_rows, column] = apply_platt(
                logits[heldout_rows, column, None], [parameters]
            )[:, 0]
        parameters_by_candidate[candidate] = fold_parameters
    return probabilities, parameters_by_candidate
