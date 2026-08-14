"""Per-candidate Platt scaling, including out-of-fold validation estimates."""

from __future__ import annotations

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
