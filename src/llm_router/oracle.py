"""Outcome-aware oracle labels and the differentiable oracle routing loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def oracle_choices(
    quality: np.ndarray,
    analytical_latency: np.ndarray,
    fallback_index: int,
    quality_epsilon: float = 0.0,
) -> np.ndarray:
    """Choose the lowest-latency model that matches fallback quality per prompt."""

    quality = np.asarray(quality, dtype=float)
    analytical_latency = np.asarray(analytical_latency, dtype=float)
    if quality.shape != analytical_latency.shape or quality.ndim != 2:
        raise ValueError("Quality and latency must be matching [prompt, model] arrays.")
    fallback_quality = quality[:, fallback_index, None]
    eligible = quality >= fallback_quality - quality_epsilon
    eligible[:, fallback_index] = True
    return np.where(eligible, analytical_latency, np.inf).argmin(axis=1)


def oracle_routing_loss(
    logits: torch.Tensor,
    quality: torch.Tensor,
    analytical_latency: torch.Tensor,
    fallback_index: int,
    *,
    quality_epsilon: float = 0.0,
    opportunity_weight: float = 1.0,
    quality_risk_weight: float = 4.0,
    latency_regret_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train ModernBERT to imitate the best quality-preserving latency decision.

    Cross entropy learns the hindsight oracle class.  Two expected-risk terms
    make the loss decision-aware: probability assigned to a quality-losing model
    is expensive, while probability assigned to a model slower than the oracle
    incurs normalized latency regret.
    """

    if logits.shape != quality.shape or logits.shape != analytical_latency.shape:
        raise ValueError("Logits, quality, and latency must have identical shapes.")
    with torch.no_grad():
        fallback_quality = quality[:, fallback_index : fallback_index + 1]
        eligible = quality >= fallback_quality - quality_epsilon
        eligible[:, fallback_index] = True
        oracle = torch.where(
            eligible,
            analytical_latency,
            torch.full_like(analytical_latency, torch.inf),
        ).argmin(dim=1)
        rows = torch.arange(len(logits), device=logits.device)
        fallback_latency = analytical_latency[:, fallback_index].clamp_min(1e-9)
        oracle_latency = analytical_latency[rows, oracle]
        opportunity = ((fallback_latency - oracle_latency) / fallback_latency).clamp_min(0)
        quality_drop = (fallback_quality - quality - quality_epsilon).clamp_min(0)
        latency_regret = (
            (analytical_latency - oracle_latency[:, None]) / fallback_latency[:, None]
        ).clamp_min(0)

    imitation = (
        F.cross_entropy(logits.float(), oracle, reduction="none")
        * (1.0 + opportunity_weight * opportunity)
    ).mean()
    probability = logits.float().softmax(dim=1)
    quality_risk = (probability * quality_drop).sum(dim=1).mean()
    latency_regret_loss = (probability * latency_regret).sum(dim=1).mean()
    total = (
        imitation
        + quality_risk_weight * quality_risk
        + latency_regret_weight * latency_regret_loss
    )
    return total, {
        "imitation": imitation,
        "quality_risk": quality_risk,
        "latency_regret": latency_regret_loss,
    }
