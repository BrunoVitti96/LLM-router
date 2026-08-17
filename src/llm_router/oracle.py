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


def replacement_safety_targets(
    quality: np.ndarray,
    fallback_index: int,
    nonfallback_indices: np.ndarray,
    quality_epsilon: float = 0.0,
) -> np.ndarray:
    """Return whether each alternative preserves fallback-relative quality."""

    quality = np.asarray(quality, dtype=float)
    nonfallback_indices = np.asarray(nonfallback_indices, dtype=int)
    if quality.ndim != 2:
        raise ValueError("Quality must be a [prompt, model] array.")
    fallback_quality = quality[:, fallback_index, None]
    return quality[:, nonfallback_indices] >= fallback_quality - quality_epsilon


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


def hybrid_routing_loss(
    safety_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    quality: torch.Tensor,
    analytical_latency: torch.Tensor,
    fallback_index: int,
    nonfallback_indices: torch.Tensor,
    *,
    quality_epsilon: float = 0.0,
    safety_loss_weight: float = 1.0,
    oracle_auxiliary_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize deployable safety estimates plus auxiliary oracle imitation.

    The safety head is the only head used by the deployed selector.  The oracle
    head regularizes the shared ModernBERT representation toward the ideal
    latency-aware decision, without entangling deployment with a fixed hardware
    scenario or candidate ranking.
    """

    if safety_logits.shape != (
        len(quality),
        len(nonfallback_indices),
    ):
        raise ValueError("Safety logits do not match the non-fallback model count.")
    fallback_quality = quality[:, fallback_index : fallback_index + 1]
    safety_target = (
        quality[:, nonfallback_indices] >= fallback_quality - quality_epsilon
    ).float()
    safety_loss = F.binary_cross_entropy_with_logits(
        safety_logits.float(), safety_target
    )
    oracle_loss, oracle_parts = oracle_routing_loss(
        oracle_logits,
        quality,
        analytical_latency,
        fallback_index,
        quality_epsilon=quality_epsilon,
    )
    total = (
        safety_loss_weight * safety_loss
        + oracle_auxiliary_weight * oracle_loss
    )
    return total, {
        "safety": safety_loss,
        "oracle_auxiliary": oracle_loss,
        **{f"oracle_{name}": value for name, value in oracle_parts.items()},
    }
