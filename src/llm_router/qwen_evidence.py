"""Validation helpers for precomputed three-tier Qwen quality evidence.

The canonical Colab workflow downloads published per-example evaluation rows;
it never loads or runs a Qwen causal language model.  This module turns each
published binary benchmark metric into an explicit ``is_correct`` field and
fails closed when the three-candidate panel is incomplete or ambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

QWEN_TIER_PARAMETERS_BILLIONS = {
    "Qwen2.5-1.5B": 1.54,
    "Qwen2.5-3B": 3.09,
    "Qwen2.5-7B": 7.61,
}

BINARY_METRIC_PRIORITY = (
    "prompt_level_strict_acc,none",
    "exact_match,none",
    "acc_norm,none",
    "acc,none",
)

REQUIRED_OUTCOME_COLUMNS = {
    "key",
    "task",
    "model",
    "prompt",
    "doc_hash",
    "published_metric",
    "score",
}


def _scalar_metric(value: object) -> float | None:
    """Return one finite scalar without accepting arbitrary containers."""

    if isinstance(value, (bool, int, float, np.number)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, list) and len(value) == 1:
        return _scalar_metric(value[0])
    return None


def _binary_value(metric_name: str, value: object) -> float:
    """Validate that a published per-example metric means wrong (0) or right (1)."""

    numeric = _scalar_metric(value)
    if numeric is None or numeric not in {0.0, 1.0}:
        raise ValueError(
            f"Published metric {metric_name!r} must be binary 0/1; got {value!r}."
        )
    return numeric


def published_binary_score(payload: Mapping[str, object]) -> tuple[str, float]:
    """Select one published correctness metric and require an exact 0/1 value.

    Named metrics are preferred in a fixed order.  The conservative fallback is
    limited to scalar keys whose names clearly describe accuracy, exact match,
    or pass status.  If several fallback metrics exist, they must agree on the
    per-example correctness result; otherwise the row is rejected as ambiguous.
    """

    for metric_name in BINARY_METRIC_PRIORITY:
        if metric_name in payload:
            return metric_name, _binary_value(metric_name, payload[metric_name])

    candidates: list[tuple[str, float]] = []
    for key, value in payload.items():
        normalized = key.lower()
        if key == "doc_id" or not any(
            token in normalized for token in ("acc", "exact", "pass")
        ):
            continue
        numeric = _scalar_metric(value)
        if numeric is not None:
            candidates.append((key, _binary_value(key, value)))

    if not candidates:
        raise ValueError(
            f"No supported binary correctness metric found in keys: {sorted(payload)}"
        )
    if len({score for _, score in candidates}) != 1:
        names = ", ".join(name for name, _ in sorted(candidates))
        raise ValueError(f"Published correctness metrics disagree: {names}.")
    return min(candidates)


def validate_qwen_tier_contract(
    candidates: Mapping[str, Mapping[str, object]],
) -> None:
    """Require the exact small, middle, and roughly-8B Qwen2.5 panel."""

    expected_names = tuple(QWEN_TIER_PARAMETERS_BILLIONS)
    if tuple(candidates) != expected_names:
        raise ValueError(
            "Candidates must be ordered as the three frozen Qwen2.5 tiers: "
            f"{expected_names}; got {tuple(candidates)}."
        )
    for name, expected_parameters in QWEN_TIER_PARAMETERS_BILLIONS.items():
        actual = float(candidates[name].get("parameters_billions", np.nan))
        if not np.isclose(actual, expected_parameters):
            raise ValueError(
                f"{name} must declare {expected_parameters:.2f}B parameters; "
                f"got {actual!r}."
            )


def audit_aligned_outcomes(
    records: pd.DataFrame,
    candidate_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate an aligned panel and return rows plus a correctness audit.

    ``score`` is kept for the generic routing pipeline.  ``is_correct`` is the
    same published 0/1 result expressed as a boolean, making the quality
    calculation explicit: quality is ``correct / outcomes`` for each group.
    """

    missing = REQUIRED_OUTCOME_COLUMNS - set(records)
    if missing:
        raise ValueError(f"Published Qwen outcomes are missing columns: {sorted(missing)}")

    expected_models = tuple(candidate_names)
    if len(expected_models) != 3 or len(set(expected_models)) != 3:
        raise ValueError("The published panel must contain exactly three unique models.")
    actual_models = set(records.model.dropna())
    if actual_models != set(expected_models):
        raise ValueError(
            "Every prompt panel must include all three models. "
            f"Expected {expected_models}; found {tuple(sorted(actual_models))}."
        )
    if records.empty:
        raise ValueError("The published Qwen outcome panel is empty.")
    if records[["key", "task", "model", "prompt", "doc_hash"]].isna().any().any():
        raise ValueError("Published identity, model, prompt, and document hash cannot be null.")
    if records.prompt.astype(str).str.strip().eq("").any():
        raise ValueError("Published prompts cannot be empty.")

    duplicates = records.duplicated(["key", "model"], keep=False)
    if duplicates.any():
        raise ValueError("Every prompt/model pair must occur exactly once.")
    coverage = records.groupby("key").model.agg(lambda values: frozenset(values))
    incomplete = coverage.ne(frozenset(expected_models))
    if incomplete.any():
        raise ValueError(
            f"{int(incomplete.sum())} prompts do not have one outcome for all three models."
        )
    for column in ("task", "doc_hash", "prompt", "published_metric"):
        inconsistent = records.groupby("key")[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(
                f"{int(inconsistent.sum())} aligned prompts disagree on {column}."
            )

    audited = records.copy()
    audited["score"] = pd.to_numeric(audited.score, errors="raise")
    if not np.isfinite(audited.score.to_numpy(float)).all():
        raise ValueError("Published correctness scores must be finite.")
    values = set(audited.score.astype(float).unique())
    if not values <= {0.0, 1.0}:
        raise ValueError(f"Published correctness scores must be binary 0/1; found {values}.")
    audited["is_correct"] = audited.score.astype(bool)

    group_columns = ["task", "model", "published_metric"]
    quality_audit = (
        audited.groupby(group_columns, sort=True, observed=True)
        .agg(
            outcomes=("is_correct", "size"),
            correct=("is_correct", "sum"),
        )
        .reset_index()
    )
    quality_audit["correct"] = quality_audit.correct.astype(int)
    quality_audit["incorrect"] = quality_audit.outcomes - quality_audit.correct
    quality_audit["quality"] = quality_audit.correct / quality_audit.outcomes
    return audited, quality_audit
