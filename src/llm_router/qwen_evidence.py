"""Validation helpers for precomputed three-tier Qwen quality evidence.

The canonical Colab workflow downloads published per-example evaluation rows;
it never generates new candidate answers. Historical detail rows use a strict
math answer format that predates the leaderboard's Math-Verify correction.
Scoring v2 regrades math responses with the pinned verifier, preserves the
original score, and audits complete tasks before sampling. For example, a
boxed answer 968 matching gold 968 changes from format score 0 to correctness 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

SCORING_CONTRACT_VERSION = "qwen-math-verify-v2"
MATH_TASK_PREFIX = "leaderboard_math_"
EXCLUDED_OVERLAPPING_TASKS = frozenset(
    {"leaderboard_gpqa_main", "leaderboard_gpqa_extended"}
)


def scoring_contract() -> dict[str, object]:
    """Describe the immutable grading policy included in the evidence hash.

    A label change is a new experiment, even when model weights and prompts
    stay the same. Aggregate checks use full task populations, never the
    subsequent 300-example cap.
    """

    return {
        "version": SCORING_CONTRACT_VERSION,
        "math_method": "Math-Verify on original solution and candidate response",
        "math_verify_version": "0.5.2",
        "latex2sympy2_extended_version": "1.0.6",
        "antlr4_python3_runtime_version": "4.13.2",
        "sympy_version": "1.13.3",
        "reference_implementation": (
            "https://github.com/huggingface/lm-evaluation-harness/blob/"
            "c2a084962aac2dd0f0813791c57c275b27573b35/"
            "lm_eval/tasks/leaderboard/math/utils.py"
        ),
        "nonmath_method": "unchanged published task-aware binary metric",
        "metric_priority": list(BINARY_METRIC_PRIORITY),
        "required_values": [0, 1],
        "quality_formula": "correct / outcomes",
        "preserve_original_scores": True,
        "reconciliation": (
            "full-task counts and nonmath leaf means required before alignment/cap; "
            "math leaf means diagnostic because upstream rescoring is not pinned"
        ),
        "math_score_authority": "local pinned Math-Verify, not aggregate-imputed labels",
        "aggregate_absolute_tolerance": 1e-8,
    }

QWEN_TIER_PARAMETERS_BILLIONS = {
    "Qwen2.5-1.5B": 1.54,
    "Qwen2.5-3B": 3.09,
    "Qwen2.5-7B": 7.61,
}

BINARY_METRIC_PRIORITY = (
    "prompt_level_strict_acc,none",
    "prompt_level_strict_acc",
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


def _single_math_response(payload: Mapping[str, object]) -> str:
    """Read the one scored generation; refuse to guess among multiple answers."""

    value = payload.get("filtered_resps")
    if value is None or value == []:
        value = payload.get("resps")
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - malformed evidence shape
            "Math evidence must contain exactly one string response."
        )
    return value


def score_published_sample(
    payload: Mapping[str, object], task: str
) -> dict[str, object]:
    """Preserve a source score and compute the versioned training score.

    Only MATH-Hard changes. Its original format score is retained separately;
    the official Math-Verify procedure grades the full gold solution against
    the recorded response. If gold 968 and a boxed 968 were originally scored
    zero, the returned ``published_score`` is 0 and ``score`` is 1. We never use
    the model's response as a gold answer or synthesize labels from aggregates.
    """

    metric, published_score = published_binary_score(payload)
    result: dict[str, object] = {
        "published_metric": metric,
        "published_score": published_score,
        "score": published_score,
        "score_source": "published_binary_metric",
        "scoring_version": SCORING_CONTRACT_VERSION,
        "gold_extraction": "",
        "prediction_extraction": "",
        "prediction_parsed": None,
    }
    if not task.startswith(MATH_TASK_PREFIX):
        return result
    if metric.removesuffix(",none") != "exact_match":
        raise ValueError(f"Unexpected published math metric: {metric!r}.")
    doc = payload.get("doc")
    solution = doc.get("solution") if isinstance(doc, Mapping) else None
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("Math evidence requires the original doc.solution as gold.")
    from .math_scoring import grade_math_response

    result.update(grade_math_response(solution, _single_math_response(payload)))
    result["score_source"] = "math_verify_0.5.2"
    return result


def reconcile_published_results(
    records: pd.DataFrame,
    published_run_metadata: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Audit complete tasks before training under the declared scoring contract.

    Leaf ``results`` contain the corrected math aggregates; the historical
    ``groups`` section can still contain obsolete format scores. Source math
    aggregates lack a pinned rescoring environment and do not all reproduce:
    local 7B counting yields 55/123 versus the source's 57/123. These differences
    are exposed, never repaired by assigning labels to match a total. Local
    pinned math scores are authoritative for v2; all counts and nonmath means
    must match exactly. A capped panel cannot pass by coincidence.
    """

    required = {
        "key", "task", "model", "doc_id", "score", "published_score",
        "published_metric", "score_source", "scoring_version",
    }
    missing = required - set(records)
    if missing:
        raise ValueError(f"Reconciliation columns missing: {sorted(missing)}")
    if records.empty or records[list(required)].isna().any().any():
        raise ValueError("Full evidence cannot be empty or contain null audit fields.")
    if set(records.model) != set(published_run_metadata):
        raise ValueError("Evidence models do not match published metadata models.")
    if records.duplicated(["model", "task", "doc_id"]).any():
        raise ValueError("Duplicate documents in full task evidence.")
    if records.duplicated(["model", "key"]).any():
        raise ValueError("Duplicate prompt/model pairs in full evidence.")
    if not records.scoring_version.eq(SCORING_CONTRACT_VERSION).all():
        raise ValueError("Mixed or obsolete scoring versions; rebuild all labels.")
    for column in ("score", "published_score"):
        values = pd.to_numeric(records[column], errors="raise")
        if not np.isfinite(values).all() or not values.isin([0, 1]).all():
            raise ValueError(f"{column} must contain only finite binary 0/1 values.")

    rows = []
    errors = []
    for model, metadata in published_run_metadata.items():
        counts = metadata.get("n-samples")
        results = metadata.get("results")
        if not isinstance(counts, Mapping) or not isinstance(results, Mapping):
            raise ValueError(  # noqa: TRY004 - missing evidence fields
                f"Missing full-task counts or leaf results for {model}."
            )
        expected_tasks = set(counts) - EXCLUDED_OVERLAPPING_TASKS
        model_rows = records.loc[records.model.eq(model)]
        if set(model_rows.task) != expected_tasks:
            raise ValueError(f"Missing or unexpected complete tasks for {model}.")
        for task, group in model_rows.groupby("task", sort=True):
            if group.published_metric.nunique() != 1:
                raise ValueError(f"Mixed metric definitions for {model}/{task}.")
            task_counts = counts.get(task)
            task_results = results.get(task)
            if not isinstance(task_counts, Mapping) or not isinstance(
                task_results, Mapping
            ):
                raise ValueError(  # noqa: TRY004 - missing evidence fields
                    f"Missing metadata for {model}/{task}."
                )
            expected_n = task_counts.get("effective")
            if (
                not isinstance(expected_n, int)
                or isinstance(expected_n, bool)
                or expected_n <= 0
            ):
                raise ValueError(f"Invalid effective sample count for {model}/{task}.")
            metric = str(group.published_metric.iloc[0]).removesuffix(",none")
            expected_mean = _scalar_metric(task_results.get(f"{metric},none"))
            if expected_mean is None or not 0 <= expected_mean <= 1:
                raise ValueError(f"Missing binary mean {metric} for {model}/{task}.")
            is_math = task.startswith(MATH_TASK_PREFIX)
            expected_source = (
                "math_verify_0.5.2" if is_math else "published_binary_metric"
            )
            if not group.score_source.eq(expected_source).all():
                raise ValueError(f"Unexpected scorer for {model}/{task}.")
            if not is_math and not group.score.eq(group.published_score).all():
                raise ValueError(f"Non-math scores changed for {model}/{task}.")
            actual_mean = float(group.score.mean())
            count_matches = len(group) == expected_n
            mean_matches = bool(
                np.isclose(actual_mean, expected_mean, atol=1e-8, rtol=0)
            )
            aggregate_required = not is_math
            passed = count_matches and (mean_matches or not aggregate_required)
            rows.append({
                "model": model,
                "task": task,
                "published_metric": metric,
                "scoring_version": SCORING_CONTRACT_VERSION,
                "score_source": expected_source,
                "outcomes": len(group),
                "expected_outcomes": expected_n,
                "published_detail_correct": int(group.published_score.sum()),
                "rescored_correct": int(group.score.sum()),
                "changed_labels": int(group.score.ne(group.published_score).sum()),
                "published_detail_mean": float(group.published_score.mean()),
                "rescored_mean": actual_mean,
                "published_leaf_mean": expected_mean,
                "absolute_difference": abs(actual_mean - expected_mean),
                "count_matches": count_matches,
                "mean_matches": mean_matches,
                "aggregate_required": aggregate_required,
                "aggregate_status": (
                    "matches" if mean_matches else "math_reference_differs"
                    if is_math else "nonmath_mismatch"
                ),
                "passed": passed,
            })
            if not passed:
                errors.append(
                    f"{model}/{task}: {len(group)}/{expected_n} rows, "
                    f"rescored={actual_mean:.8f}, published={expected_mean:.8f}"
                )
    audit = pd.DataFrame(rows)
    if errors:
        error = ValueError(
            "Published evidence reconciliation failed before sampling/training. "
            "Repair task coverage/nonmath provenance; do not reuse old math labels. "
            + "; ".join(errors)
        )
        error.reconciliation_audit = audit
        raise error
    return audit


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

    ``score`` is the current contract's 0/1 outcome (locally regraded on MATH
    in v2). ``is_correct`` expresses that score as a boolean. Thus two correct
    answers among four outcomes yield quality ``2 / 4 = 0.5``; the original
    format-based score remains separately available as ``published_score``.
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
