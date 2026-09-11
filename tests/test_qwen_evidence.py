import pandas as pd
import pytest

from llm_router.qwen_evidence import (
    QWEN_TIER_PARAMETERS_BILLIONS,
    SCORING_CONTRACT_VERSION,
    audit_aligned_outcomes,
    published_binary_score,
    reconcile_published_results,
    score_published_sample,
    scoring_contract,
    validate_qwen_tier_contract,
)

MODELS = tuple(QWEN_TIER_PARAMETERS_BILLIONS)


def _records(scores=(1.0, 0.0, 1.0)):
    return pd.DataFrame(
        [
            {
                "key": "task::0",
                "task": "leaderboard_task",
                "model": model,
                "prompt": "What is 2 + 2?",
                "doc_hash": "same-document",
                "published_metric": "acc,none",
                "score": score,
            }
            for model, score in zip(MODELS, scores, strict=True)
        ]
    )


def test_published_binary_score_prefers_named_correctness_metric():
    metric, score = published_binary_score(
        {"doc_id": 7, "acc,none": 1.0, "other_number": 0.5}
    )
    assert metric == "acc,none"
    assert score == 1.0


def test_published_binary_score_prefers_ifeval_prompt_level_strict_metric():
    metric, score = published_binary_score(
        {
            "doc_id": 7,
            "inst_level_loose_acc": 1.0,
            "inst_level_strict_acc": 1.0,
            "prompt_level_loose_acc": 1.0,
            "prompt_level_strict_acc": 0.0,
        }
    )
    assert metric == "prompt_level_strict_acc"
    assert score == 0.0


def test_published_binary_score_rejects_fractional_or_ambiguous_metrics():
    with pytest.raises(ValueError, match="binary 0/1"):
        published_binary_score({"exact_match,none": 0.5})
    with pytest.raises(ValueError, match="disagree"):
        published_binary_score({"pass@1": 1.0, "custom_acc": 0.0})


def test_qwen_contract_requires_small_middle_and_roughly_8b_tier():
    candidates = {
        name: {"parameters_billions": parameters}
        for name, parameters in QWEN_TIER_PARAMETERS_BILLIONS.items()
    }
    validate_qwen_tier_contract(candidates)
    candidates["Qwen2.5-7B"]["parameters_billions"] = 7.0
    with pytest.raises(ValueError, match="7.61B"):
        validate_qwen_tier_contract(candidates)


def test_aligned_outcomes_make_correctness_and_quality_explicit():
    records, quality_audit = audit_aligned_outcomes(_records(), MODELS)
    assert records.is_correct.tolist() == [True, False, True]
    assert quality_audit.correct.sum() == 2
    assert quality_audit.incorrect.sum() == 1
    assert quality_audit.quality.tolist() == [1.0, 0.0, 1.0]


def test_aligned_outcomes_fail_on_fractional_quality_or_missing_model():
    with pytest.raises(ValueError, match="binary 0/1"):
        audit_aligned_outcomes(_records(scores=(1.0, 0.5, 1.0)), MODELS)
    with pytest.raises(ValueError, match="all three models"):
        audit_aligned_outcomes(_records().iloc[:-1], MODELS)


def test_math_format_error_is_regraded_and_original_is_preserved():
    """A correct boxed 968 must not inherit the obsolete format score zero."""
    result = score_published_sample(
        {
            "doc": {"solution": r"Subtract: $1024-56=\boxed{968}$."},
            "target": "968",
            "filtered_resps": [r"The answer is \(\boxed{968}\)."],
            "exact_match": 0,
        },
        "leaderboard_math_counting_and_prob_hard",
    )
    assert result["published_score"] == 0
    assert result["score"] == 1
    assert result["score_source"] == "math_verify_0.5.2"
    assert result["scoring_version"] == SCORING_CONTRACT_VERSION
    assert result["gold_extraction"]


def test_nonmath_uses_published_quality_without_response_regrading():
    result = score_published_sample(
        {"acc_norm,none": 0, "resps": [[[-0.3, False]]]},
        "leaderboard_bbh_boolean_expressions",
    )
    assert result["score"] == result["published_score"] == 0
    assert result["score_source"] == "published_binary_metric"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"target": "968", "resps": [["968"]]}, "original doc.solution"),
        (
            {"doc": {"solution": r"\boxed{968}"}, "resps": [["968", "2"]]},
            "exactly one string",
        ),
    ],
)
def test_math_regrading_refuses_missing_gold_or_ambiguous_generation(payload, message):
    with pytest.raises(ValueError, match=message):
        score_published_sample(
            {"exact_match": 0, **payload}, "leaderboard_math_algebra_hard"
        )


def _full_audit_fixture(task="leaderboard_bbh_demo"):
    is_math = task.startswith("leaderboard_math_")
    metric = "exact_match" if is_math else "acc_norm"
    rows = pd.DataFrame([
        {
            "key": f"{task}::{i}", "task": task, "model": "test-model",
            "doc_id": str(i), "score": score,
            "published_score": 0 if is_math else score,
            "published_metric": metric,
            "score_source": "math_verify_0.5.2" if is_math else "published_binary_metric",
            "scoring_version": SCORING_CONTRACT_VERSION,
        }
        for i, score in enumerate([1, 0])
    ])
    metadata = {
        "test-model": {
            "n-samples": {task: {"effective": 2, "original": 2}},
            "results": {task: {f"{metric},none": 0.5}},
            # Stale group totals must not be mistaken for corrected leaf data.
            "groups": {task: {f"{metric},none": 0.0}},
        }
    }
    return rows, metadata


def test_full_task_reconciliation_uses_leaf_results_and_records_label_changes():
    rows, metadata = _full_audit_fixture("leaderboard_math_demo")
    audit = reconcile_published_results(rows, metadata)
    assert audit.passed.all()
    assert audit.changed_labels.tolist() == [1]
    assert audit.published_detail_mean.tolist() == [0]
    assert audit.rescored_mean.tolist() == [0.5]


def test_math_aggregate_difference_is_explicit_without_imputing_labels():
    rows, metadata = _full_audit_fixture("leaderboard_math_demo")
    metadata["test-model"]["results"]["leaderboard_math_demo"]["exact_match,none"] = 1
    audit = reconcile_published_results(rows, metadata)
    assert audit.passed.all()  # Counts and local scorer provenance are valid.
    assert not audit.mean_matches.any()
    assert not audit.aggregate_required.any()
    assert audit.aggregate_status.tolist() == ["math_reference_differs"]
    assert rows.score.tolist() == [1, 0]  # Never fabricate a second correct answer.


def test_reconciliation_rejects_sampling_and_retains_failure_diagnostics():
    rows, metadata = _full_audit_fixture()
    with pytest.raises(ValueError, match="before sampling/training") as error:
        reconcile_published_results(rows.iloc[:1], metadata)
    assert not error.value.reconciliation_audit.count_matches.all()


def test_reconciliation_rejects_wrong_scores_duplicates_and_old_contract():
    rows, metadata = _full_audit_fixture()
    wrong = rows.assign(score=0, published_score=0)
    with pytest.raises(ValueError, match="rescored=0.00000000"):
        reconcile_published_results(wrong, metadata)
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile_published_results(pd.concat([rows, rows]), metadata)
    with pytest.raises(ValueError, match="obsolete scoring versions"):
        reconcile_published_results(rows.assign(scoring_version="v1"), metadata)


def test_reconciliation_rejects_missing_tasks_and_metric_mismatch():
    rows, metadata = _full_audit_fixture()
    metadata["test-model"]["n-samples"]["missing_task"] = {"effective": 1}
    with pytest.raises(ValueError, match="Missing or unexpected complete tasks"):
        reconcile_published_results(rows, metadata)
    rows, metadata = _full_audit_fixture()
    rows.loc[0, "published_metric"] = "acc"
    with pytest.raises(ValueError, match="Mixed metric definitions"):
        reconcile_published_results(rows, metadata)


def test_scoring_contract_freezes_grader_and_changes_evidence_identity():
    contract = scoring_contract()
    assert contract["version"] == SCORING_CONTRACT_VERSION
    assert contract["math_verify_version"] == "0.5.2"
    assert contract["sympy_version"] == "1.13.3"
