import pandas as pd
import pytest

from llm_router.qwen_evidence import (
    QWEN_TIER_PARAMETERS_BILLIONS,
    audit_aligned_outcomes,
    published_binary_score,
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
