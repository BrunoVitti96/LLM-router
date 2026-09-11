"""MATH score regression: 968 must be correct and 967 must remain wrong."""

import json
from importlib.metadata import PackageNotFoundError

import pytest

from llm_router import math_scoring
from llm_router.math_scoring import grade_math_response


@pytest.mark.parametrize(
    ("solution", "prediction", "score", "parsed"),
    [
        (
            r"The count is $1024-(1+10+45)=\boxed{968}$.",
            r"There are $1024-(1+10+45)=968$ possibilities. Thus $\boxed{968}$.",
            1.0,
            True,
        ),
        (r"The answer is $\boxed{968}$.", r"The answer is $\boxed{967}$.", 0.0, True),
        (r"$\boxed{\frac{1}{2}}$", r"$\boxed{\frac{2}{4}}$", 1.0, True),
        (r"$\boxed{968}$", "I cannot solve this problem.", 0.0, False),
        (r"$\boxed{968}$", "", 0.0, False),
    ],
)
def test_official_math_equivalence(solution, prediction, score, parsed):
    result = grade_math_response(solution, prediction)
    assert result["score"] == score
    assert result["prediction_parsed"] is parsed
    assert isinstance(json.loads(result["gold_extraction"]), list)
    assert isinstance(json.loads(result["prediction_extraction"]), list)


@pytest.mark.parametrize("gold", ["", "   ", None, "No reference answer provided"])
def test_unusable_gold_aborts_instead_of_becoming_zero(gold):
    with pytest.raises(ValueError, match="gold"):
        grade_math_response(gold, r"$\boxed{968}$")


def test_runtime_version_mismatch_aborts(monkeypatch):
    math_scoring.validate_math_scoring_runtime.cache_clear()
    monkeypatch.setattr(math_scoring, "version", lambda package: "0.0.0")
    try:
        with pytest.raises(RuntimeError, match="requires math-verify==0.5.2"):
            math_scoring.validate_math_scoring_runtime()
    finally:
        math_scoring.validate_math_scoring_runtime.cache_clear()


def test_missing_dependency_aborts(monkeypatch):
    def missing(package):
        raise PackageNotFoundError(package)

    math_scoring.validate_math_scoring_runtime.cache_clear()
    monkeypatch.setattr(math_scoring, "version", missing)
    try:
        with pytest.raises(RuntimeError, match="pinned notebook dependencies"):
            math_scoring.validate_math_scoring_runtime()
    finally:
        math_scoring.validate_math_scoring_runtime.cache_clear()


def test_unexpected_parser_failure_propagates(monkeypatch):
    import math_verify

    def fail(*args, **kwargs):
        raise RuntimeError("symbolic runtime failed")

    monkeypatch.setattr(math_verify, "parse", fail)
    with pytest.raises(RuntimeError, match="symbolic runtime failed"):
        math_scoring._grade_math_response_local(r"$\boxed{968}$", "968")


def test_fallback_string_alone_is_not_usable_gold(monkeypatch):
    import math_verify

    monkeypatch.setattr(math_verify, "parse", lambda *args, **kwargs: ["unparsed"])
    with pytest.raises(ValueError, match="no usable parsed"):
        math_scoring._grade_math_response_local("broken gold", "unparsed")


def test_unexpected_verifier_failure_propagates(monkeypatch):
    import math_verify
    from sympy import Integer

    def fail(*args, **kwargs):
        raise RuntimeError("symbolic comparison failed")

    monkeypatch.setattr(math_verify, "parse", lambda *args, **kwargs: [Integer(968)])
    monkeypatch.setattr(math_verify, "verify", fail)
    with pytest.raises(RuntimeError, match="symbolic comparison failed"):
        math_scoring._grade_math_response_local(r"$\boxed{968}$", "968")


def test_spawn_worker_reuses_process_without_mutating_parent_timeouts():
    from math_verify import grader, parser

    original_timeouts = (parser.timeout, grader.timeout)
    worker = math_scoring._MathGradingWorker()
    try:
        assert worker.grade(r"$\boxed{968}$", "968")["score"] == 1.0
        pid = worker._process.pid
        assert worker.grade(r"$\boxed{\frac{1}{2}}$", "0.5")["score"] == 1.0
        assert worker._process.pid == pid
        assert (parser.timeout, grader.timeout) == original_timeouts
    finally:
        worker.close()
    assert worker._process is None
    assert worker._connection is None


def test_spawn_worker_gold_error_reaches_caller_and_can_restart():
    worker = math_scoring._MathGradingWorker()
    try:
        with pytest.raises(ValueError, match="no usable parsed"):
            worker.grade("No reference answer provided", "968")
        assert worker._process is None
        assert worker.grade(r"$\boxed{968}$", "968")["score"] == 1.0
    finally:
        worker.close()


def test_spawn_worker_watchdog_aborts_and_cleans_up():
    # A zero deadline deterministically expires before a fresh spawned Python
    # imports the symbolic stack; this exercises the real process cleanup path.
    worker = math_scoring._MathGradingWorker(timeout_seconds=0.0)
    try:
        with pytest.raises(TimeoutError, match="no score was recorded"):
            worker.grade(r"$\boxed{968}$", "968")
        assert worker._process is None
        assert worker._connection is None
    finally:
        worker.close()
