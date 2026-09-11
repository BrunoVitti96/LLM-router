"""Regrade saved MATH responses using the pinned Hugging Face grader.

An answer ending in ``\\boxed{968}`` should score 1 against a solution ending
in ``\\boxed{968}``, even when the published row incorrectly records 0. This
module applies the leaderboard's mathematical-equivalence rule, rather than
comparing answer strings or generating new candidate answers.

The extraction/verification calls match ``process_results`` in:
https://github.com/huggingface/lm-evaluation-harness/blob/
c2a084962aac2dd0f0813791c57c275b27573b35/lm_eval/tasks/leaderboard/math/utils.py

Math-Verify 0.5.2's Windows timeout starts an unpicklable nested process target.
On Windows (and non-main Python threads), a persistent spawned process instead
owns all symbolic work. Only that isolated process disables upstream timeout
decorators; a parent watchdog terminates it after 30 seconds. Linux/Colab main
threads retain the upstream signal-based timeouts. No installed files change.
"""

from __future__ import annotations

import atexit
import json
import multiprocessing
import os
import threading
import traceback
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import Connection
from typing import Any

PINNED_MATH_SCORING_VERSIONS = {
    "math-verify": "0.5.2",
    "latex2sympy2_extended": "1.0.6",
    "antlr4-python3-runtime": "4.13.2",
    "sympy": "1.13.3",
}
MATH_WORKER_TIMEOUT_SECONDS = 30.0


@lru_cache(maxsize=1)
def validate_math_scoring_runtime() -> None:
    """Fail before grading if installed symbolic-grader versions differ."""

    for package, expected in PINNED_MATH_SCORING_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"MATH scoring requires {package}=={expected}; install the "
                "pinned notebook dependencies before loading quality evidence."
            ) from exc
        if actual != expected:
            raise RuntimeError(
                f"MATH scoring requires {package}=={expected}, found {actual}. "
                "Install the pinned notebook dependencies and restart Python."
            )


def _grade_math_response_local(solution: str, prediction: str) -> dict[str, Any]:
    """Use upstream extraction defaults; never turn a grading error into 0."""

    from math_verify import LatexExtractionConfig, parse, verify
    from sympy import Basic, MatrixBase

    gold = parse(solution, extraction_config=[LatexExtractionConfig()])
    # A fallback string alone is not proof that the reference was parsed. Abort
    # rather than treating a broken reference as a wrong candidate answer.
    if not gold or not any(isinstance(item, (Basic, MatrixBase)) for item in gold):
        raise ValueError(
            "MATH gold solution has no usable parsed mathematical answer; "
            "quality evidence must be repaired before training."
        )
    predicted = parse(prediction)
    score = float(verify(gold, predicted))
    return {
        "score": score,
        "gold_extraction": json.dumps([str(item) for item in gold]),
        "prediction_extraction": json.dumps([str(item) for item in predicted]),
        "prediction_parsed": any(
            isinstance(item, (Basic, MatrixBase)) for item in predicted
        ),
    }


def _identity_timeout(timeout_seconds: int = 10):
    """Worker-only replacement: the parent process owns the hard deadline."""

    def decorator(function):
        return function

    return decorator


def _math_worker_main(connection: Connection) -> None:
    """Top-level target is spawn-picklable; send only primitive audit values."""

    try:
        validate_math_scoring_runtime()
        from math_verify import grader, parser

        parser.timeout = _identity_timeout
        grader.timeout = _identity_timeout
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                result = _grade_math_response_local(*request)
            except Exception as exc:  # noqa: BLE001 - re-raised by the parent
                connection.send(
                    ("error", type(exc).__name__, str(exc), traceback.format_exc())
                )
            else:
                connection.send(("ok", result))
    except (EOFError, BrokenPipeError):
        return
    except Exception as exc:  # noqa: BLE001 - re-raised by the parent
        # Initialization failures must reach the caller, never become scores.
        connection.send(
            ("error", type(exc).__name__, str(exc), traceback.format_exc())
        )
    finally:
        connection.close()


class _MathGradingWorker:
    """Reuse one isolated grader, restarting after timeout or process failure."""

    def __init__(self, timeout_seconds: float = MATH_WORKER_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds
        self._process = None
        self._connection: Connection | None = None
        self._lock = threading.RLock()

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_math_worker_main, args=(child,), daemon=True)
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection = parent
        self._process = process

    def grade(self, solution: str, prediction: str) -> dict[str, Any]:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self.close()
                self._start()
            try:
                self._connection.send((solution, prediction))
                if not self._connection.poll(self.timeout_seconds):
                    raise TimeoutError(
                        "MATH scoring exceeded the isolated worker deadline "
                        f"of {self.timeout_seconds:g} seconds; no score was recorded."
                    )
                response = self._connection.recv()
            except Exception:
                self.close()
                raise
            if response[0] == "error":
                self.close()
                _, error_type, message, remote_traceback = response
                exception = ValueError if error_type == "ValueError" else RuntimeError
                raise exception(
                    f"MATH grading worker failed ({error_type}): {message}\n"
                    f"{remote_traceback}"
                )
            return response[1]

    def close(self) -> None:
        """Release the Pipe and terminate/join the owned worker at shutdown."""

        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._process is not None:
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=1.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)
                self._process.close()
                self._process = None


_worker = _MathGradingWorker()
atexit.register(_worker.close)


def grade_math_response(solution: str, prediction: str) -> dict[str, Any]:
    """Return a binary score and serialized extraction evidence for one answer.

    ``solution`` is the full reference solution, including its LaTeX answer;
    ``prediction`` is the saved, complete model response. For example, gold
    ``\\boxed{1/2}`` and prediction ``\\boxed{2/4}`` score 1 by equivalence.
    Empty/unparsed predictions score 0. Missing or unparsed gold, incompatible
    dependencies, propagated runtime failures, and worker timeouts abort grading.
    Upstream Math-Verify still owns its expected parse/comparison failure rules.
    """

    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("MATH gold solution must be a nonempty string.")
    if not isinstance(prediction, str):
        raise TypeError("MATH prediction must be a string (empty is allowed).")
    validate_math_scoring_runtime()
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return _worker.grade(solution, prediction)
    return _grade_math_response_local(solution, prediction)
