"""Inference for the standard 1.5B Qwen replacement candidate."""

from llm_router.config import CandidateSpec

from .base_candidate import BaseCandidateInference


class QwenOnePointFiveBInference(BaseCandidateInference):
    spec = CandidateSpec(
        name="qwen2.5-1.5b-ar",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
    )
