"""NF4 inference for the strongest Qwen fallback candidate."""

from llm_router.config import CandidateSpec

from .base_candidate import BaseCandidateInference


class QwenSevenB4BitInference(BaseCandidateInference):
    spec = CandidateSpec(
        name="qwen2.5-7b-4bit",
        repo="Qwen/Qwen2.5-7B-Instruct",
        four_bit=True,
    )
