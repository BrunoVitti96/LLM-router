"""Inference for Fast-dLLM v2 with its synchronized block-decoding settings."""

import torch

from llm_router.config import CandidateSpec

from .base_candidate import BaseCandidateInference


FAST_DLLM_PARAMETERS = {
    "block_size": 32,
    "small_block_size": 8,
    "threshold": 0.9,
}


class FastDLLMV2OnePointFiveBInference(BaseCandidateInference):
    spec = CandidateSpec(
        name="fast-dllm-v2-1.5b",
        repo="Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        kind="fast_dllm",
        pinned_revision="25093b6f63300adfd57f72145083c8a528fe4f16",
    )

    def _generate_tokens(
        self, inputs: dict[str, torch.Tensor], max_new_tokens: int
    ) -> torch.Tensor:
        return self.model.generate(
            inputs["input_ids"],
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            **FAST_DLLM_PARAMETERS,
        )
