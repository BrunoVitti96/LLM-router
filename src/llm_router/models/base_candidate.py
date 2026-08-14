"""Shared generation mechanics; model policy lives in model-specific modules."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_router.config import CandidateSpec


SYSTEM_PROMPT = (
    "You are a careful benchmark assistant. Think through the problem "
    "internally and return only the requested JSON object, with no prose."
)
TASK_MAX_NEW_TOKENS = {"gsm8k": 192, "mmlu": 64, "arc_challenge": 64}
MAX_INPUT_TOKENS = 1024


@dataclass(frozen=True)
class GenerationResult:
    response: str
    encode_s: float
    generation_s: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    peak_vram_gb: float


class BaseCandidateInference:
    """Warm, deterministic candidate inference matching the v3 measurements."""

    spec: CandidateSpec

    def __init__(
        self,
        *,
        revision: str | None = None,
        compute_dtype: torch.dtype | None = None,
        device_map: str = "auto",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Candidate timing requires a CUDA GPU.")
        self.revision = revision or self.spec.pinned_revision
        self.compute_dtype = compute_dtype or (
            torch.bfloat16
            if torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )
        quantization = None
        if self.spec.four_bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        started = time.perf_counter()
        is_fast = self.spec.kind == "fast_dllm"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.repo,
            revision=self.revision,
            trust_remote_code=is_fast,
        )
        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": self.compute_dtype,
            "trust_remote_code": is_fast,
            "revision": self.revision,
            "low_cpu_mem_usage": True,
        }
        if quantization is not None:
            load_kwargs["quantization_config"] = quantization
        self.model = AutoModelForCausalLM.from_pretrained(self.spec.repo, **load_kwargs)
        self.model.eval()
        self.load_time_s = time.perf_counter() - started

    def encode(self, prompt: str) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        return {key: value.to(self.model.device) for key, value in encoded.items()}

    def _generate_tokens(
        self, inputs: dict[str, torch.Tensor], max_new_tokens: int
    ) -> torch.Tensor:
        return self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )

    @torch.inference_mode()
    def generate(self, prompt: str, task: str) -> GenerationResult:
        if task not in TASK_MAX_NEW_TOKENS:
            raise ValueError(
                f"Unknown task {task!r}; expected {tuple(TASK_MAX_NEW_TOKENS)}"
            )

        encode_started = time.perf_counter()
        inputs = self.encode(prompt)
        encode_s = time.perf_counter() - encode_started
        input_tokens = int(inputs["input_ids"].shape[1])

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = self._generate_tokens(inputs, TASK_MAX_NEW_TOKENS[task])
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - started

        new_ids = output[0, input_tokens:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        output_tokens = int(new_ids.numel())
        return GenerationResult(
            response=response,
            encode_s=encode_s,
            generation_s=generation_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / max(generation_s, 1e-9),
            peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
        )

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def __enter__(self) -> "BaseCandidateInference":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
