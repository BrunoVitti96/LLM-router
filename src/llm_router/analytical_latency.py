"""Hardware-aware, measurement-free latency estimates for routing POCs.

The estimator is deliberately simple.  It turns model-card facts and a prompt
length into a deterministic serving-time proxy; it is not a production latency
claim.  No candidate model is loaded or executed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Architecture = Literal["autoregressive", "diffusion"]


@dataclass(frozen=True)
class HardwareProfile:
    """Effective single-request hardware assumptions used by the proxy."""

    effective_tflops: float = 60.0
    memory_bandwidth_gbps: float = 900.0
    fixed_overhead_s: float = 0.015

    def validate(self) -> None:
        if self.effective_tflops <= 0 or self.memory_bandwidth_gbps <= 0:
            raise ValueError("Hardware throughput assumptions must be positive.")
        if self.fixed_overhead_s < 0:
            raise ValueError("fixed_overhead_s cannot be negative.")


@dataclass(frozen=True)
class AnalyticalModelProfile:
    """Facts needed to compare candidate latency without timing inference."""

    name: str
    parameters_billions: float
    architecture: Architecture
    active_parameters_billions: float | None = None
    weight_bits: int = 16
    diffusion_steps: int = 1
    diffusion_block_size: int = 1
    architecture_factor: float = 1.0

    def validate(self) -> None:
        active = self.active_parameters_billions or self.parameters_billions
        if not self.name or self.parameters_billions <= 0 or active <= 0:
            raise ValueError("Model names and parameter counts must be positive.")
        if self.architecture not in {"autoregressive", "diffusion"}:
            raise ValueError(f"Unsupported architecture: {self.architecture!r}")
        if self.weight_bits <= 0 or self.diffusion_steps <= 0:
            raise ValueError("Precision and diffusion steps must be positive.")
        if self.diffusion_block_size <= 0 or self.architecture_factor <= 0:
            raise ValueError("Block size and architecture factor must be positive.")


@dataclass(frozen=True)
class OutputLengthPolicy:
    """Leakage-free response-length assumption derived only from prompt size."""

    base_tokens: float = 24.0
    tokens_per_prompt_token: float = 0.20
    minimum_tokens: int = 16
    maximum_tokens: int = 256

    def estimate(self, prompt_tokens: np.ndarray | Sequence[float]) -> np.ndarray:
        prompt_tokens = np.asarray(prompt_tokens, dtype=float)
        if np.any(prompt_tokens < 0) or not np.isfinite(prompt_tokens).all():
            raise ValueError("Prompt-token counts must be finite and non-negative.")
        estimate = self.base_tokens + self.tokens_per_prompt_token * prompt_tokens
        return np.clip(estimate, self.minimum_tokens, self.maximum_tokens)


def estimate_latency_seconds(
    prompt_tokens: np.ndarray | Sequence[float],
    model: AnalyticalModelProfile,
    hardware: HardwareProfile,
    output_policy: OutputLengthPolicy | None = None,
) -> np.ndarray:
    """Estimate warm batch-one latency from size, architecture, and prompt size.

    Prefill is treated as compute-bound: roughly two floating-point operations
    per active parameter and input token.  Autoregressive decoding is treated as
    memory-bound because weights are streamed once per sequential output token.
    Diffusion decoding instead performs a fixed number of block-denoising passes.
    The maximum of compute and memory time is used for every decoding pass.
    """

    output_policy = output_policy or OutputLengthPolicy()
    model.validate()
    hardware.validate()
    prompt = np.asarray(prompt_tokens, dtype=float)
    output = output_policy.estimate(prompt)
    active_parameters = (
        model.active_parameters_billions or model.parameters_billions
    ) * 1e9
    parameter_bytes = active_parameters * model.weight_bits / 8.0
    seconds_per_compute_token = (
        2.0 * active_parameters / (hardware.effective_tflops * 1e12)
    )
    seconds_per_memory_pass = parameter_bytes / (
        hardware.memory_bandwidth_gbps * 1e9
    )
    prefill = prompt * seconds_per_compute_token
    decode_pass = max(seconds_per_compute_token, seconds_per_memory_pass)

    if model.architecture == "autoregressive":
        decode = output * decode_pass
    else:
        blocks = np.ceil(output / model.diffusion_block_size)
        decode = blocks * model.diffusion_steps * decode_pass
    return hardware.fixed_overhead_s + model.architecture_factor * (prefill + decode)


def estimate_latency_matrix(
    prompt_tokens: np.ndarray | Sequence[float],
    models: Sequence[AnalyticalModelProfile],
    hardware: HardwareProfile,
    output_policy: OutputLengthPolicy | None = None,
) -> np.ndarray:
    """Return ``[prompt, model]`` analytical latency estimates."""

    output_policy = output_policy or OutputLengthPolicy()
    if len(models) < 2:
        raise ValueError("Routing requires at least two analytical model profiles.")
    return np.column_stack(
        [
            estimate_latency_seconds(prompt_tokens, model, hardware, output_policy)
            for model in models
        ]
    )
