"""Measure ModernBERT router overhead without timing candidate LLMs.

The benchmark is diagnostic evidence.  Candidate latency remains analytical,
and the frozen policy continues to use its predeclared 4 ms nominal and 20 ms
conservative assumptions so hardware noise cannot change setup selection.
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class RouterOverheadBenchmark:
    """Summary and request-level timings for one batch-one router benchmark."""

    summary: dict[str, Any]
    samples: pd.DataFrame

    def export(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Write auditable JSON summary and CSV timings into a report directory."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "modernbert_overhead_benchmark.json"
        samples_path = output_dir / "modernbert_overhead_samples.csv"
        summary_path.write_text(
            json.dumps(self.summary, indent=2), encoding="utf-8"
        )
        self.samples.to_csv(samples_path, index=False)
        return summary_path, samples_path


def _distribution_ms(values_s: np.ndarray) -> dict[str, float]:
    values_ms = np.asarray(values_s, dtype=float) * 1_000.0
    if values_ms.size == 0 or not np.isfinite(values_ms).all():
        raise ValueError("Timing samples must be non-empty and finite.")
    return {
        "mean": float(values_ms.mean()),
        "p50": float(np.quantile(values_ms, 0.50)),
        "p95": float(np.quantile(values_ms, 0.95)),
        "maximum": float(values_ms.max()),
    }


def summarize_router_overhead_samples(
    end_to_end_s: np.ndarray,
    model_only_s: np.ndarray,
    *,
    device: str,
    warmup_requests: int,
    max_input_tokens: int,
) -> dict[str, Any]:
    """Build the stable, JSON-serializable benchmark summary."""

    end_to_end_s = np.asarray(end_to_end_s, dtype=float)
    model_only_s = np.asarray(model_only_s, dtype=float)
    if end_to_end_s.shape != model_only_s.shape:
        raise ValueError("End-to-end and model-only timings must have equal length.")
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    return {
        "schema_version": 1,
        "measured_component": (
            "ModernBERT tokenization, host-to-device transfer, and router inference"
        ),
        "candidate_latency_method": "analytical; candidate LLMs were not loaded",
        "measurement_affects_frozen_policy": False,
        "batch_size": 1,
        "timed_requests": int(end_to_end_s.size),
        "warmup_requests": int(warmup_requests),
        "max_input_tokens": int(max_input_tokens),
        "device": device,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "end_to_end_ms": _distribution_ms(end_to_end_s),
        "model_only_ms": _distribution_ms(model_only_s),
    }


def benchmark_modernbert_overhead(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str] | np.ndarray,
    prompt_tokens: list[float] | np.ndarray,
    *,
    device: str,
    max_input_tokens: int = 512,
    timed_requests: int = 100,
    warmup_requests: int = 10,
    seed: int = 42,
) -> RouterOverheadBenchmark:
    """Time batch-one ModernBERT routing on deterministic prompt samples.

    End-to-end samples include the tokenizer, tensor transfer, and ModernBERT.
    Model-only samples use pre-tokenized device tensors.  No candidate model is
    imported, loaded, called, or timed by this function.
    """

    prompts = np.asarray(prompts, dtype=object)
    prompt_tokens = np.asarray(prompt_tokens, dtype=float)
    if len(prompts) != len(prompt_tokens) or len(prompts) == 0:
        raise ValueError("Prompts and prompt-token counts must be non-empty and align.")
    if timed_requests <= 0 or warmup_requests < 0:
        raise ValueError("timed_requests must be positive and warmup cannot be negative.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA router timing was requested but CUDA is unavailable.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(prompts), size=timed_requests, replace=True)
    texts = [
        f"classification: [PROMPT_TOKENS={int(prompt_tokens[index])}] "
        f"{prompts[index]}"
        for index in indices
    ]
    original_device = next(model.parameters()).device
    model.to(device).eval()
    compute_dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    )

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def encode(text: str):
        return tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        ).to(device)

    def infer(encoded: Any) -> None:
        context = (
            torch.autocast("cuda", dtype=compute_dtype)
            if device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), context:
            model(**encoded)

    warmup_text = texts[0]
    for _ in range(warmup_requests):
        infer(encode(warmup_text))
    synchronize()

    end_to_end: list[float] = []
    model_only: list[float] = []
    input_lengths: list[int] = []
    for text in texts:
        synchronize()
        started = time.perf_counter()
        encoded = encode(text)
        infer(encoded)
        synchronize()
        end_to_end.append(time.perf_counter() - started)
        input_lengths.append(int(encoded["attention_mask"].sum().item()))

        synchronize()
        started = time.perf_counter()
        infer(encoded)
        synchronize()
        model_only.append(time.perf_counter() - started)

    model.to(original_device)
    end_to_end_array = np.asarray(end_to_end)
    model_only_array = np.asarray(model_only)
    summary = summarize_router_overhead_samples(
        end_to_end_array,
        model_only_array,
        device=device,
        warmup_requests=warmup_requests,
        max_input_tokens=max_input_tokens,
    )
    samples = pd.DataFrame(
        {
            "sample": np.arange(1, timed_requests + 1),
            "source_index": indices,
            "router_input_tokens": input_lengths,
            "end_to_end_ms": end_to_end_array * 1_000.0,
            "model_only_ms": model_only_array * 1_000.0,
        }
    )
    return RouterOverheadBenchmark(summary=summary, samples=samples)
