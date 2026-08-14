"""Standalone inference runtime for an exported ModernBERT router artifact."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from llm_router.utils.text import format_router_input

from .modernbert_router import DecisionAlignedRouter


@dataclass(frozen=True)
class RouterDecision:
    selected_model: str
    router_active: bool
    safety_probabilities: dict[str, float]
    predicted_latency_s: dict[str, float]
    eligible: dict[str, bool]
    router_overhead_s: float


class ModernBERTRouterInference:
    """Load calibration, selector policy, adapter, and heads from one artifact."""

    def __init__(
        self,
        manifest: dict,
        model: DecisionAlignedRouter | None,
        tokenizer: AutoTokenizer | None,
        device: str,
        compute_dtype: torch.dtype,
    ) -> None:
        self.manifest = manifest
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.compute_dtype = compute_dtype
        self.model_names = tuple(manifest["model_names"])
        self.fallback_model = manifest["fallback_model"]
        self.nonfallback_models = tuple(manifest["nonfallback_models"])
        self.router_active = bool(manifest["router_active"])

    @classmethod
    def from_artifact(
        cls, artifact_dir: str | Path, device: str = "cuda"
    ) -> "ModernBERTRouterInference":
        artifact_dir = Path(artifact_dir)
        manifest = json.loads(
            (artifact_dir / "router_manifest.json").read_text(encoding="utf-8")
        )
        compute_dtype = (
            torch.bfloat16
            if device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )

        # A disabled policy is deliberately cheap: fallback immediately.
        if not manifest["router_active"]:
            return cls(manifest, None, None, device, compute_dtype)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "The active router requested CUDA, but CUDA is unavailable."
            )

        tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
        base = AutoModel.from_pretrained(
            manifest["encoder_repo"],
            revision=manifest["encoder_revision"],
            attn_implementation="sdpa",
        )
        hidden_size = int(base.config.hidden_size)
        encoder = PeftModel.from_pretrained(base, artifact_dir / "lora_adapter")
        model = DecisionAlignedRouter(
            encoder,
            hidden_size,
            len(manifest["nonfallback_models"]),
            len(manifest["model_names"]),
        )
        head_state = torch.load(
            artifact_dir / "router_heads.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(head_state, strict=False)
        model.to(device).eval()
        return cls(manifest, model, tokenizer, device, compute_dtype)

    @torch.inference_mode()
    def route(
        self,
        *,
        prompt: str,
        task: str,
        subject: str,
        num_choices: int,
        length_bin: str,
    ) -> RouterDecision:
        if not self.router_active:
            return RouterDecision(
                selected_model=self.fallback_model,
                router_active=False,
                safety_probabilities={},
                predicted_latency_s={},
                eligible={self.fallback_model: True},
                router_overhead_s=0.0,
            )
        assert self.model is not None and self.tokenizer is not None
        if task not in self.manifest["task_model_median_latency"]:
            raise ValueError(f"Task {task!r} was not present during training.")

        text = format_router_input(prompt, task, subject, num_choices, length_bin)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        encoded = self.tokenizer(
            [f"classification: {text}"],
            padding=True,
            truncation=True,
            max_length=int(self.manifest["router_max_input_tokens"]),
            return_tensors="pt",
        ).to(self.device)
        autocast = (
            torch.autocast("cuda", dtype=self.compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with autocast:
            outputs = self.model(**encoded)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        overhead_s = time.perf_counter() - started

        logits = outputs["safety_logits"].float().cpu().numpy()[0]
        neural_latency = np.maximum(
            1e-6, np.expm1(outputs["latency_log"].float().cpu().numpy()[0])
        )
        probabilities = {}
        for candidate, logit in zip(self.nonfallback_models, logits):
            parameters = self.manifest["platt_parameters"][candidate]
            calibrated = np.clip(
                parameters["slope"] * logit + parameters["intercept"], -40, 40
            )
            probabilities[candidate] = float(1.0 / (1.0 + np.exp(-calibrated)))

        baseline = np.array(
            [
                self.manifest["task_model_median_latency"][task][name]
                for name in self.model_names
            ],
            dtype=float,
        )
        blend = float(self.manifest["selected_latency_blend"])
        predicted = blend * neural_latency + (1.0 - blend) * baseline
        fallback_index = self.model_names.index(self.fallback_model)
        eligible = {name: False for name in self.model_names}
        eligible[self.fallback_model] = True
        min_speedup = float(self.manifest["minimum_predicted_speedup"])
        for candidate in self.nonfallback_models:
            index = self.model_names.index(candidate)
            eligible[candidate] = probabilities[candidate] >= float(
                self.manifest["selected_thresholds"][candidate]
            ) and predicted[index] <= predicted[fallback_index] * (1.0 - min_speedup)
        chosen_index = min(
            (index for index, name in enumerate(self.model_names) if eligible[name]),
            key=lambda index: predicted[index],
        )
        return RouterDecision(
            selected_model=self.model_names[chosen_index],
            router_active=True,
            safety_probabilities=probabilities,
            predicted_latency_s={
                name: float(predicted[index])
                for index, name in enumerate(self.model_names)
            },
            eligible=eligible,
            router_overhead_s=overhead_s,
        )
