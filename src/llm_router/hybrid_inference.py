"""Inference and demo helpers for the exported schema-v5 hybrid router.

ModernBERT supplies calibrated replacement-safety probabilities.  Candidate
latencies are always calculated from the analytical scenario; this runtime does
not load or time Fin-R1, Qwen, or any other candidate LLM.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from llm_router.analytical_latency import (
    AnalyticalModelProfile,
    HardwareProfile,
    OutputLengthPolicy,
    estimate_latency_seconds,
)
from llm_router.config import RouterConfig
from llm_router.input_representation import encode_router_texts
from llm_router.models.modernbert_router import (
    MODERNBERT_REFERENCE_COMPILE,
    HybridModernBERTRouter,
)
from llm_router.public_benchmark import EconomicsScenario, ModelProfile


@dataclass(frozen=True)
class HybridRouterDecision:
    """One inspectable routing decision for the interactive demo."""

    selected_model: str
    fallback_model: str
    fallback_used: bool
    router_active: bool
    reason: str
    safety_probabilities: dict[str, float]
    analytical_candidate_latency_s: dict[str, float]
    eligible: dict[str, bool]
    measured_router_overhead_s: float
    estimated_net_savings_s: float
    estimated_net_savings_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly values with readable millisecond fields."""

        payload = asdict(self)
        payload["measured_router_overhead_ms"] = (
            1_000.0 * self.measured_router_overhead_s
        )
        payload["estimated_net_savings_ms"] = 1_000.0 * self.estimated_net_savings_s
        return payload


def select_analytical_route(
    *,
    model_names: tuple[str, ...],
    fallback_model: str,
    safety_probabilities: dict[str, float],
    analytical_latency_s: dict[str, float],
    threshold: float,
    minimum_predicted_savings: float,
    router_active: bool,
) -> tuple[str, dict[str, bool], str]:
    """Apply the deployed safety and analytical-speed eligibility rules."""

    if fallback_model not in model_names:
        raise ValueError("Fallback model must be part of model_names.")
    if set(analytical_latency_s) != set(model_names):
        raise ValueError("Every model requires one analytical latency estimate.")
    eligible = {name: False for name in model_names}
    eligible[fallback_model] = True
    if not router_active:
        return fallback_model, eligible, "policy disabled; fail closed to fallback"

    fallback_latency = analytical_latency_s[fallback_model]
    for candidate in model_names:
        if candidate == fallback_model:
            continue
        probability = safety_probabilities.get(candidate)
        if probability is None:
            raise ValueError(f"Missing safety probability for {candidate!r}.")
        eligible[candidate] = (
            probability >= threshold
            and analytical_latency_s[candidate]
            <= fallback_latency * (1.0 - minimum_predicted_savings)
        )
    selected = min(
        (name for name in model_names if eligible[name]),
        key=analytical_latency_s.__getitem__,
    )
    if selected == fallback_model:
        reason = "no alternative passed both safety and analytical-speed gates"
    else:
        reason = "fastest alternative passing safety and analytical-speed gates"
    return selected, eligible, reason


def scenario_from_experiment_manifest(path: str | Path) -> EconomicsScenario:
    """Reconstruct the analytical scenario saved beside a router artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))["scenario"]
    hardware = payload["hardware"]
    output = payload["output_length_policy"]
    profiles = tuple(
        ModelProfile(name=name, **settings)
        for name, settings in payload["models"].items()
    )
    scenario = EconomicsScenario(
        name=payload["name"],
        as_of=payload["as_of"],
        profiles=profiles,
        network_s=payload["network_s"],
        router_overhead_s=payload["router_overhead_s"],
        notes=payload["notes"],
        latency_method=payload["latency_method"],
        effective_tflops=hardware["effective_tflops"],
        memory_bandwidth_gbps=hardware["memory_bandwidth_gbps"],
        fixed_model_overhead_s=hardware["fixed_model_overhead_s"],
        output_base_tokens=output["base_tokens"],
        output_tokens_per_prompt_token=output["tokens_per_prompt_token"],
        output_min_tokens=output["minimum_tokens"],
        output_max_tokens=output["maximum_tokens"],
    )
    scenario.validate()
    return scenario


class HybridModernBERTRouterRuntime:
    """Run calibrated ModernBERT safety plus analytical candidate selection."""

    def __init__(
        self,
        *,
        model: HybridModernBERTRouter,
        tokenizer: Any,
        model_names: tuple[str, ...],
        fallback_model: str,
        calibration_parameters: dict[str, tuple[float, float]],
        selected_threshold: float,
        router_active: bool,
        minimum_predicted_savings: float,
        max_input_tokens: int,
        scenario: EconomicsScenario,
        device: str,
        input_truncation_strategy: str = "prefix",
    ) -> None:
        if scenario.latency_method != "analytical":
            raise ValueError("The hybrid demo requires analytical candidate latency.")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.model_names = model_names
        self.fallback_model = fallback_model
        self.nonfallback_models = tuple(
            name for name in model_names if name != fallback_model
        )
        self.calibration_parameters = calibration_parameters
        self.selected_threshold = float(selected_threshold)
        self.router_active = bool(router_active)
        self.minimum_predicted_savings = float(minimum_predicted_savings)
        self.max_input_tokens = int(max_input_tokens)
        self.input_truncation_strategy = input_truncation_strategy
        self.scenario = scenario
        self.device = device
        self.compute_dtype = (
            torch.bfloat16
            if device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )

    @classmethod
    def from_training_result(
        cls,
        training: Any,
        *,
        model_names: tuple[str, ...],
        fallback_model: str,
        selected_threshold: float,
        router_active: bool,
        minimum_predicted_savings: float,
        scenario: EconomicsScenario,
        config: RouterConfig,
        device: str,
    ) -> HybridModernBERTRouterRuntime:
        """Reuse the already-loaded best checkpoint inside the Colab notebook."""

        return cls(
            model=training.model,
            tokenizer=training.tokenizer,
            model_names=model_names,
            fallback_model=fallback_model,
            calibration_parameters=training.calibration_parameters,
            selected_threshold=selected_threshold,
            router_active=router_active,
            minimum_predicted_savings=minimum_predicted_savings,
            max_input_tokens=config.max_input_tokens,
            input_truncation_strategy=config.input_truncation_strategy,
            scenario=scenario,
            device=device,
        )

    @classmethod
    def from_artifact(
        cls, artifact_dir: str | Path, *, device: str = "cuda"
    ) -> HybridModernBERTRouterRuntime:
        """Load the best exported adapter, heads, policy, and analytical scenario."""

        artifact_dir = Path(artifact_dir)
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        scenario = scenario_from_experiment_manifest(
            artifact_dir.parent / "experiment_manifest.json"
        )
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
        base = AutoModel.from_pretrained(
            manifest["encoder_repo"],
            revision=manifest["encoder_revision"],
            attn_implementation="sdpa",
            reference_compile=manifest.get(
                "encoder_reference_compile", MODERNBERT_REFERENCE_COMPILE
            ),
        )
        hidden_size = int(base.config.hidden_size)
        encoder = PeftModel.from_pretrained(base, artifact_dir / "lora_adapter")
        model = HybridModernBERTRouter(
            encoder,
            hidden_size,
            len(manifest["nonfallback_models"]),
            len(manifest["model_names"]),
        )
        heads = torch.load(
            artifact_dir / "router_heads.pt", map_location="cpu", weights_only=True
        )
        model.safety_head.load_state_dict(heads["safety_head"])
        model.oracle_head.load_state_dict(heads["oracle_head"])
        calibration = {
            name: (values["slope"], values["intercept"])
            for name, values in manifest["calibration"][
                "deployment_parameters"
            ].items()
        }
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_names=tuple(manifest["model_names"]),
            fallback_model=manifest["fallback_model"],
            calibration_parameters=calibration,
            selected_threshold=manifest["selected_safety_threshold"],
            router_active=manifest["router_active"],
            minimum_predicted_savings=manifest["minimum_predicted_savings"],
            max_input_tokens=manifest["router_max_input_tokens"],
            input_truncation_strategy=manifest.get(
                "router_input_truncation_strategy", "prefix"
            ),
            scenario=scenario,
            device=device,
        )

    def _candidate_latency(self, prompt_tokens: float) -> dict[str, float]:
        hardware = HardwareProfile(
            effective_tflops=self.scenario.effective_tflops,
            memory_bandwidth_gbps=self.scenario.memory_bandwidth_gbps,
            fixed_overhead_s=self.scenario.fixed_model_overhead_s,
        )
        output_policy = OutputLengthPolicy(
            base_tokens=self.scenario.output_base_tokens,
            tokens_per_prompt_token=self.scenario.output_tokens_per_prompt_token,
            minimum_tokens=self.scenario.output_min_tokens,
            maximum_tokens=self.scenario.output_max_tokens,
        )
        latencies: dict[str, float] = {}
        for profile in self.scenario.profiles:
            if profile.parameters_billions is None or profile.architecture is None:
                raise ValueError(
                    f"Candidate {profile.name!r} lacks an analytical profile."
                )
            analytical = AnalyticalModelProfile(
                name=profile.name,
                parameters_billions=profile.parameters_billions,
                active_parameters_billions=profile.active_parameters_billions,
                architecture=profile.architecture,
                weight_bits=profile.weight_bits,
                diffusion_steps=profile.diffusion_steps,
                diffusion_block_size=profile.diffusion_block_size,
                architecture_factor=profile.architecture_factor,
            )
            model_latency = estimate_latency_seconds(
                [prompt_tokens], analytical, hardware, output_policy
            )[0]
            latencies[profile.name] = float(
                self.scenario.network_s + profile.queue_s + model_latency
            )
        return latencies

    @torch.inference_mode()
    def route(self, prompt: str, prompt_tokens: float) -> HybridRouterDecision:
        """Measure this ModernBERT call and analytically select a candidate."""

        if not str(prompt).strip():
            raise ValueError("Prompt cannot be empty.")
        if not np.isfinite(prompt_tokens) or prompt_tokens < 0:
            raise ValueError("prompt_tokens must be finite and non-negative.")
        text = f"classification: [PROMPT_TOKENS={int(prompt_tokens)}] {prompt}"
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        encoded = encode_router_texts(
            self.tokenizer,
            [text],
            max_input_tokens=self.max_input_tokens,
            truncation_strategy=self.input_truncation_strategy,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        context = (
            torch.autocast("cuda", dtype=self.compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with context:
            logits = self.model(**encoded)["safety_logits"]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        overhead_s = time.perf_counter() - started

        raw_logits = logits.float().cpu().numpy()[0]
        probabilities: dict[str, float] = {}
        for candidate, logit in zip(self.nonfallback_models, raw_logits):
            slope, intercept = self.calibration_parameters[candidate]
            calibrated = np.clip(slope * float(logit) + intercept, -40, 40)
            probabilities[candidate] = float(1.0 / (1.0 + np.exp(-calibrated)))

        latencies = self._candidate_latency(float(prompt_tokens))
        selected, eligible, reason = select_analytical_route(
            model_names=self.model_names,
            fallback_model=self.fallback_model,
            safety_probabilities=probabilities,
            analytical_latency_s=latencies,
            threshold=self.selected_threshold,
            minimum_predicted_savings=self.minimum_predicted_savings,
            router_active=self.router_active,
        )
        fallback_latency = latencies[self.fallback_model]
        net_savings = fallback_latency - latencies[selected] - overhead_s
        return HybridRouterDecision(
            selected_model=selected,
            fallback_model=self.fallback_model,
            fallback_used=selected == self.fallback_model,
            router_active=self.router_active,
            reason=reason,
            safety_probabilities=probabilities,
            analytical_candidate_latency_s=latencies,
            eligible=eligible,
            measured_router_overhead_s=overhead_s,
            estimated_net_savings_s=float(net_savings),
            estimated_net_savings_fraction=float(net_savings / fallback_latency),
        )


def create_gradio_demo(runtime: HybridModernBERTRouterRuntime):
    """Create a small Colab-shareable interface without loading candidate LLMs."""

    import gradio as gr

    def evaluate(prompt: str, prompt_tokens: float) -> dict[str, Any]:
        decision = runtime.route(prompt, prompt_tokens)
        payload = decision.to_dict()
        payload["prompt"] = prompt
        payload["candidate_latency_notice"] = (
            "Analytical estimates only; candidate LLMs were not loaded or timed."
        )
        return payload

    return gr.Interface(
        fn=evaluate,
        inputs=[
            gr.Textbox(lines=8, label="Prompt"),
            gr.Number(value=128, minimum=0, label="Estimated candidate prompt tokens"),
        ],
        outputs=gr.JSON(label="Routing decision"),
        title="Calibrated ModernBERT analytical-latency router",
        description=(
            "ModernBERT is measured for this request. Candidate latency is computed "
            "analytically from model size, architecture, precision, prompt length, "
            "and the frozen hardware scenario."
        ),
    )
