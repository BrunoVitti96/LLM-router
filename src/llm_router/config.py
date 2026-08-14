"""Single source of truth for the synchronized v4 experiment contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    repo: str
    kind: str = "causal"
    four_bit: bool = False
    pinned_revision: str | None = None


CANDIDATES = (
    CandidateSpec("qwen2.5-1.5b-ar", "Qwen/Qwen2.5-1.5B-Instruct"),
    CandidateSpec(
        "fast-dllm-v2-1.5b",
        "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        kind="fast_dllm",
        pinned_revision="25093b6f63300adfd57f72145083c8a528fe4f16",
    ),
    CandidateSpec(
        "qwen2.5-7b-4bit",
        "Qwen/Qwen2.5-7B-Instruct",
        four_bit=True,
    ),
)
MODEL_NAMES = tuple(candidate.name for candidate in CANDIDATES)
EXPECTED_MODEL_REPOS = {candidate.name: candidate.repo for candidate in CANDIDATES}


@dataclass(frozen=True)
class RouterConfig:
    """Settings that affect evidence identity, training, or deployment."""

    scope_schema_version: int = 4
    required_v3_schema_version: int = 3
    required_v3_prompt_template: str = "v3-json-only-2026-08-07"
    seed: int = 42
    tasks: tuple[str, ...] = ("gsm8k", "mmlu", "arc_challenge")
    n_per_task: int = 300
    model_names: tuple[str, ...] = MODEL_NAMES

    minimum_quality_retention: float = 0.98
    quality_safety_epsilon: float = 0.0
    minimum_predicted_speedup: float = 0.02
    safety_definition: str = "candidate_quality >= fallback_quality - epsilon"

    encoder_repo: str = "nomic-ai/modernbert-embed-base"
    encoder_revision: str = "d556a88e332558790b210f7bdbe87da2fa94a8d8"
    max_input_tokens: int = 512
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    batch_size: int = 8
    gradient_accumulation: int = 2
    max_epochs: int = 15
    min_epochs: int = 6
    early_stopping_patience: int = 5
    lora_lr: float = 1e-4
    head_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    warmup_prompts: int = 2

    safety_loss_weight: float = 1.0
    latency_loss_weight: float = 0.75
    token_loss_weight: float = 0.10
    opportunity_margin_loss_weight: float = 0.25
    missed_opportunity_weight: float = 2.0
    unsafe_selection_weight: float = 4.0
    safety_logit_margin: float = 1.0

    platt_folds: int = 5
    safety_threshold_grid: tuple[float, ...] = (
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.975,
    )
    latency_blend_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)

    def validate(self) -> None:
        assert self.scope_schema_version == 4
        assert self.required_v3_schema_version == 3
        assert self.n_per_task == 300 and len(self.tasks) == 3
        assert self.minimum_quality_retention == 0.98
        assert self.lora_r == 4 and self.lora_alpha == 8
        assert self.lora_target_modules == "all-linear"
        assert self.max_input_tokens == 512
        assert self.platt_folds == 5
        assert self.minimum_predicted_speedup == 0.02

    def contract(
        self, evidence_fingerprint: str, gpu: str, dtype: str
    ) -> dict[str, Any]:
        """Return the exact JSON-serializable router identity."""
        return {
            "scope_schema_version": self.scope_schema_version,
            "scope_file": "llm_router_project_scope_4.md",
            "evidence_fingerprint": evidence_fingerprint,
            "seed": self.seed,
            "tasks": self.tasks,
            "models": self.model_names,
            "split": [0.60, 0.20, 0.20],
            "safety_definition": self.safety_definition,
            "quality_safety_epsilon": self.quality_safety_epsilon,
            "minimum_quality_retention": self.minimum_quality_retention,
            "minimum_predicted_speedup": self.minimum_predicted_speedup,
            "encoder_repo": self.encoder_repo,
            "encoder_revision": self.encoder_revision,
            "router_max_input_tokens": self.max_input_tokens,
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
            },
            "optimization": {
                "batch_size": self.batch_size,
                "gradient_accumulation": self.gradient_accumulation,
                "max_epochs": self.max_epochs,
                "min_epochs": self.min_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "lora_lr": self.lora_lr,
                "head_lr": self.head_lr,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
                "max_grad_norm": self.max_grad_norm,
            },
            "loss": {
                "safety": self.safety_loss_weight,
                "latency": self.latency_loss_weight,
                "tokens": self.token_loss_weight,
                "opportunity_margin": self.opportunity_margin_loss_weight,
                "missed_opportunity_multiplier": self.missed_opportunity_weight,
                "unsafe_selection_multiplier": self.unsafe_selection_weight,
                "safety_logit_margin": self.safety_logit_margin,
                "base_bce_pos_weight": 1.0,
            },
            "calibration": {"method": "per-candidate Platt", "folds": self.platt_folds},
            "safety_threshold_grid": self.safety_threshold_grid,
            "latency_blend_grid": self.latency_blend_grid,
            "checkpoint_rule": "calibrated overhead-inclusive validation routing",
            "deployment_guard": "disable unless retention >= 0.98 and net savings > 0",
            "gpu": gpu,
            "dtype": dtype,
        }


DEFAULT_CONFIG = RouterConfig()


def candidate_dict() -> dict[str, dict[str, Any]]:
    return {candidate.name: asdict(candidate) for candidate in CANDIDATES}
