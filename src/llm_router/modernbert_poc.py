"""Train the hybrid ModernBERT safety router with auxiliary oracle alignment."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from llm_router.config import DEFAULT_CONFIG, RouterConfig
from llm_router.models.modernbert_router import (
    HybridModernBERTRouter,
    build_hybrid_router,
)
from llm_router.oracle import hybrid_routing_loss
from llm_router.public_benchmark import BenchmarkPanel, BenchmarkSplit
from llm_router.utils.training import seed_everything


@dataclass(frozen=True)
class ModernBERTHybridPOCResult:
    model: HybridModernBERTRouter
    tokenizer: object
    safety_probabilities: np.ndarray
    fallback_index: int
    nonfallback_indices: np.ndarray
    history: pd.DataFrame
    training_seconds: float


def _autocast(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)


def train_modernbert_hybrid_poc(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    *,
    config: RouterConfig = DEFAULT_CONFIG,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    quality_epsilon: float = 0.0,
    safety_loss_weight: float = 1.0,
    oracle_auxiliary_weight: float = 0.25,
    device: str | None = None,
) -> ModernBERTHybridPOCResult:
    """Train safety estimates while using the oracle only as an auxiliary task.

    Candidate LLMs are never loaded. Benchmark outcomes supervise replacement
    safety; ``panel.latency`` must contain the deterministic analytical proxy.
    """

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if safety_loss_weight <= 0 or oracle_auxiliary_weight < 0:
        raise ValueError("Loss weights must be non-negative and safety must be positive.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    compute_dtype = (
        torch.bfloat16
        if device.startswith("cuda")
        and torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    )
    seed_everything(config.seed)
    fallback_index = int(panel.score[split.train].mean(axis=0).argmax())
    nonfallback_indices = np.array(
        [index for index in range(len(panel.models)) if index != fallback_index],
        dtype=int,
    )
    texts = np.array(
        [
            f"[DATASET={row.dataset}] [PROMPT_TOKENS={int(row.prompt_tokens)}] "
            f"{row.prompt}"
            for row in panel.examples.itertuples()
        ],
        dtype=object,
    )
    model, tokenizer = build_hybrid_router(
        config,
        nonfallback_count=len(nonfallback_indices),
        model_count=len(panel.models),
    )
    model.to(device)

    def collate(indices: list[int]):
        normalized = np.asarray(indices, dtype=int)
        encoded = tokenizer(
            [f"classification: {texts[index]}" for index in normalized],
            padding=True,
            truncation=True,
            max_length=config.max_input_tokens,
            return_tensors="pt",
        )
        return normalized, encoded

    train_loader = DataLoader(
        split.train.tolist(),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * config.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.startswith("cuda") and compute_dtype == torch.float16
    )
    nonfallback_tensor = torch.tensor(
        nonfallback_indices, dtype=torch.long, device=device
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    def compute_loss(indices: np.ndarray, encoded: object):
        outputs = model(**encoded.to(device))
        return hybrid_routing_loss(
            outputs["safety_logits"],
            outputs["oracle_logits"],
            torch.tensor(panel.score[indices], dtype=torch.float32, device=device),
            torch.tensor(panel.latency[indices], dtype=torch.float32, device=device),
            fallback_index,
            nonfallback_tensor,
            quality_epsilon=quality_epsilon,
            safety_loss_weight=safety_loss_weight,
            oracle_auxiliary_weight=oracle_auxiliary_weight,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_examples = 0
        for indices, encoded in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, compute_dtype):
                loss, _ = compute_loss(indices, encoded)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_total += float(loss.detach()) * len(indices)
            train_examples += len(indices)

        model.eval()
        validation_totals = {
            "total": 0.0,
            "safety": 0.0,
            "oracle_auxiliary": 0.0,
        }
        with torch.inference_mode():
            for start in range(0, len(split.validation), batch_size * 2):
                indices = split.validation[start : start + batch_size * 2]
                _, encoded = collate(indices.tolist())
                with _autocast(device, compute_dtype):
                    loss, parts = compute_loss(indices, encoded)
                validation_totals["total"] += float(loss) * len(indices)
                validation_totals["safety"] += float(parts["safety"]) * len(indices)
                validation_totals["oracle_auxiliary"] += float(
                    parts["oracle_auxiliary"]
                ) * len(indices)
        validation_loss = validation_totals["total"] / len(split.validation)
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total / train_examples,
                "validation_total_loss": validation_loss,
                "validation_safety_loss": (
                    validation_totals["safety"] / len(split.validation)
                ),
                "validation_oracle_auxiliary_loss": (
                    validation_totals["oracle_auxiliary"] / len(split.validation)
                ),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name in trainable_names
            }

    if best_state is None:
        raise RuntimeError("ModernBERT training produced no checkpoint.")
    model.load_state_dict(best_state, strict=False)
    model.eval()
    safety_parts = []
    with torch.inference_mode():
        all_indices = np.arange(len(panel.examples))
        for start in range(0, len(all_indices), batch_size * 2):
            indices = all_indices[start : start + batch_size * 2]
            _, encoded = collate(indices.tolist())
            logits = model(**encoded.to(device))["safety_logits"]
            safety_parts.append(logits.float().sigmoid().cpu().numpy())
    alternative_probability = np.concatenate(safety_parts)
    safety_probabilities = np.ones_like(panel.score, dtype=float)
    safety_probabilities[:, nonfallback_indices] = alternative_probability
    return ModernBERTHybridPOCResult(
        model=model,
        tokenizer=tokenizer,
        safety_probabilities=safety_probabilities,
        fallback_index=fallback_index,
        nonfallback_indices=nonfallback_indices,
        history=pd.DataFrame(history),
        training_seconds=time.perf_counter() - started,
    )


def export_modernbert_hybrid_poc(
    result: ModernBERTHybridPOCResult,
    model_names: tuple[str, ...],
    output_dir: str | Path,
    *,
    selected_threshold: float,
    router_active: bool,
    minimum_predicted_savings: float = 0.02,
    safety_loss_weight: float = 1.0,
    oracle_auxiliary_weight: float = 0.25,
    config: RouterConfig = DEFAULT_CONFIG,
) -> Path:
    """Save LoRA, both heads, tokenizer, safety policy, and training metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.encoder.save_pretrained(output_dir / "lora_adapter")
    torch.save(
        {
            "safety_head": result.model.safety_head.state_dict(),
            "oracle_head": result.model.oracle_head.state_dict(),
        },
        output_dir / "router_heads.pt",
    )
    result.tokenizer.save_pretrained(output_dir / "tokenizer")
    result.history.to_csv(output_dir / "training_history.csv", index=False)
    nonfallback_models = tuple(
        model_names[index] for index in result.nonfallback_indices
    )
    manifest = {
        "schema_version": 2,
        "router": "ModernBERT hybrid safety router",
        "deployed_prediction": "fallback-relative replacement safety",
        "oracle_role": "training-only auxiliary loss",
        "encoder_repo": config.encoder_repo,
        "encoder_revision": config.encoder_revision,
        "router_max_input_tokens": config.max_input_tokens,
        "model_names": model_names,
        "fallback_model": model_names[result.fallback_index],
        "nonfallback_models": nonfallback_models,
        "safety_definition": "candidate_quality >= fallback_quality - epsilon",
        "loss": {
            "safety_bce_weight": safety_loss_weight,
            "oracle_auxiliary_weight": oracle_auxiliary_weight,
        },
        "selected_safety_threshold": selected_threshold,
        "minimum_predicted_savings": minimum_predicted_savings,
        "router_active": router_active,
        "latency_source": "analytical_model_profile_and_prompt_tokens",
        "candidate_inference_used_for_latency": False,
        "training_seconds": result.training_seconds,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
