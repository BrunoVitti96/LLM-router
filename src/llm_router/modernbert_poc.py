"""Train ModernBERT to imitate an analytical-latency, quality-aware oracle."""

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
    OracleModernBERTRouter,
    build_oracle_router,
)
from llm_router.oracle import oracle_choices, oracle_routing_loss
from llm_router.public_benchmark import BenchmarkPanel, BenchmarkSplit
from llm_router.utils.training import seed_everything


@dataclass(frozen=True)
class ModernBERTPOCResult:
    model: OracleModernBERTRouter
    tokenizer: object
    probabilities: np.ndarray
    oracle_targets: np.ndarray
    fallback_index: int
    history: pd.DataFrame
    training_seconds: float


def _autocast(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)


def train_modernbert_oracle_poc(
    panel: BenchmarkPanel,
    split: BenchmarkSplit,
    *,
    config: RouterConfig = DEFAULT_CONFIG,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    quality_epsilon: float = 0.0,
    device: str | None = None,
) -> ModernBERTPOCResult:
    """Fine-tune ModernBERT on pre-collected quality and analytical latency.

    Candidate LLMs are never loaded.  Benchmark scores supervise quality, while
    ``panel.latency`` must contain the deterministic analytical proxy.
    """

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
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
    targets = oracle_choices(
        panel.score, panel.latency, fallback_index, quality_epsilon
    )
    texts = np.array(
        [
            f"[DATASET={row.dataset}] [PROMPT_TOKENS={int(row.prompt_tokens)}] "
            f"{row.prompt}"
            for row in panel.examples.itertuples()
        ],
        dtype=object,
    )
    model, tokenizer = build_oracle_router(config, len(panel.models))
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

    loader = DataLoader(
        split.train.tolist(),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, epochs * len(loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * config.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.startswith("cuda") and compute_dtype == torch.float16
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    def batch_loss(indices: np.ndarray) -> tuple[torch.Tensor, dict]:
        _, encoded = collate(indices.tolist())
        encoded = encoded.to(device)
        with _autocast(device, compute_dtype):
            logits = model(**encoded)["routing_logits"]
            return oracle_routing_loss(
                logits,
                torch.tensor(panel.score[indices], dtype=torch.float32, device=device),
                torch.tensor(panel.latency[indices], dtype=torch.float32, device=device),
                fallback_index,
                quality_epsilon=quality_epsilon,
            )

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_examples = 0
        for indices, encoded in loader:
            encoded = encoded.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, compute_dtype):
                logits = model(**encoded)["routing_logits"]
                loss, _ = oracle_routing_loss(
                    logits,
                    torch.tensor(
                        panel.score[indices], dtype=torch.float32, device=device
                    ),
                    torch.tensor(
                        panel.latency[indices], dtype=torch.float32, device=device
                    ),
                    fallback_index,
                    quality_epsilon=quality_epsilon,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_total += float(loss.detach()) * len(indices)
            train_examples += len(indices)

        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for start in range(0, len(split.validation), batch_size * 2):
                indices = split.validation[start : start + batch_size * 2]
                loss, _ = batch_loss(indices)
                validation_total += float(loss) * len(indices)
        validation_loss = validation_total / len(split.validation)
        history.append(
            {
                "epoch": epoch,
                "train_oracle_loss": train_total / train_examples,
                "validation_oracle_loss": validation_loss,
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
    probability_parts = []
    with torch.inference_mode():
        all_indices = np.arange(len(panel.examples))
        for start in range(0, len(all_indices), batch_size * 2):
            indices = all_indices[start : start + batch_size * 2]
            _, encoded = collate(indices.tolist())
            logits = model(**encoded.to(device))["routing_logits"]
            probability_parts.append(logits.float().softmax(dim=1).cpu().numpy())
    return ModernBERTPOCResult(
        model=model,
        tokenizer=tokenizer,
        probabilities=np.concatenate(probability_parts),
        oracle_targets=targets,
        fallback_index=fallback_index,
        history=pd.DataFrame(history),
        training_seconds=time.perf_counter() - started,
    )


def export_modernbert_oracle_poc(
    result: ModernBERTPOCResult,
    model_names: tuple[str, ...],
    output_dir: str | Path,
    *,
    selected_threshold: float,
    router_active: bool,
    minimum_predicted_savings: float = 0.02,
    config: RouterConfig = DEFAULT_CONFIG,
) -> Path:
    """Save the LoRA adapter, routing head, tokenizer, and training metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.encoder.save_pretrained(output_dir / "lora_adapter")
    torch.save(
        result.model.routing_head.state_dict(), output_dir / "routing_head.pt"
    )
    result.tokenizer.save_pretrained(output_dir / "tokenizer")
    result.history.to_csv(output_dir / "training_history.csv", index=False)
    manifest = {
        "schema_version": 1,
        "router": "ModernBERT oracle imitation",
        "encoder_repo": config.encoder_repo,
        "encoder_revision": config.encoder_revision,
        "router_max_input_tokens": config.max_input_tokens,
        "model_names": model_names,
        "fallback_model": model_names[result.fallback_index],
        "selected_threshold": selected_threshold,
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
