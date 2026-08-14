"""ModernBERT optimization with deployment-aligned checkpoint selection."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

from llm_router.config import RouterConfig
from llm_router.models.modernbert_router import DecisionAlignedRouter
from llm_router.utils.calibration import out_of_fold_platt
from llm_router.utils.data import RouterData
from llm_router.utils.routing import search_selector


@dataclass(frozen=True)
class TensorTargets:
    safety: torch.Tensor
    safety_weight: torch.Tensor
    opportunity: torch.Tensor
    latency: torch.Tensor
    tokens: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_epoch: int
    training_time_s: float
    trainable_parameters: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("ModernBERT training requires a CUDA GPU.")
    return (
        torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    )


def make_targets(data: RouterData) -> TensorTargets:
    return TensorTargets(
        safety=torch.tensor(data.replacement_safe.astype(float), dtype=torch.float32),
        safety_weight=torch.tensor(data.safety_example_weight, dtype=torch.float32),
        opportunity=torch.tensor(data.opportunity_gain, dtype=torch.float32),
        latency=torch.tensor(np.log1p(data.latency), dtype=torch.float32),
        tokens=torch.tensor(np.log1p(data.tokens), dtype=torch.float32),
    )


def make_encoder(
    tokenizer: PreTrainedTokenizerBase,
    text: np.ndarray,
    config: RouterConfig,
) -> Callable:
    def encode_indices(indices: list[int] | np.ndarray) -> tuple[torch.Tensor, object]:
        normalized = [int(index) for index in indices]
        encoded = tokenizer(
            [f"classification: {text[index]}" for index in normalized],
            padding=True,
            truncation=True,
            max_length=config.max_input_tokens,
            return_tensors="pt",
        )
        return torch.tensor(normalized, dtype=torch.long), encoded

    return encode_indices


def router_batch_loss(
    outputs: dict[str, torch.Tensor],
    indices: torch.Tensor,
    targets: TensorTargets,
    config: RouterConfig,
    device: str,
) -> tuple[torch.Tensor, ...]:
    safety_target = targets.safety[indices].to(device)
    safety_weight = targets.safety_weight[indices].to(device)
    opportunity = targets.opportunity[indices].to(device)
    latency_target = targets.latency[indices].to(device)
    token_target = targets.tokens[indices].to(device)

    element_bce = F.binary_cross_entropy_with_logits(
        outputs["safety_logits"].float(), safety_target, reduction="none"
    )
    safety_loss = (element_bce * safety_weight).mean()
    latency_loss = F.smooth_l1_loss(outputs["latency_log"].float(), latency_target)
    token_loss = F.smooth_l1_loss(outputs["token_log"].float(), token_target)
    opportunity_margin_loss = (
        F.softplus(config.safety_logit_margin - outputs["safety_logits"].float())
        * opportunity
    ).mean()
    total = (
        config.safety_loss_weight * safety_loss
        + config.latency_loss_weight * latency_loss
        + config.token_loss_weight * token_loss
        + config.opportunity_margin_loss_weight * opportunity_margin_loss
    )
    return total, safety_loss, latency_loss, token_loss, opportunity_margin_loss


@torch.inference_mode()
def evaluate_loss(
    model: DecisionAlignedRouter,
    indices: np.ndarray,
    encode_indices: Callable,
    targets: TensorTargets,
    config: RouterConfig,
    compute_dtype: torch.dtype,
    device: str = "cuda",
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        [int(index) for index in indices],
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=encode_indices,
    )
    totals = np.zeros(5, dtype=float)
    examples = 0
    for batch_indices, encoded in loader:
        encoded = encoded.to(device)
        with torch.autocast("cuda", dtype=compute_dtype):
            losses = router_batch_loss(
                model(**encoded), batch_indices, targets, config, device
            )
        examples += len(batch_indices)
        totals += len(batch_indices) * np.array(
            [float(loss.detach()) for loss in losses]
        )
    return totals / examples


@torch.inference_mode()
def predict_indices(
    model: DecisionAlignedRouter,
    indices: np.ndarray,
    encode_indices: Callable,
    config: RouterConfig,
    compute_dtype: torch.dtype,
    measure_batch_one_overhead: bool = False,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Predict in index order; optional timing includes tokenization and forward."""
    model.eval()
    indices = np.asarray(indices, dtype=int)
    safety_parts, latency_parts, token_parts = [], [], []
    overhead_s = np.zeros(len(indices), dtype=float)

    if measure_batch_one_overhead:
        for index in indices[: config.warmup_prompts]:
            _, encoded = encode_indices([int(index)])
            with torch.autocast("cuda", dtype=compute_dtype):
                model(**encoded.to(device))
        torch.cuda.synchronize()

        for position, index in enumerate(indices):
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, encoded = encode_indices([int(index)])
            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(**encoded.to(device))
            torch.cuda.synchronize()
            overhead_s[position] = time.perf_counter() - started
            safety_parts.append(outputs["safety_logits"].float().cpu().numpy())
            latency_parts.append(outputs["latency_log"].float().cpu().numpy())
            token_parts.append(outputs["token_log"].float().cpu().numpy())
    else:
        loader = DataLoader(
            indices.tolist(),
            batch_size=config.batch_size * 2,
            shuffle=False,
            collate_fn=encode_indices,
        )
        for _, encoded in loader:
            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(**encoded.to(device))
            safety_parts.append(outputs["safety_logits"].float().cpu().numpy())
            latency_parts.append(outputs["latency_log"].float().cpu().numpy())
            token_parts.append(outputs["token_log"].float().cpu().numpy())

    return {
        "indices": indices,
        "safety_logits": np.concatenate(safety_parts),
        "latency": np.maximum(1e-6, np.expm1(np.concatenate(latency_parts))),
        "tokens": np.maximum(1.0, np.expm1(np.concatenate(token_parts))),
        "overhead_s": overhead_s,
    }


def train_router(
    model: DecisionAlignedRouter,
    tokenizer: PreTrainedTokenizerBase,
    data: RouterData,
    config: RouterConfig,
    compute_dtype: torch.dtype,
    reports_dir: str | Path | None = None,
    device: str = "cuda",
) -> TrainingResult:
    """Train and rank every epoch by its calibrated deployed behavior."""
    targets = make_targets(data)
    encode_indices = make_encoder(tokenizer, data.text, config)
    train_loader = DataLoader(
        data.train_indices.tolist(),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=encode_indices,
    )
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": config.lora_lr},
            {"params": head_parameters, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_steps = updates_per_epoch * config.max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * config.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype == torch.float16)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    history = []
    best_key = (-1, -np.inf, -np.inf)
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_totals = np.zeros(5, dtype=float)
        train_examples = 0
        accumulation_count = 0

        for step, (batch_indices, encoded) in enumerate(train_loader, 1):
            encoded = encoded.to(device)
            accumulation_count += 1
            with torch.autocast("cuda", dtype=compute_dtype):
                losses = router_batch_loss(
                    model(**encoded), batch_indices, targets, config, device
                )
                scaled_loss = losses[0] / config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                accumulation_count == config.gradient_accumulation
                or step == len(train_loader)
            )
            if should_step:
                if accumulation_count != config.gradient_accumulation:
                    correction = config.gradient_accumulation / accumulation_count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0

            batch_size = len(batch_indices)
            train_examples += batch_size
            train_totals += batch_size * np.array(
                [float(loss.detach()) for loss in losses]
            )

        validation_losses = evaluate_loss(
            model,
            data.validation_indices,
            encode_indices,
            targets,
            config,
            compute_dtype,
            device,
        )
        prediction = predict_indices(
            model,
            data.validation_indices,
            encode_indices,
            config,
            compute_dtype,
            measure_batch_one_overhead=True,
            device=device,
        )
        oof_probability, _ = out_of_fold_platt(
            prediction["safety_logits"],
            data.replacement_safe[data.validation_indices],
            data.nonfallback_names,
            config.platt_folds,
            config.seed,
        )
        _, epoch_best = search_selector(prediction, oof_probability, data, config)
        feasible = epoch_best is not None
        if feasible:
            route_score = float(epoch_best.latency_reduction)
            quality_score = -float(epoch_best.quality_loss_rate)
        else:
            route_score = -float(validation_losses[0])
            quality_score = -float(validation_losses[1])
        candidate_key = (int(feasible), route_score, quality_score)
        record = {
            "epoch": epoch,
            "train_loss": train_totals[0] / train_examples,
            "validation_loss": validation_losses[0],
            "validation_safety_loss": validation_losses[1],
            "validation_latency_loss": validation_losses[2],
            "validation_token_loss": validation_losses[3],
            "validation_opportunity_margin_loss": validation_losses[4],
            "validation_route_feasible": feasible,
            "validation_route_score": route_score,
            "validation_quality_retention": (
                np.nan if epoch_best is None else float(epoch_best.quality_retention)
            ),
            "validation_fallback_usage": (
                np.nan if epoch_best is None else float(epoch_best.fallback_usage)
            ),
            "mean_router_overhead_ms": float(1000 * prediction["overhead_s"].mean()),
        }
        history.append(record)
        printable = {
            key: round(value, 4)
            if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else value
            for key, value in record.items()
        }
        print(printable)

        if candidate_key > best_key:
            best_key = candidate_key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name in trainable_names
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                epoch >= config.min_epochs
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                print(
                    f"Early stopping after epoch {epoch}; restoring epoch {best_epoch}."
                )
                break

    training_time_s = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")
    model.load_state_dict(best_state, strict=False)
    history_frame = pd.DataFrame(history)
    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        history_frame.to_csv(reports_dir / "training_history_v4.csv", index=False)
    return TrainingResult(
        history=history_frame,
        best_epoch=best_epoch,
        training_time_s=training_time_s,
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
