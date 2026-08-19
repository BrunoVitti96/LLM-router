"""Train the hybrid ModernBERT safety router with auxiliary oracle alignment."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from llm_router.config import DEFAULT_CONFIG, RouterConfig
from llm_router.models.modernbert_router import (
    HybridModernBERTRouter,
    build_hybrid_router,
)
from llm_router.oracle import hybrid_routing_loss, replacement_safety_targets
from llm_router.public_benchmark import BenchmarkPanel, BenchmarkSplit
from llm_router.utils.calibration import calibrate_with_validation
from llm_router.utils.training import seed_everything


@dataclass(frozen=True)
class ModernBERTHybridPOCResult:
    model: HybridModernBERTRouter
    tokenizer: object
    safety_probabilities: np.ndarray
    raw_safety_probabilities: np.ndarray
    safety_logits: np.ndarray
    calibration_parameters: dict[str, tuple[float, float]]
    calibration_diagnostics: pd.DataFrame
    safety_pos_weights: dict[str, float]
    fallback_index: int
    nonfallback_indices: np.ndarray
    history: pd.DataFrame
    input_diagnostics: dict[str, float | int]
    router_input_lengths: np.ndarray
    router_was_truncated: np.ndarray
    encoder_learning_rate: float
    head_learning_rate: float
    minimum_epochs: int
    early_stopping_patience: int | None
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    dataset_balanced_sampling: bool
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
    head_learning_rate: float | None = None,
    minimum_epochs: int = 2,
    early_stopping_patience: int | None = 2,
    quality_epsilon: float = 0.0,
    safety_loss_weight: float = 1.0,
    oracle_auxiliary_weight: float = 0.25,
    dataset_balanced_sampling: bool = False,
    device: str | None = None,
    progress_callback: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> ModernBERTHybridPOCResult:
    """Train safety estimates while using the oracle only as an auxiliary task.

    Candidate LLMs are never loaded. Benchmark outcomes supervise replacement
    safety; ``panel.latency`` must contain the deterministic analytical proxy.
    Dataset-balanced sampling gives every dataset equal expected draw probability
    and recomputes class weights under that distribution. ``progress_callback``
    receives one structured dictionary after every epoch for notebook logging.
    """

    if head_learning_rate is None:
        head_learning_rate = learning_rate * 2
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or head_learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if minimum_epochs <= 0:
        raise ValueError("minimum_epochs must be positive.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive or None.")
    if safety_loss_weight <= 0 or oracle_auxiliary_weight < 0:
        raise ValueError(
            "Loss weights must be non-negative and safety must be positive."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    compute_dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    )
    seed_everything(config.seed)
    fallback_index = int(panel.score[split.train].mean(axis=0).argmax())
    nonfallback_indices = np.array(
        [index for index in range(len(panel.models)) if index != fallback_index],
        dtype=int,
    )
    # Exact dataset names are deliberately excluded: random splits should test
    # prompt-level interpolation, while dataset-OOD runs should not learn brittle
    # identifiers that disappear at deployment.
    texts = np.array(
        [
            f"[PROMPT_TOKENS={int(row.prompt_tokens)}] {row.prompt}"
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

    # Measure the router's own tokenization rather than approximating truncation
    # with token counts recorded by a different candidate tokenizer.
    router_input_lengths: list[int] = []
    for start in range(0, len(texts), 64):
        diagnostic_batch = tokenizer(
            [f"classification: {text}" for text in texts[start : start + 64]],
            padding=False,
            truncation=False,
        )
        attention_mask = diagnostic_batch["attention_mask"]
        if isinstance(attention_mask, torch.Tensor):
            router_input_lengths.extend(
                attention_mask.sum(dim=1).to(torch.int64).tolist()
            )
        else:
            router_input_lengths.extend(sum(mask) for mask in attention_mask)
    input_lengths = np.asarray(router_input_lengths, dtype=int)
    truncated = input_lengths > config.max_input_tokens
    input_diagnostics: dict[str, float | int] = {
        "examples": len(input_lengths),
        "max_input_tokens": int(config.max_input_tokens),
        "truncated_examples": int(truncated.sum()),
        "truncation_rate": float(truncated.mean()),
        "router_tokens_p50": float(np.quantile(input_lengths, 0.50)),
        "router_tokens_p95": float(np.quantile(input_lengths, 0.95)),
        "router_tokens_max": int(input_lengths.max()),
    }

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

    sampling_generator = torch.Generator().manual_seed(config.seed)
    train_sampling_weight = np.ones(len(split.train), dtype=float)
    sampler = None
    if dataset_balanced_sampling:
        train_datasets = panel.examples.dataset.iloc[split.train].astype(str)
        dataset_counts = train_datasets.value_counts()
        train_sampling_weight = train_datasets.map(
            lambda name: 1.0 / dataset_counts[name]
        ).to_numpy(dtype=float, copy=True)
        train_sampling_weight *= (
            len(train_sampling_weight) / train_sampling_weight.sum()
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(train_sampling_weight, dtype=torch.double),
            num_samples=len(split.train),
            replacement=True,
            generator=sampling_generator,
        )
    train_loader = DataLoader(
        split.train.tolist(),
        batch_size=batch_size,
        shuffle=not dataset_balanced_sampling,
        sampler=sampler,
        generator=sampling_generator,
        collate_fn=collate,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("encoder.")
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": head_learning_rate},
        ],
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
    train_safety_target = replacement_safety_targets(
        panel.score[split.train],
        fallback_index,
        nonfallback_indices,
        quality_epsilon,
    )
    positive = (train_safety_target * train_sampling_weight[:, None]).sum(axis=0)
    negative = ((~train_safety_target) * train_sampling_weight[:, None]).sum(axis=0)
    safety_pos_weight = torch.tensor(
        np.clip(negative / np.maximum(positive, 1), 0.10, 10.0),
        dtype=torch.float32,
        device=device,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    stopped_early = False
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
            safety_pos_weight=safety_pos_weight,
        )

    skipped_optimizer_steps = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_total = 0.0
        train_safety = 0.0
        train_oracle_auxiliary = 0.0
        train_examples = 0
        for indices, encoded in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, compute_dtype):
                loss, parts = compute_loss(indices, encoded)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = scaler.is_enabled() and (
                scaler.get_scale() < scale_before_step
            )
            if step_was_skipped:
                skipped_optimizer_steps += 1
            else:
                scheduler.step()
            train_total += float(loss.detach()) * len(indices)
            train_safety += float(parts["safety"].detach()) * len(indices)
            train_oracle_auxiliary += float(parts["oracle_auxiliary"].detach()) * len(
                indices
            )
            train_examples += len(indices)

        train_seconds = time.perf_counter() - epoch_started
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
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total / train_examples,
                "train_safety_loss": train_safety / train_examples,
                "train_oracle_auxiliary_loss": (
                    train_oracle_auxiliary / train_examples
                ),
                "validation_total_loss": validation_loss,
                "validation_safety_loss": (
                    validation_totals["safety"] / len(split.validation)
                ),
                "validation_oracle_auxiliary_loss": (
                    validation_totals["oracle_auxiliary"] / len(split.validation)
                ),
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "epoch_seconds": epoch_seconds,
                "train_examples_per_second": train_examples / max(train_seconds, 1e-9),
                "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )
        improved = validation_loss < best_validation_loss
        should_stop = False
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name in trainable_names
            }
        elif (
            early_stopping_patience is not None
            and epoch >= minimum_epochs
            and epoch - best_epoch >= early_stopping_patience
        ):
            should_stop = True
        if progress_callback is not None:
            progress_callback(
                {
                    **history[-1],
                    "is_best_epoch": improved,
                    "best_epoch_so_far": best_epoch,
                    "will_stop_early": should_stop,
                }
            )
        if should_stop:
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("ModernBERT training produced no checkpoint.")
    model.load_state_dict(best_state, strict=False)
    model.eval()
    safety_logit_parts = []
    with torch.inference_mode():
        all_indices = np.arange(len(panel.examples))
        for start in range(0, len(all_indices), batch_size * 2):
            indices = all_indices[start : start + batch_size * 2]
            _, encoded = collate(indices.tolist())
            logits = model(**encoded.to(device))["safety_logits"]
            safety_logit_parts.append(logits.float().cpu().numpy())
    safety_logits = np.concatenate(safety_logit_parts)
    raw_alternative_probability = 1 / (1 + np.exp(-np.clip(safety_logits, -40, 40)))
    calibration_target = np.zeros_like(safety_logits, dtype=float)
    calibration_target[split.validation] = replacement_safety_targets(
        panel.score[split.validation],
        fallback_index,
        nonfallback_indices,
        quality_epsilon,
    )
    nonfallback_models = tuple(panel.models[index] for index in nonfallback_indices)
    safety_pos_weights = {
        candidate: float(safety_pos_weight[position].detach().cpu())
        for position, candidate in enumerate(nonfallback_models)
    }
    calibrated_alternative_probability, calibration_parameters, diagnostics = (
        calibrate_with_validation(
            safety_logits,
            calibration_target,
            split.validation,
            nonfallback_models,
            folds=config.platt_folds,
            seed=config.seed,
        )
    )
    raw_safety_probabilities = np.ones_like(panel.score, dtype=float)
    raw_safety_probabilities[:, nonfallback_indices] = raw_alternative_probability
    safety_probabilities = np.ones_like(panel.score, dtype=float)
    safety_probabilities[:, nonfallback_indices] = calibrated_alternative_probability
    return ModernBERTHybridPOCResult(
        model=model,
        tokenizer=tokenizer,
        safety_probabilities=safety_probabilities,
        raw_safety_probabilities=raw_safety_probabilities,
        safety_logits=safety_logits,
        calibration_parameters=calibration_parameters,
        calibration_diagnostics=pd.DataFrame(diagnostics),
        safety_pos_weights=safety_pos_weights,
        fallback_index=fallback_index,
        nonfallback_indices=nonfallback_indices,
        history=pd.DataFrame(history),
        input_diagnostics=input_diagnostics,
        router_input_lengths=input_lengths,
        router_was_truncated=truncated,
        encoder_learning_rate=learning_rate,
        head_learning_rate=head_learning_rate,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        dataset_balanced_sampling=dataset_balanced_sampling,
        training_seconds=time.perf_counter() - started,
    )


def collect_runtime_metadata() -> dict[str, object]:
    """Collect enough environment identity to explain reproducibility failures."""

    project_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for package in (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pandas",
        "peft",
        "scikit-learn",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "repository_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def export_modernbert_hybrid_poc(
    result: ModernBERTHybridPOCResult,
    model_names: tuple[str, ...],
    output_dir: str | Path,
    *,
    selected_threshold: float,
    router_active: bool,
    poc_passed: bool,
    failure_reasons: tuple[str, ...] = (),
    minimum_predicted_savings: float = 0.02,
    validation_quality_margin: float = 0.0,
    minimum_macro_quality_retention: float | None = None,
    maximum_quality_loss_rate_ucl: float | None = None,
    minimum_routed_safety_precision_lcb: float | None = None,
    minimum_guarded_dataset_quality_retention_lcb: float | None = None,
    minimum_guarded_dataset_prompts: int = 0,
    conservative_router_overhead_s: float = 0.0,
    minimum_consecutive_feasible_thresholds: int = 2,
    benchmark_fingerprint: str = "",
    setup_name: str = "default",
    safety_loss_weight: float = 1.0,
    oracle_auxiliary_weight: float = 0.25,
    router_overhead_benchmark: dict[str, object] | None = None,
    config: RouterConfig = DEFAULT_CONFIG,
) -> Path:
    """Save the best checkpoint, policy, and optional router-only timing evidence.

    ``router_overhead_benchmark`` may contain measured ModernBERT latency.  It is
    diagnostic metadata only: candidate-model latency remains analytical and the
    timing does not retroactively alter the frozen validation policy.
    """

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
    result.calibration_diagnostics.to_csv(
        output_dir / "calibration_diagnostics.csv", index=False
    )
    (output_dir / "input_diagnostics.json").write_text(
        json.dumps(result.input_diagnostics, indent=2), encoding="utf-8"
    )
    nonfallback_models = tuple(
        model_names[index] for index in result.nonfallback_indices
    )
    manifest = {
        "schema_version": 5,
        "router": "ModernBERT hybrid safety router",
        "deployed_prediction": "fallback-relative replacement safety",
        "oracle_role": "training-only auxiliary loss",
        "encoder_repo": config.encoder_repo,
        "encoder_revision": config.encoder_revision,
        "router_max_input_tokens": config.max_input_tokens,
        "model_names": model_names,
        "fallback_model": model_names[result.fallback_index],
        "nonfallback_models": nonfallback_models,
        "input_features": "prompt text and prompt-token count; no dataset identifier",
        "safety_definition": "candidate_quality >= fallback_quality - epsilon",
        "loss": {
            "safety_bce_weight": safety_loss_weight,
            "safety_positive_weights": result.safety_pos_weights,
            "oracle_auxiliary_weight": oracle_auxiliary_weight,
        },
        "selected_setup": setup_name,
        "selected_safety_threshold": selected_threshold,
        "calibration": {
            "method": "per-candidate Platt scaling",
            "validation_probabilities": "out-of-fold",
            "deployment_parameters": {
                candidate: {"slope": values[0], "intercept": values[1]}
                for candidate, values in result.calibration_parameters.items()
            },
        },
        "minimum_predicted_savings": minimum_predicted_savings,
        "validation_quality_margin": validation_quality_margin,
        "policy_gates": {
            "minimum_macro_quality_retention": minimum_macro_quality_retention,
            "maximum_quality_loss_rate_ucl": maximum_quality_loss_rate_ucl,
            "minimum_routed_safety_precision_lcb": (
                minimum_routed_safety_precision_lcb
            ),
            "minimum_guarded_dataset_quality_retention_lcb": (
                minimum_guarded_dataset_quality_retention_lcb
            ),
            "minimum_guarded_dataset_prompts": minimum_guarded_dataset_prompts,
            "conservative_router_overhead_s": conservative_router_overhead_s,
            "minimum_consecutive_feasible_thresholds": (
                minimum_consecutive_feasible_thresholds
            ),
        },
        "validation_router_active": router_active,
        "single_run_passed": poc_passed,
        "study_status": (
            "single_run_passed_requires_multi_seed_and_dataset_ood_confirmation"
            if poc_passed
            else "single_run_failed"
        ),
        # Compatibility alias for schema-v4 artifact readers.
        "poc_passed": poc_passed,
        "failure_reasons": failure_reasons,
        "router_active": router_active and poc_passed,
        "deployment_enabled": router_active and poc_passed,
        "latency_source": "analytical_model_profile_and_prompt_tokens",
        "candidate_inference_used_for_latency": False,
        "router_overhead_benchmark": router_overhead_benchmark,
        "optimization": {
            "encoder_learning_rate": result.encoder_learning_rate,
            "head_learning_rate": result.head_learning_rate,
            "minimum_epochs": result.minimum_epochs,
            "early_stopping_patience": result.early_stopping_patience,
            "best_epoch": result.best_epoch,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "dataset_balanced_sampling": result.dataset_balanced_sampling,
        },
        "input_diagnostics": result.input_diagnostics,
        "training_seconds": result.training_seconds,
        "benchmark_fingerprint_sha256": benchmark_fingerprint,
        "reproducibility": collect_runtime_metadata(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
