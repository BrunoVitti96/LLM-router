"""Experiment identity plus report and reconstructable artifact export."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from llm_router.config import RouterConfig
from llm_router.models.modernbert_router import DecisionAlignedRouter
from llm_router.utils.data import Evidence, RouterData
from llm_router.utils.evaluation import EvaluationResult
from llm_router.utils.training import TrainingResult


@dataclass(frozen=True)
class ExperimentPaths:
    reports: Path
    artifacts: Path


def prepare_experiment(
    evidence: Evidence,
    config: RouterConfig,
    gpu: str,
    dtype: str,
    packages: dict[str, str],
) -> tuple[ExperimentPaths, dict, str]:
    paths = ExperimentPaths(
        reports=evidence.root / "reports_v4",
        artifacts=evidence.root / "artifacts_v4",
    )
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    contract = config.contract(evidence.fingerprint, gpu, dtype)
    contract["packages"] = packages
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    (paths.reports / "synchronized_contract_v4.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return paths, contract, fingerprint


def export_experiment(
    *,
    evidence: Evidence,
    data: RouterData,
    config: RouterConfig,
    paths: ExperimentPaths,
    contract: dict,
    router_fingerprint: str,
    model: DecisionAlignedRouter,
    tokenizer: object,
    training: TrainingResult,
    evaluation: EvaluationResult,
    router_load_time_s: float,
) -> Path:
    evaluation.selector_search.to_csv(
        paths.reports / "selector_search_v4.csv", index=False
    )
    evaluation.evaluation.to_csv(paths.reports / "evaluation_v4.csv")
    evaluation.report.to_parquet(
        paths.reports / "test_decisions_v4.parquet", index=False
    )
    evaluation.calibration_diagnostics.to_csv(
        paths.reports / "calibration_diagnostics_v4.csv"
    )
    evaluation.latency_diagnostics.to_csv(paths.reports / "latency_diagnostics_v4.csv")
    evaluation.confidence_intervals.to_csv(
        paths.reports / "paired_bootstrap_intervals_v4.csv"
    )

    artifact_dir = paths.artifacts / (
        f"{evidence.fingerprint[:16]}__{router_fingerprint[:12]}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(artifact_dir / "lora_adapter")
    head_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("encoder.")
    }
    torch.save(head_state, artifact_dir / "router_heads.pt")
    tokenizer.save_pretrained(artifact_dir / "tokenizer")

    manifest = {
        **contract,
        "artifact_schema_version": 1,
        "router_fingerprint": router_fingerprint,
        "model_names": config.model_names,
        "fallback_model": data.fallback_name,
        "nonfallback_models": data.nonfallback_names,
        "router_active": evaluation.router_active,
        "best_epoch": training.best_epoch,
        "selected_thresholds": dict(
            zip(data.nonfallback_names, evaluation.selected_thresholds.tolist())
        ),
        "selected_latency_blend": evaluation.selected_latency_blend,
        "platt_parameters": {
            candidate: {"slope": parameters[0], "intercept": parameters[1]}
            for candidate, parameters in zip(
                data.nonfallback_names, evaluation.platt_parameters
            )
        },
        "task_model_median_latency": {
            task: {
                model_name: float(data.task_model_median.loc[task, model_name])
                for model_name in config.model_names
            }
            for task in config.tasks
        },
        "router_input_template": (
            "[TASK={task}] [SUBJECT={subject}] [CHOICES={num_choices}] "
            "[LENGTH_BIN={length_bin}] {prompt}"
        ),
        "router_load_time_s": router_load_time_s,
        "router_training_time_s": training.training_time_s,
    }
    (artifact_dir / "router_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return artifact_dir
