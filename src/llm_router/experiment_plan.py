"""Frozen run presets for the schema-v5 validation study.

The presets prevent a Colab rerun from silently changing the split mode, seed,
or training ablations after a result has been seen.  Random-split feasibility
and dataset-OOD generalization remain separate claims and separate artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

ANALYTICAL_SCENARIO_AS_OF = "2026-08-19"


@dataclass(frozen=True)
class ExperimentRunSpec:
    """Identity of one independently downloadable Colab run."""

    run_id: str
    split_mode: str
    seed: int
    purpose: str

    def validate(self) -> None:
        if self.split_mode not in {"random", "dataset_ood"}:
            raise ValueError(f"Unsupported split mode: {self.split_mode!r}")
        expected = f"{self.split_mode}_seed_{self.seed}"
        if self.run_id != expected:
            raise ValueError(
                f"Run ID {self.run_id!r} must equal the frozen identity {expected!r}."
            )


@dataclass(frozen=True)
class TrainingSetupSpec:
    """One validation-only training ablation shared by every run preset."""

    name: str
    lora_r: int
    lora_alpha: int
    oracle_auxiliary_weight: float
    dataset_balanced_sampling: bool


FROZEN_EXPERIMENT_PLAN = (
    ExperimentRunSpec(
        "random_seed_42", "random", 42, "completed feasibility baseline"
    ),
    ExperimentRunSpec(
        "random_seed_43", "random", 43, "random-split stability confirmation"
    ),
    ExperimentRunSpec(
        "random_seed_44", "random", 44, "random-split stability confirmation"
    ),
    ExperimentRunSpec(
        "dataset_ood_seed_42", "dataset_ood", 42, "unseen-dataset stress test"
    ),
    ExperimentRunSpec(
        "dataset_ood_seed_43", "dataset_ood", 43, "unseen-dataset stress test"
    ),
    ExperimentRunSpec(
        "dataset_ood_seed_44", "dataset_ood", 44, "unseen-dataset stress test"
    ),
)

FROZEN_TRAINING_SETUPS = (
    TrainingSetupSpec("hybrid_r4", 4, 8, 0.25, False),
    TrainingSetupSpec("safety_only_r4", 4, 8, 0.0, False),
    TrainingSetupSpec("hybrid_r4_dataset_balanced", 4, 8, 0.25, True),
)


def get_experiment_run(run_id: str) -> ExperimentRunSpec:
    """Return and validate one named run from the frozen study plan."""

    matches = [spec for spec in FROZEN_EXPERIMENT_PLAN if spec.run_id == run_id]
    if not matches:
        available = ", ".join(spec.run_id for spec in FROZEN_EXPERIMENT_PLAN)
        raise ValueError(f"Unknown RUN_ID {run_id!r}. Choose one of: {available}.")
    matches[0].validate()
    return matches[0]


def frozen_setup_dicts() -> dict[str, dict[str, int | float | bool]]:
    """Return notebook-friendly copies of the immutable setup definitions."""

    return {
        setup.name: {
            "lora_r": setup.lora_r,
            "lora_alpha": setup.lora_alpha,
            "oracle_auxiliary_weight": setup.oracle_auxiliary_weight,
            "dataset_balanced_sampling": setup.dataset_balanced_sampling,
        }
        for setup in FROZEN_TRAINING_SETUPS
    }
