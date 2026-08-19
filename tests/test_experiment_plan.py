import numpy as np
import pytest
import torch

from llm_router.experiment_plan import (
    ANALYTICAL_SCENARIO_AS_OF,
    FROZEN_EXPERIMENT_PLAN,
    FROZEN_TRAINING_SETUPS,
    frozen_setup_dicts,
    get_experiment_run,
)
from llm_router.hybrid_inference import (
    HybridModernBERTRouterRuntime,
    select_analytical_route,
)
from llm_router.public_benchmark import EconomicsScenario, ModelProfile
from llm_router.router_overhead import summarize_router_overhead_samples


def test_frozen_plan_contains_three_random_and_three_dataset_ood_runs():
    assert ANALYTICAL_SCENARIO_AS_OF == "2026-08-19"
    assert len(FROZEN_EXPERIMENT_PLAN) == 6
    assert {run.seed for run in FROZEN_EXPERIMENT_PLAN} == {42, 43, 44}
    assert sum(run.split_mode == "random" for run in FROZEN_EXPERIMENT_PLAN) == 3
    assert sum(run.split_mode == "dataset_ood" for run in FROZEN_EXPERIMENT_PLAN) == 3
    assert get_experiment_run("random_seed_43").seed == 43
    with pytest.raises(ValueError, match="Unknown RUN_ID"):
        get_experiment_run("random_seed_99")


def test_frozen_training_setups_match_the_schema_v5_contract():
    setups = frozen_setup_dicts()
    assert tuple(setups) == tuple(setup.name for setup in FROZEN_TRAINING_SETUPS)
    assert setups["hybrid_r4"]["oracle_auxiliary_weight"] == 0.25
    assert setups["safety_only_r4"]["oracle_auxiliary_weight"] == 0.0
    assert setups["hybrid_r4_dataset_balanced"]["dataset_balanced_sampling"]


def test_analytical_selector_requires_safety_and_speed_and_fails_closed():
    arguments = {
        "model_names": ("fast", "fallback"),
        "fallback_model": "fallback",
        "analytical_latency_s": {"fast": 0.70, "fallback": 1.00},
        "threshold": 0.91,
        "minimum_predicted_savings": 0.02,
    }
    selected, eligible, _ = select_analytical_route(
        safety_probabilities={"fast": 0.92}, router_active=True, **arguments
    )
    assert selected == "fast"
    assert eligible == {"fast": True, "fallback": True}

    selected, eligible, _ = select_analytical_route(
        safety_probabilities={"fast": 0.90}, router_active=True, **arguments
    )
    assert selected == "fallback"
    assert not eligible["fast"]

    selected, _, reason = select_analytical_route(
        safety_probabilities={"fast": 0.99}, router_active=False, **arguments
    )
    assert selected == "fallback"
    assert "fail closed" in reason


def test_router_overhead_summary_keeps_candidate_latency_analytical():
    summary = summarize_router_overhead_samples(
        np.array([0.004, 0.006, 0.008]),
        np.array([0.003, 0.004, 0.005]),
        device="cpu",
        warmup_requests=2,
        max_input_tokens=512,
    )
    assert summary["end_to_end_ms"]["p50"] == pytest.approx(6.0)
    assert summary["model_only_ms"]["p50"] == pytest.approx(4.0)
    assert summary["candidate_latency_method"].startswith("analytical")
    assert not summary["measurement_affects_frozen_policy"]


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return _FakeBatch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )


class _FakeHybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **inputs):
        return {"safety_logits": torch.zeros((1, 1)) + self.anchor}


def test_demo_runtime_measures_router_but_keeps_candidate_latency_analytical():
    scenario = EconomicsScenario(
        name="test",
        as_of="2026-08-19",
        profiles=(
            ModelProfile(
                "fast",
                parameters_billions=1.0,
                architecture="autoregressive",
            ),
            ModelProfile(
                "fallback",
                parameters_billions=2.0,
                architecture="autoregressive",
            ),
        ),
        latency_method="analytical",
    )
    runtime = HybridModernBERTRouterRuntime(
        model=_FakeHybridModel(),
        tokenizer=_FakeTokenizer(),
        model_names=("fast", "fallback"),
        fallback_model="fallback",
        calibration_parameters={"fast": (0.0, 2.944439)},
        selected_threshold=0.91,
        router_active=True,
        minimum_predicted_savings=0.02,
        max_input_tokens=512,
        scenario=scenario,
        device="cpu",
    )
    decision = runtime.route("Explain compound interest.", 100)
    assert decision.selected_model == "fast"
    assert decision.safety_probabilities["fast"] == pytest.approx(0.95)
    assert decision.analytical_candidate_latency_s["fast"] < (
        decision.analytical_candidate_latency_s["fallback"]
    )
    assert decision.measured_router_overhead_s >= 0
