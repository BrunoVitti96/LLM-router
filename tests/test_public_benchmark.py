import json

import numpy as np

from llm_router.public_benchmark import (
    EconomicsScenario,
    ModelProfile,
    load_llmrouterbench,
    make_complete_panel,
    run_public_benchmark,
    simulate_economics,
    split_benchmark,
)

MODELS = ("fast", "specialist", "strong")


def write_result(root, dataset, model, records):
    path = root / "results" / "bench" / dataset / "test" / model / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def synthetic_release(tmp_path):
    for dataset_number, dataset in enumerate(("math", "code", "knowledge")):
        for model in MODELS:
            records = []
            for index in range(15):
                easy = index % 3 != 0
                if model == "strong":
                    score = 1.0
                    completion = 30
                elif model == "fast":
                    score = float(easy)
                    completion = 10
                else:
                    score = float(dataset_number == 1 or easy)
                    completion = 18
                records.append(
                    {
                        "index": index,
                        "origin_query": (
                            f"{dataset} {'easy-pattern' if easy else 'hard-pattern'} "
                            f"question {index}"
                        ),
                        "prompt": f"formatted {index}",
                        "prediction": "answer",
                        "ground_truth": "answer",
                        "score": score,
                        "prompt_tokens": 20 + index,
                        "completion_tokens": completion,
                        "cost": None,
                    }
                )
            write_result(tmp_path, dataset, model, records)
    return tmp_path


def scenario():
    return EconomicsScenario(
        name="test",
        as_of="2026-08-14",
        latency_method="throughput",
        network_s=0.1,
        router_overhead_s=0.01,
        profiles=(
            ModelProfile("fast", 0.1, 0.2, 0.2, 100.0),
            ModelProfile("specialist", 0.2, 0.4, 0.3, 80.0),
            ModelProfile("strong", 1.0, 2.0, 0.5, 40.0),
        ),
    )


def test_load_and_simulate_llmrouterbench_release(tmp_path):
    root = synthetic_release(tmp_path)
    records = load_llmrouterbench(root, models=MODELS)
    assert len(records) == 3 * 15 * 3

    simulated = simulate_economics(records, scenario())
    fast = simulated.loc[simulated.model.eq("fast")].iloc[0]
    expected_cost = (fast.prompt_tokens * 0.1 + fast.completion_tokens * 0.2) / 1e6
    assert np.isclose(fast.simulated_cost, expected_cost)
    assert np.isclose(fast.simulated_latency_s, 0.1 + 0.2 + 10 / 100)

    panel = make_complete_panel(simulated, MODELS)
    assert panel.score.shape == (45, 3)
    assert panel.models == MODELS


def test_dataset_ood_split_is_disjoint(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    panel = make_complete_panel(simulate_economics(records, scenario()), MODELS)
    split = split_benchmark(panel, mode="dataset_ood", seed=42)
    assert set(split.train_datasets).isdisjoint(split.validation_datasets)
    assert set(split.train_datasets).isdisjoint(split.test_datasets)
    assert set(split.validation_datasets).isdisjoint(split.test_datasets)
    assert len(split.train) + len(split.validation) + len(split.test) == 45


def test_public_experiment_reports_headroom_and_fails_closed(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    panel = make_complete_panel(simulate_economics(records, scenario()), MODELS)
    split = split_benchmark(panel, mode="random", seed=42)
    result = run_public_benchmark(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=1.1,
        router_overhead_s=scenario().router_overhead_s,
    )
    assert not result.router_active
    assert result.fallback_model == "strong"
    assert result.summary.loc["outcome_oracle", "resource_savings"] > 0
    assert result.summary.loc["tfidf_safety_router", "fallback_usage"] == 1.0
    assert len(result.decisions) == len(split.test)

    permissive = run_public_benchmark(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        router_overhead_s=scenario().router_overhead_s,
    )
    assert permissive.router_active
    assert permissive.selected_threshold <= 0.95
    assert permissive.summary.loc["tfidf_safety_router", "resource_savings"] > 0


def test_oracle_probabilities_drive_modernbert_policy_without_timing_inference(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    analytical = EconomicsScenario(
        name="analytic",
        as_of="2026-08-17",
        latency_method="analytical",
        profiles=(
            ModelProfile("fast", 0.1, 0.2, parameters_billions=1.0, architecture="autoregressive"),
            ModelProfile("specialist", 0.2, 0.4, parameters_billions=2.0, architecture="diffusion", diffusion_steps=4, diffusion_block_size=8),
            ModelProfile("strong", 1.0, 2.0, parameters_billions=7.0, architecture="autoregressive"),
        ),
    )
    simulated = simulate_economics(records, analytical)
    assert simulated.latency_source.eq("analytical").all()
    # Changing realized completion length cannot change analytical latency.
    changed = records.copy()
    changed["completion_tokens"] = changed.completion_tokens * 100
    changed_simulated = simulate_economics(changed, analytical)
    assert np.allclose(
        simulated.simulated_latency_s, changed_simulated.simulated_latency_s
    )

    panel = make_complete_panel(simulated, MODELS)
    split = split_benchmark(panel, mode="random", seed=42)
    probabilities = np.full_like(panel.score, 0.01)
    probabilities[:, 0] = 0.98
    result = run_public_benchmark(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        routing_probabilities=probabilities,
        probability_kind="oracle",
        router_name="modernbert_oracle_router",
    )
    assert result.router_name == "modernbert_oracle_router"
    assert "modernbert_oracle_router" in result.summary.index
