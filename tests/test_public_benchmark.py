import json

import numpy as np
import pandas as pd

from llm_router.public_benchmark import (
    BenchmarkPanel,
    EconomicsScenario,
    ModelProfile,
    benchmark_fingerprint,
    benchmark_inventory,
    export_public_benchmark,
    load_llmrouterbench,
    make_complete_panel,
    normalized_prompt_hash,
    run_public_benchmark,
    select_validation_policy,
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


def test_inventory_discovers_a_release_under_unknown_wrapper_directories(tmp_path):
    release_root = tmp_path / "download"
    result_path = (
        release_root
        / "bench-release-2026"
        / "payload"
        / "math"
        / "test"
        / "fast"
        / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    inventory = benchmark_inventory(release_root)

    assert inventory[["dataset", "source_split", "model"]].to_dict("records") == [
        {"dataset": "math", "source_split": "test", "model": "fast"}
    ]


def test_dataset_ood_split_is_disjoint(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    panel = make_complete_panel(simulate_economics(records, scenario()), MODELS)
    split = split_benchmark(panel, mode="dataset_ood", seed=42)
    assert set(split.train_datasets).isdisjoint(split.validation_datasets)
    assert set(split.train_datasets).isdisjoint(split.test_datasets)
    assert set(split.validation_datasets).isdisjoint(split.test_datasets)
    assert len(split.train) + len(split.validation) + len(split.test) == 45


def test_random_split_groups_repeated_prompt_content():
    prompts = [f"unique prompt {index}" for index in range(20)]
    prompts[19] = prompts[0]
    examples = np.array(["a"] * 10 + ["b"] * 10, dtype=object)
    frame = {
        "dataset": examples,
        "prompt": prompts,
        "prompt_tokens": np.arange(20) + 10,
    }
    example_frame = pd.DataFrame(frame)
    example_frame["prompt_hash"] = example_frame.prompt.map(normalized_prompt_hash)
    panel = BenchmarkPanel(
        examples=example_frame,
        models=("fast", "fallback"),
        score=np.ones((20, 2)),
        cost=np.ones((20, 2)),
        latency=np.ones((20, 2)),
    )

    split = split_benchmark(panel, mode="random", seed=42)
    split_name = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for index in indices:
            split_name[index] = name
    assert split_name[0] == split_name[19]


def test_export_records_explicit_poc_status_and_candidate_diagnostics(tmp_path):
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

    output = export_public_benchmark(result, scenario(), tmp_path / "report")
    manifest = json.loads(
        (output / "experiment_manifest.json").read_text(encoding="utf-8")
    )

    assert not manifest["poc_passed"]
    assert manifest["failure_reasons"]
    assert manifest["success_criteria"]["requires_nontrivial_routing"]
    assert (output / "candidate_diagnostics.csv").is_file()
    assert (output / "per_dataset_metrics.csv").is_file()
    assert (output / "test_router_overhead_sensitivity.csv").is_file()
    assert manifest["schema_version"] == 5
    assert manifest["single_run_passed"] == manifest["poc_passed"]
    assert manifest["benchmark_fingerprint_sha256"]


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
    assert not result.poc_passed
    assert not result.single_run_passed
    assert result.failure_reasons
    assert result.fallback_model == "strong"
    assert result.summary.loc["outcome_oracle", "resource_savings"] > 0
    assert result.summary.loc["tfidf_safety_router", "fallback_usage"] == 1.0
    assert len(result.decisions) == len(split.test)
    assert set(result.candidate_diagnostics.split) == {
        "train",
        "validation",
        "test",
    }
    assert result.summary.loc["outcome_oracle", "oracle_savings_capture"] == 1.0
    assert {
        "quality_loss_rate_ucl",
        "routed_safety_precision",
        "routed_safety_precision_lcb",
        "safe_opportunity_recall",
        "macro_dataset_quality_retention",
        "guarded_dataset_quality_retention_lcb",
        "conservative_resource_savings",
    }.issubset(result.summary.columns)
    assert not result.per_dataset_metrics.empty
    assert not result.router_overhead_sensitivity.empty

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


def test_safety_probabilities_drive_hybrid_policy_without_timing_inference(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    analytical = EconomicsScenario(
        name="analytic",
        as_of="2026-08-17",
        latency_method="analytical",
        profiles=(
            ModelProfile(
                "fast", 0.1, 0.2, parameters_billions=1.0, architecture="autoregressive"
            ),
            ModelProfile(
                "specialist",
                0.2,
                0.4,
                parameters_billions=2.0,
                architecture="diffusion",
                diffusion_steps=4,
                diffusion_block_size=8,
            ),
            ModelProfile(
                "strong",
                1.0,
                2.0,
                parameters_billions=7.0,
                architecture="autoregressive",
            ),
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
    probabilities[:, 2] = 1.0
    result = run_public_benchmark(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        routing_probabilities=probabilities,
        router_name="modernbert_hybrid_router",
    )
    assert result.router_name == "modernbert_hybrid_router"
    assert "modernbert_hybrid_router" in result.summary.index
    assert {
        "safety_probability__fast",
        "safety_probability__specialist",
        "safety_probability__strong",
    }.issubset(result.decisions.columns)


def test_strict_policy_gates_and_decision_metadata_are_enforced(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    panel = make_complete_panel(simulate_economics(records, scenario()), MODELS)
    split = split_benchmark(panel, mode="random", seed=42)
    metadata = np.arange(len(panel.examples))

    result = run_public_benchmark(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        router_overhead_s=scenario().router_overhead_s,
        conservative_router_overhead_s=0.02,
        minimum_macro_quality_retention=0.5,
        maximum_quality_loss_rate_ucl=0.5,
        minimum_routed_safety_precision_lcb=0.5,
        minimum_guarded_dataset_quality_retention_lcb=0.5,
        minimum_guarded_dataset_prompts=2,
        decision_metadata={"router_input_tokens": metadata},
    )

    assert result.router_active
    assert (
        result.summary.loc["tfidf_safety_router", "conservative_resource_savings"] > 0
    )
    assert result.decisions.router_input_tokens.isin(metadata).all()
    assert result.minimum_macro_quality_retention == 0.5

    blocked = run_public_benchmark(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        router_overhead_s=scenario().router_overhead_s,
        conservative_router_overhead_s=10.0,
    )
    assert not blocked.router_active
    assert not blocked.poc_passed


def test_validation_setup_selection_never_uses_sealed_test_outcomes(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    panel = make_complete_panel(simulate_economics(records, scenario()), MODELS)
    split = split_benchmark(panel, mode="random", seed=42)
    probabilities = np.full_like(panel.score, 0.01)
    probabilities[:, 0] = 0.98
    probabilities[:, 2] = 1.0
    settings = {
        "objective": "latency",
        "minimum_quality_retention": 0.5,
        "confidence": 0.8,
        "threshold_grid": (0.90, 0.95),
        "routing_probabilities": probabilities,
        "minimum_consecutive_feasible_thresholds": 2,
    }

    original = select_validation_policy(panel, split, **settings)
    changed_score = panel.score.copy()
    changed_score[split.test] = 1.0 - changed_score[split.test]
    changed_panel = BenchmarkPanel(
        examples=panel.examples,
        models=panel.models,
        score=changed_score,
        cost=panel.cost,
        latency=panel.latency,
    )
    changed = select_validation_policy(changed_panel, split, **settings)

    pd.testing.assert_frame_equal(original.threshold_search, changed.threshold_search)
    assert original.selected_threshold == changed.selected_threshold
    assert benchmark_fingerprint(panel) != benchmark_fingerprint(changed_panel)
    assert {
        "passes_base_gates",
        "feasible_block_size",
        "passes_stability_gate",
        "is_feasible",
    }.issubset(original.threshold_search.columns)


def test_threshold_stability_gate_rejects_an_isolated_pass(tmp_path):
    records = load_llmrouterbench(synthetic_release(tmp_path), models=MODELS)
    panel = make_complete_panel(simulate_economics(records, scenario()), MODELS)
    split = split_benchmark(panel, mode="random", seed=42)

    permissive = select_validation_policy(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        minimum_consecutive_feasible_thresholds=1,
    )
    required_block = int(permissive.threshold_search.passes_base_gates.sum()) + 1
    stable = select_validation_policy(
        panel,
        split,
        objective="latency",
        minimum_quality_retention=0.5,
        confidence=0.8,
        minimum_consecutive_feasible_thresholds=required_block,
    )

    assert permissive.router_active
    assert not stable.router_active
    assert stable.failure_reasons
