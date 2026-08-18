"""Command-line driver for the public routing-benchmark experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from llm_router.config import DEFAULT_CONFIG
from llm_router.public_benchmark import (
    EconomicsScenario,
    benchmark_inventory,
    export_public_benchmark,
    load_llmrouterbench,
    make_complete_panel,
    run_public_benchmark,
    simulate_economics,
    split_benchmark,
)


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a hybrid ModernBERT safety router and evaluate headroom on an "
            "extracted LLMRouterBench result bundle."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--list-inventory",
        action="store_true",
        help="Print available dataset/model pairs and exit.",
    )
    parser.add_argument("--scenario", help="Path to a dated economics JSON file.")
    parser.add_argument("--models", help="Comma-separated model directory names.")
    parser.add_argument("--datasets", help="Optional comma-separated datasets.")
    parser.add_argument(
        "--objective", choices=("cost", "latency"), default="latency"
    )
    parser.add_argument(
        "--split-mode", choices=("random", "dataset_ood"), default="random"
    )
    parser.add_argument("--minimum-quality-retention", type=float, default=0.98)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--router",
        choices=("modernbert-hybrid", "tfidf"),
        default="modernbert-hybrid",
        help="ModernBERT is the POC router; TF-IDF is a cheap diagnostic baseline.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", help="Optional torch device, e.g. cuda or cpu.")
    parser.add_argument("--output-dir", default="reports_benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_inventory:
        inventory = benchmark_inventory(args.data_root)
        if inventory.empty:
            raise SystemExit("No result files were found under the benchmark root.")
        print(
            inventory.groupby(["dataset", "source_split", "model"])
            .size()
            .rename("files")
            .to_string()
        )
        return 0
    if not args.scenario or not args.models:
        raise SystemExit("--scenario and --models are required unless listing inventory.")

    models = _comma_list(args.models)
    scenario = EconomicsScenario.from_json(args.scenario)
    records = load_llmrouterbench(
        args.data_root,
        models=models,
        datasets=_comma_list(args.datasets),
    )
    simulated = simulate_economics(records, scenario)
    panel = make_complete_panel(simulated, models=models)
    split = split_benchmark(panel, mode=args.split_mode, seed=args.seed)
    probabilities = None
    router_name = "tfidf_safety_router"
    training = None
    router_config = replace(DEFAULT_CONFIG, seed=args.seed)
    if args.router == "modernbert-hybrid":
        from llm_router.modernbert_poc import train_modernbert_hybrid_poc

        training = train_modernbert_hybrid_poc(
            panel,
            split,
            config=router_config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        probabilities = training.safety_probabilities
        router_name = "modernbert_hybrid_router"
    result = run_public_benchmark(
        panel,
        split,
        objective=args.objective,
        minimum_quality_retention=args.minimum_quality_retention,
        confidence=args.confidence,
        router_overhead_s=scenario.router_overhead_s,
        seed=args.seed,
        routing_probabilities=probabilities,
        router_name=router_name,
    )
    output_dir = export_public_benchmark(result, scenario, Path(args.output_dir))
    if training is not None:
        from llm_router.modernbert_poc import export_modernbert_hybrid_poc

        export_modernbert_hybrid_poc(
            training,
            panel.models,
            output_dir / "modernbert_router",
            selected_threshold=result.selected_threshold,
            router_active=result.router_active,
            poc_passed=result.poc_passed,
            failure_reasons=result.failure_reasons,
            config=router_config,
        )
    print(result.summary.to_string())
    print(
        f"\nrouter_active={result.router_active} "
        f"poc_passed={result.poc_passed} "
        f"threshold={result.selected_threshold:g} "
        f"fallback={result.fallback_model}"
    )
    if result.failure_reasons:
        print("Failure reasons:")
        for reason in result.failure_reasons:
            print(f"- {reason}")
    print(f"Reports written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
