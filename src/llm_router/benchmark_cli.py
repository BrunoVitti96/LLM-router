"""Command-line driver for the public routing-benchmark experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

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
            "Evaluate routing headroom and a TF-IDF safety router on an "
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
        "--objective", choices=("cost", "latency"), default="cost"
    )
    parser.add_argument(
        "--split-mode", choices=("random", "dataset_ood"), default="dataset_ood"
    )
    parser.add_argument("--minimum-quality-retention", type=float, default=0.98)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
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
    result = run_public_benchmark(
        panel,
        split,
        objective=args.objective,
        minimum_quality_retention=args.minimum_quality_retention,
        confidence=args.confidence,
        router_overhead_s=scenario.router_overhead_s,
        seed=args.seed,
    )
    output_dir = export_public_benchmark(result, scenario, Path(args.output_dir))
    print(result.summary.to_string())
    print(
        f"\nrouter_active={result.router_active} "
        f"threshold={result.selected_threshold:g} "
        f"fallback={result.fallback_model}"
    )
    print(f"Reports written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
