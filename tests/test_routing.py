from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from llm_router.config import DEFAULT_CONFIG
from llm_router.utils.routing import route_metrics, search_selector, select_routes


def synthetic_data():
    return SimpleNamespace(
        strongest_idx=2,
        nonfallback_indices=np.array([0, 1]),
        nonfallback_names=("small", "fast"),
        quality=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        latency=np.array([[5.0, 7.0, 10.0], [6.0, 8.0, 10.0]]),
        task_latency_baseline=np.array([[5.0, 7.0, 10.0], [6.0, 8.0, 10.0]]),
    )


def test_selector_always_keeps_fallback_eligible():
    data = synthetic_data()
    probabilities = np.array([[0.9, 0.1], [0.1, 0.8]])
    predicted = np.array([[5.0, 7.0, 10.0], [6.0, 9.9, 10.0]])
    chosen, eligible, selected_fallback, no_alternative = select_routes(
        probabilities,
        predicted,
        np.array([0.75, 0.75]),
        data,
        DEFAULT_CONFIG,
    )
    assert chosen.tolist() == [0, 2]
    assert eligible[:, 2].all()
    assert selected_fallback.tolist() == [False, True]
    assert no_alternative.tolist() == [False, True]


def test_search_finds_quality_preserving_speedup():
    data = synthetic_data()
    config = replace(
        DEFAULT_CONFIG,
        safety_threshold_grid=(0.5,),
        latency_blend_grid=(0.0,),
    )
    prediction = {
        "indices": np.array([0, 1]),
        "latency": data.task_latency_baseline.copy(),
        "overhead_s": np.zeros(2),
    }
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    search, best = search_selector(prediction, probabilities, data, config)
    assert len(search) == 1
    assert best is not None
    assert best.quality_retention == 1.0
    assert best.latency_reduction > 0


def test_overhead_is_included_in_latency_reduction():
    data = synthetic_data()
    metrics = route_metrics(
        np.array([0]),
        np.array([0]),
        data,
        DEFAULT_CONFIG,
        overhead_s=np.array([6.0]),
    )
    assert metrics["generation_s"] == 5.0
    assert metrics["latency_s"] == 11.0
    assert metrics["latency_reduction"] < 0
