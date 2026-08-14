import torch

from llm_router.models.modernbert_router_inference import ModernBERTRouterInference


def test_disabled_router_returns_fallback_without_model_load():
    manifest = {
        "model_names": ["small", "fallback"],
        "fallback_model": "fallback",
        "nonfallback_models": ["small"],
        "router_active": False,
    }
    router = ModernBERTRouterInference(
        manifest, model=None, tokenizer=None, device="cpu", compute_dtype=torch.float16
    )
    decision = router.route(
        prompt="hello",
        task="mmlu",
        subject="test",
        num_choices=4,
        length_bin="xs",
    )
    assert decision.selected_model == "fallback"
    assert not decision.router_active
    assert decision.router_overhead_s == 0.0
