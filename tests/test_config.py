from llm_router.config import DEFAULT_CONFIG, EXPECTED_MODEL_REPOS, MODEL_NAMES


def test_synchronized_contract_constants():
    DEFAULT_CONFIG.validate()
    assert MODEL_NAMES == (
        "qwen2.5-1.5b-ar",
        "fast-dllm-v2-1.5b",
        "qwen2.5-7b-4bit",
    )
    assert set(EXPECTED_MODEL_REPOS) == set(MODEL_NAMES)


def test_contract_keeps_deployment_guard():
    contract = DEFAULT_CONFIG.contract("evidence", "gpu", "torch.float16")
    assert contract["minimum_quality_retention"] == 0.98
    assert contract["quality_confidence"] == 0.95
    assert contract["minimum_predicted_speedup"] == 0.02
    assert contract["policy_gates"] == {
        "validation_quality_margin": 0.01,
        "minimum_macro_quality_retention": 0.98,
        "maximum_quality_loss_rate_ucl": 0.025,
        "minimum_routed_safety_precision_lcb": 0.90,
        "minimum_guarded_dataset_quality_retention_lcb": 0.90,
        "minimum_guarded_dataset_prompts": 100,
        "conservative_router_overhead_s": 0.020,
    }
    assert contract["checkpoint_rule"] == (
        "calibrated overhead-inclusive validation routing"
    )
    assert "retention LCB" in contract["deployment_guard"]
