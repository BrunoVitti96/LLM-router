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
    assert contract["minimum_predicted_speedup"] == 0.02
    assert contract["checkpoint_rule"] == (
        "calibrated overhead-inclusive validation routing"
    )
