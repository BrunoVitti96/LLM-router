from dataclasses import replace
from pathlib import Path

import pandas as pd

from llm_router.config import DEFAULT_CONFIG, EXPECTED_MODEL_REPOS
from llm_router.utils.data import Evidence, prepare_router_data


def test_prompt_level_split_and_fallback_targets_are_deterministic():
    config = replace(DEFAULT_CONFIG, n_per_task=10)
    prompts = []
    measurements = []
    for task in config.tasks:
        for number in range(config.n_per_task):
            prompt_id = f"{task}-{number}"
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "task": task,
                    "subject": "subject",
                    "num_choices": 4,
                    "prompt_words": 2,
                    "length_bin": "xs",
                    "prompt": f"question {number}",
                    "reference": "A",
                }
            )
            for model_index, model_name in enumerate(config.model_names):
                measurements.append(
                    {
                        "prompt_id": prompt_id,
                        "task": task,
                        "model": model_name,
                        "model_repo": EXPECTED_MODEL_REPOS[model_name],
                        "quality": float(model_index == 2 or number % 2 == 0),
                        "generation_s": float(model_index + 1),
                        "output_tokens": 10 + model_index,
                    }
                )

    evidence = Evidence(
        root=Path("."),
        manifest={},
        prompts=pd.DataFrame(prompts),
        measurements=pd.DataFrame(measurements),
        fingerprint="test",
    )
    data = prepare_router_data(evidence, config)

    assert data.masks["train"].sum() == 18
    assert data.masks["validation"].sum() == 6
    assert data.masks["test"].sum() == 6
    assert data.fallback_name == "qwen2.5-7b-4bit"
    assert data.replacement_safe.shape == (30, 2)
    assert data.text[0].startswith(
        "[TASK=gsm8k] [SUBJECT=subject] [CHOICES=4] [LENGTH_BIN=xs]"
    )
