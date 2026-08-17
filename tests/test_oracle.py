from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from llm_router.analytical_latency import (
    AnalyticalModelProfile,
    HardwareProfile,
    estimate_latency_matrix,
)
from llm_router.oracle import oracle_choices, oracle_routing_loss
from llm_router.public_benchmark import BenchmarkPanel, BenchmarkSplit


def test_analytical_latency_uses_prompt_size_model_size_and_architecture():
    models = (
        AnalyticalModelProfile("small-ar", 1.5, "autoregressive"),
        AnalyticalModelProfile(
            "small-diffusion",
            1.5,
            "diffusion",
            diffusion_steps=4,
            diffusion_block_size=8,
        ),
        AnalyticalModelProfile("large-ar", 7.0, "autoregressive"),
    )
    latency = estimate_latency_matrix([32, 512], models, HardwareProfile())
    assert latency.shape == (2, 3)
    assert np.all(latency[1] > latency[0])
    assert np.all(latency[:, 2] > latency[:, 0])
    assert not np.allclose(latency[:, 0], latency[:, 1])


def test_oracle_loss_prefers_fastest_quality_preserving_model():
    quality = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    latency = np.array([[1.0, 0.5, 5.0], [1.0, 2.0, 5.0]])
    targets = oracle_choices(quality, latency, fallback_index=2)
    assert targets.tolist() == [0, 1]

    good_logits = torch.tensor([[8.0, -4.0, -4.0], [-4.0, 8.0, -4.0]])
    bad_logits = torch.tensor([[-4.0, 8.0, -4.0], [8.0, -4.0, -4.0]])
    quality_tensor = torch.tensor(quality, dtype=torch.float32)
    latency_tensor = torch.tensor(latency, dtype=torch.float32)
    good_loss, _ = oracle_routing_loss(
        good_logits, quality_tensor, latency_tensor, fallback_index=2
    )
    bad_loss, parts = oracle_routing_loss(
        bad_logits, quality_tensor, latency_tensor, fallback_index=2
    )
    assert good_loss < bad_loss
    assert parts["quality_risk"] > 0


def test_modernbert_poc_training_connects_oracle_loss_to_probabilities(monkeypatch):
    from llm_router import modernbert_poc
    from llm_router.models.modernbert_router import OracleModernBERTRouter

    class TinyBatch(dict):
        def to(self, device):
            return TinyBatch({key: value.to(device) for key, value in self.items()})

    class TinyTokenizer:
        def __call__(self, texts, **_):
            ids = torch.tensor([[len(text) % 7 + 1] for text in texts])
            return TinyBatch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 6)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    monkeypatch.setattr(
        modernbert_poc,
        "build_oracle_router",
        lambda config, model_count: (
            OracleModernBERTRouter(TinyEncoder(), 6, model_count),
            TinyTokenizer(),
        ),
    )
    examples = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b", "c", "c"],
            "prompt_tokens": [10, 20, 30, 40, 50, 60],
            "prompt": [f"prompt {index}" for index in range(6)],
        }
    )
    panel = BenchmarkPanel(
        examples=examples,
        models=("fast", "fallback"),
        score=np.array(
            [[1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [0, 1]], dtype=float
        ),
        cost=np.ones((6, 2)),
        latency=np.array([[1, 4]] * 6, dtype=float),
    )
    split = BenchmarkSplit(
        train=np.array([0, 1, 2]),
        validation=np.array([3]),
        test=np.array([4, 5]),
        mode="random",
        train_datasets=("a", "b"),
        validation_datasets=("b",),
        test_datasets=("c",),
    )
    result = modernbert_poc.train_modernbert_oracle_poc(
        panel, split, epochs=1, batch_size=2, device="cpu"
    )
    assert result.probabilities.shape == panel.score.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert len(result.history) == 1
