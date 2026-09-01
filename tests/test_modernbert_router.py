from types import SimpleNamespace

import pytest
import torch

from llm_router.config import DEFAULT_CONFIG
from llm_router.models import modernbert_router


@pytest.mark.parametrize(
    "builder_name", ("build_trainable_router", "build_hybrid_router")
)
def test_router_builders_disable_reference_compile(monkeypatch, builder_name):
    """Threaded v5 training must never enter ModernBERT's Dynamo path."""

    model_loads = []

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            model_loads.append((repo, kwargs))
            return TinyEncoder()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            return (repo, kwargs)

    monkeypatch.setattr(modernbert_router, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(modernbert_router, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        modernbert_router, "get_peft_model", lambda encoder, _config: encoder
    )

    builder = getattr(modernbert_router, builder_name)
    model, _ = builder(DEFAULT_CONFIG, nonfallback_count=2, model_count=3)

    assert isinstance(model, torch.nn.Module)
    assert len(model_loads) == 1
    assert model_loads[0][1]["reference_compile"] is False
