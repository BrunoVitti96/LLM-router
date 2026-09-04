from types import SimpleNamespace

import torch

from llm_router.config import DEFAULT_CONFIG
from llm_router.models import qwen_last_token_router


class PositionEncoder(torch.nn.Module):
    """Expose token positions so the selected pooling position is observable."""

    def forward(self, input_ids, attention_mask):
        batch, tokens = input_ids.shape
        positions = torch.arange(tokens, dtype=torch.float32).view(1, tokens, 1)
        hidden = positions.expand(batch, tokens, 2)
        return SimpleNamespace(last_hidden_state=hidden)


def test_qwen_router_pools_last_nonpadding_token():
    model = qwen_last_token_router.QwenLastTokenRouter(
        PositionEncoder(), hidden_size=2, nonfallback_count=1, model_count=2
    )
    model.dropout = torch.nn.Identity()
    model.safety_head.weight.data.fill_(1.0)
    model.safety_head.bias.data.zero_()

    output = model(
        input_ids=torch.ones((2, 5), dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
    )

    # Lengths five and three select zero-based positions four and two.  Each
    # duplicated hidden coordinate is summed by the head: 8 and 4.
    assert output["safety_logits"].squeeze(-1).tolist() == [8.0, 4.0]


def test_qwen_builder_uses_right_padding_and_disables_cache(monkeypatch):
    tokenizer = SimpleNamespace(
        padding_side="left",
        pad_token_id=0,
        eos_token_id=1,
    )

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=4,
                use_cache=True,
                pad_token_id=None,
            )
            self.gradient_checkpointing_kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    base = TinyEncoder()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    class FakeModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return base

    monkeypatch.setattr(qwen_last_token_router, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(qwen_last_token_router, "AutoModel", FakeModel)
    monkeypatch.setattr(
        qwen_last_token_router,
        "get_peft_model",
        lambda encoder, _config: encoder,
    )

    model, returned_tokenizer = (
        qwen_last_token_router.build_qwen_last_token_router(
            DEFAULT_CONFIG, nonfallback_count=2, model_count=3
        )
    )

    assert isinstance(model, qwen_last_token_router.QwenLastTokenRouter)
    assert returned_tokenizer.padding_side == "right"
    assert base.config.use_cache is False
    assert base.config.pad_token_id == tokenizer.pad_token_id
    assert base.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_qwen_router_formatter_places_sentinel_last():
    formatted = qwen_last_token_router.format_qwen_router_text("example prompt")
    assert formatted == "example prompt\n<|endoftext|>"
