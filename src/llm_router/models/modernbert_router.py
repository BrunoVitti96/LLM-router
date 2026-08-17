"""Trainable ModernBERT encoder and decision heads."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

from llm_router.config import RouterConfig


class DecisionAlignedRouter(torch.nn.Module):
    """Masked-mean ModernBERT representation with three prediction heads."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        hidden_size: int,
        nonfallback_count: int,
        model_count: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(0.10)
        self.safety_head = torch.nn.Linear(hidden_size, nonfallback_count)
        self.latency_head = torch.nn.Linear(hidden_size, model_count)
        self.token_head = torch.nn.Linear(hidden_size, model_count)

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return {
            "safety_logits": self.safety_head(pooled),
            "latency_log": self.latency_head(pooled),
            "token_log": self.token_head(pooled),
        }


class HybridModernBERTRouter(torch.nn.Module):
    """Predict replacement safety; imitate the oracle through an auxiliary head."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        hidden_size: int,
        nonfallback_count: int,
        model_count: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(0.10)
        self.safety_head = torch.nn.Linear(hidden_size, nonfallback_count)
        self.oracle_head = torch.nn.Linear(hidden_size, model_count)

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return {
            "safety_logits": self.safety_head(pooled),
            "oracle_logits": self.oracle_head(pooled),
        }


def build_trainable_router(
    config: RouterConfig,
    nonfallback_count: int,
    model_count: int,
) -> tuple[DecisionAlignedRouter, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_repo, revision=config.encoder_revision
    )
    base_encoder = AutoModel.from_pretrained(
        config.encoder_repo,
        revision=config.encoder_revision,
        attn_implementation="sdpa",
    )
    hidden_size = int(base_encoder.config.hidden_size)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    encoder = get_peft_model(base_encoder, lora_config)
    model = DecisionAlignedRouter(
        encoder=encoder,
        hidden_size=hidden_size,
        nonfallback_count=nonfallback_count,
        model_count=model_count,
    )
    return model, tokenizer


def build_hybrid_router(
    config: RouterConfig,
    nonfallback_count: int,
    model_count: int,
) -> tuple[HybridModernBERTRouter, AutoTokenizer]:
    """Build the rank-4 LoRA ModernBERT used by the hybrid routing POC."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_repo, revision=config.encoder_revision
    )
    base_encoder = AutoModel.from_pretrained(
        config.encoder_repo,
        revision=config.encoder_revision,
        attn_implementation="sdpa",
    )
    hidden_size = int(base_encoder.config.hidden_size)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    encoder = get_peft_model(base_encoder, lora_config)
    return (
        HybridModernBERTRouter(
            encoder, hidden_size, nonfallback_count, model_count
        ),
        tokenizer,
    )
