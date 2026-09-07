"""Qwen2.5 causal router using the final non-padding token representation.

The router never generates an answer or a model-name token.  It appends a
fixed routing sentinel to the prompt, reads the hidden state at that final
position, and applies the same independent fallback-relative safety heads used
by the ModernBERT experiment.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

from llm_router.config import RouterConfig
from llm_router.models.modernbert_router import HybridModernBERTRouter

QWEN_ROUTER_TEXT_PREFIX = ""
# Qwen already has this one-token sentinel in its vocabulary.  Reusing it avoids
# a randomly initialized special-token embedding and lets ``prefix_with_last``
# preserve exactly one known final token under truncation.
QWEN_ROUTER_TEXT_SUFFIX = "\n<|endoftext|>"
QWEN_LAST_TOKEN_POOLING = "last_nonpadding_token"


def format_qwen_router_text(text: str) -> str:
    """Append the fixed sentinel whose state is used for safety prediction."""

    return f"{QWEN_ROUTER_TEXT_PREFIX}{text}{QWEN_ROUTER_TEXT_SUFFIX}"


class QwenLastTokenRouter(HybridModernBERTRouter):
    """Causal Qwen encoder with safety logits read from its final input token.

    With right padding, example lengths 7 and 4 produce final indices 6 and 3.
    Selecting those positions prevents padding from becoming the representation
    for the shorter example.  The final sentinel can attend to every preceding
    prompt token under Qwen's causal mask.
    """

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(**inputs).last_hidden_state
        attention_mask = inputs["attention_mask"]
        final_indices = attention_mask.to(torch.long).sum(dim=1).sub(1)
        final_indices = final_indices.clamp_min(0)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_indices, final_indices]
        pooled = self.dropout(pooled)
        # The encoder can return FP16/BF16 while the heads remain FP32.
        # Match each head at this boundary so calibration, timing, and demo
        # inference also work without an autocast context. For example,
        # FP16 [4, 4] becomes FP32 [4, 4], preserving a unit-weight logit of 8.
        return {
            "safety_logits": self.safety_head(
                pooled.to(dtype=self.safety_head.weight.dtype)
            ),
            "oracle_logits": self.oracle_head(
                pooled.to(dtype=self.oracle_head.weight.dtype)
            ),
        }


def build_qwen_last_token_router(
    config: RouterConfig,
    nonfallback_count: int,
    model_count: int,
) -> tuple[QwenLastTokenRouter, AutoTokenizer]:
    """Build a LoRA Qwen router without enabling answer generation.

    The base weights use FP16 on pre-Ampere CUDA devices such as a Tesla T4 and
    BF16 on Ampere-or-newer GPUs.  Gradient checkpointing exchanges additional
    computation for lower activation memory, which makes the 1.5B/1,024-token
    experiment practical on a single Colab GPU.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_repo,
        revision=config.encoder_revision,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The Qwen router tokenizer requires a pad or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        model_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        model_dtype = torch.float32
    base_encoder = AutoModel.from_pretrained(
        config.encoder_repo,
        revision=config.encoder_revision,
        attn_implementation="sdpa",
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    base_encoder.config.use_cache = False
    base_encoder.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(base_encoder, "gradient_checkpointing_enable"):
        base_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
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
    if hasattr(encoder, "enable_input_require_grads"):
        encoder.enable_input_require_grads()
    return (
        QwenLastTokenRouter(
            encoder,
            hidden_size,
            nonfallback_count,
            model_count,
        ),
        tokenizer,
    )
