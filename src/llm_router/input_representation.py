"""Token-budget strategies for ModernBERT router inputs.

Prefix truncation reproduces the historical behavior. Head-tail truncation
keeps both the beginning and ending tokens inside the same declared budget,
which makes context retention an explicit, auditable experiment variable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

TRUNCATION_STRATEGIES = ("prefix", "head_tail")


def validate_truncation_strategy(strategy: str) -> None:
    """Reject unknown input strategies before tokenization or training."""

    if strategy not in TRUNCATION_STRATEGIES:
        raise ValueError(
            f"Unknown truncation strategy {strategy!r}; expected one of "
            f"{TRUNCATION_STRATEGIES}."
        )


def _slice_head_tail(values: Iterable[int], max_input_tokens: int) -> list[int]:
    sequence = list(values)
    if len(sequence) <= max_input_tokens:
        return sequence
    head_tokens = (max_input_tokens + 1) // 2
    tail_tokens = max_input_tokens - head_tokens
    if tail_tokens == 0:
        return sequence[:head_tokens]
    return sequence[:head_tokens] + sequence[-tail_tokens:]


def encode_router_texts(
    tokenizer: Any,
    texts: list[str] | tuple[str, ...],
    *,
    max_input_tokens: int,
    truncation_strategy: str = "prefix",
    padding: bool = True,
    return_tensors: str | None = "pt",
) -> Any:
    """Encode router texts using a reproducible prefix or head-tail budget.

    Tokenization first adds the tokenizer's normal special tokens. Therefore a
    head-tail slice retains the leading classification token and final separator
    when the tokenizer supplies them. Every sequence-like tokenizer field is
    sliced identically before the tokenizer pads the batch.
    """

    validate_truncation_strategy(truncation_strategy)
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive.")
    normalized = list(texts)
    if not normalized:
        raise ValueError("At least one router text is required.")

    if truncation_strategy == "prefix":
        return tokenizer(
            normalized,
            padding=padding,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors=return_tensors,
        )

    untruncated = tokenizer(
        normalized,
        padding=False,
        truncation=False,
    )
    rows: list[dict[str, list[int]]] = []
    for row_index in range(len(normalized)):
        row: dict[str, list[int]] = {}
        for field, batch_values in untruncated.items():
            row[field] = _slice_head_tail(
                batch_values[row_index], max_input_tokens
            )
        rows.append(row)
    return tokenizer.pad(
        rows,
        padding=padding,
        return_tensors=return_tensors,
    )
