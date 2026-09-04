"""Token-budget strategies for ModernBERT router inputs.

Prefix truncation reproduces the historical behavior. Head-tail truncation
keeps both the beginning and ending tokens inside the same declared budget.
Prefix-with-last keeps the prompt beginning plus one final sentinel token for a
causal classifier. These make context retention an explicit, auditable
experiment variable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

TRUNCATION_STRATEGIES = ("prefix", "head_tail", "prefix_with_last")


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


def _slice_prefix_with_last(
    values: Iterable[int], max_input_tokens: int
) -> list[int]:
    """Keep a prefix plus the final sentinel token inside one fixed budget.

    For example, values ``[10, 11, 12, 13, 99]`` with budget four become
    ``[10, 11, 12, 99]``.  Notebook 06 uses ``99`` conceptually as Qwen's
    existing end-of-text token, so causal pooling always reads the same final
    position even when the prompt itself exceeds the budget.
    """

    sequence = list(values)
    if len(sequence) <= max_input_tokens:
        return sequence
    if max_input_tokens == 1:
        return sequence[-1:]
    return sequence[: max_input_tokens - 1] + sequence[-1:]


def encode_router_texts(
    tokenizer: Any,
    texts: list[str] | tuple[str, ...],
    *,
    max_input_tokens: int,
    truncation_strategy: str = "prefix",
    padding: bool = True,
    return_tensors: str | None = "pt",
) -> Any:
    """Encode router texts using a reproducible declared truncation budget.

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
            if truncation_strategy == "head_tail":
                row[field] = _slice_head_tail(
                    batch_values[row_index], max_input_tokens
                )
            else:
                row[field] = _slice_prefix_with_last(
                    batch_values[row_index], max_input_tokens
                )
        rows.append(row)
    return tokenizer.pad(
        rows,
        padding=padding,
        return_tensors=return_tensors,
    )
