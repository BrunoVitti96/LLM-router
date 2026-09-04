import pytest

from llm_router.input_representation import encode_router_texts


class FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length=None,
        return_tensors=None,
    ):
        del padding, return_tensors
        rows = []
        for text in texts:
            values = [101] + [int(value) for value in text.split()] + [102]
            if truncation:
                values = values[:max_length]
            rows.append(values)
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def pad(self, rows, *, padding, return_tensors):
        del padding, return_tensors
        width = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": [
                row["input_ids"] + [0] * (width - len(row["input_ids"]))
                for row in rows
            ],
            "attention_mask": [
                row["attention_mask"] + [0] * (width - len(row["attention_mask"]))
                for row in rows
            ],
        }


def test_prefix_and_head_tail_keep_different_context_with_equal_budget():
    tokenizer = FakeTokenizer()
    prefix = encode_router_texts(
        tokenizer,
        ["1 2 3 4 5 6"],
        max_input_tokens=5,
        truncation_strategy="prefix",
        return_tensors=None,
    )
    head_tail = encode_router_texts(
        tokenizer,
        ["1 2 3 4 5 6"],
        max_input_tokens=5,
        truncation_strategy="head_tail",
        return_tensors=None,
    )
    assert prefix["input_ids"] == [[101, 1, 2, 3, 4]]
    assert head_tail["input_ids"] == [[101, 1, 2, 6, 102]]
    assert head_tail["attention_mask"] == [[1, 1, 1, 1, 1]]


def test_prefix_with_last_preserves_one_final_sentinel_token():
    encoded = encode_router_texts(
        FakeTokenizer(),
        ["1 2 3 4 5 6"],
        max_input_tokens=5,
        truncation_strategy="prefix_with_last",
        return_tensors=None,
    )

    assert encoded["input_ids"] == [[101, 1, 2, 3, 102]]
    assert encoded["attention_mask"] == [[1, 1, 1, 1, 1]]


def test_input_representation_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown truncation strategy"):
        encode_router_texts(
            FakeTokenizer(),
            ["1 2"],
            max_input_tokens=5,
            truncation_strategy="middle",
            return_tensors=None,
        )
