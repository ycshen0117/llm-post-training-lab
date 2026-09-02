import pytest
import torch

from llm_post_training_lab.sft import (
    IGNORE_INDEX,
    make_collate_fn,
    tokenize_example,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_tensors,
    ):
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"

        prompt_ids = [10, 11]

        if add_generation_prompt:
            token_ids = prompt_ids
        else:
            response = messages[-1]["content"]

            if response == "short":
                assistant_ids = [20]
            else:
                assistant_ids = [20, 21]

            token_ids = prompt_ids + assistant_ids

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
        )

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def make_example(response: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": "question",
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]
    }


def test_tokenize_example_masks_prompt_tokens() -> None:
    tokenizer = FakeTokenizer()
    example = make_example("long")

    result = tokenize_example(
        example,
        tokenizer,
        max_length=4,
    )

    assert result["input_ids"].tolist() == [
        10,
        11,
        20,
        21,
    ]

    assert result["attention_mask"].tolist() == [
        1,
        1,
        1,
        1,
    ]

    assert result["labels"].tolist() == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        20,
        21,
    ]


def test_tokenize_example_rejects_truncated_answer() -> None:
    tokenizer = FakeTokenizer()
    example = make_example("short")

    with pytest.raises(
        ValueError,
        match="removed all supervised assistant tokens",
    ):
        tokenize_example(
            example,
            tokenizer,
            max_length=2,
        )


def test_collate_fn_pads_batch() -> None:
    tokenizer = FakeTokenizer()

    collate_fn = make_collate_fn(
        tokenizer,
        max_length=4,
    )

    batch = collate_fn(
        [
            make_example("short"),
            make_example("long"),
        ]
    )

    assert batch["input_ids"].tolist() == [
        [10, 11, 20, 0],
        [10, 11, 20, 21],
    ]

    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]

    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 20, IGNORE_INDEX],
        [IGNORE_INDEX, IGNORE_INDEX, 20, 21],
    ]
