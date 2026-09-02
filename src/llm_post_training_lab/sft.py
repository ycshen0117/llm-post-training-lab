import torch
from torch.nn.utils.rnn import pad_sequence


IGNORE_INDEX = -100


def tokenize_example(
    example: dict,
    tokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    messages = example["messages"]

    full_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_encoding = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = full_encoding["input_ids"].squeeze(0)
    attention_mask = full_encoding["attention_mask"].squeeze(0)

    prompt_ids = prompt_encoding["input_ids"].squeeze(0)
    prompt_length = prompt_ids.shape[0]

    assert torch.equal(
        prompt_ids,
        input_ids[:prompt_length],
    ), "Prompt tokenization is not a prefix of the full conversation."

    labels = input_ids.clone()
    labels[:prompt_length] = IGNORE_INDEX

    original_length = input_ids.shape[0]

    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    if not (labels != IGNORE_INDEX).any():
        raise ValueError(
            "Truncation removed all supervised assistant tokens. "
            f"original_length={original_length}, "
            f"prompt_length={prompt_length}, "
            f"max_length={max_length}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def make_collate_fn(
    tokenizer,
    max_length: int,
):
    def collate_fn(
        examples: list[dict],
    ) -> dict[str, torch.Tensor]:
        tokenized = [
            tokenize_example(
                example,
                tokenizer,
                max_length,
            )
            for example in examples
        ]

        input_ids = pad_sequence(
            [example["input_ids"] for example in tokenized],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        attention_mask = pad_sequence(
            [example["attention_mask"] for example in tokenized],
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            [example["labels"] for example in tokenized],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        assert input_ids.shape == attention_mask.shape == labels.shape
        assert (labels != IGNORE_INDEX).any(dim=1).all(), (
            "Every example must contain at least one supervised token."
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn
