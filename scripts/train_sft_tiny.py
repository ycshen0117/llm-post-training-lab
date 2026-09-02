from pathlib import Path

import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "data/processed/gsm8k/train"
CHECKPOINT_DIR = Path("checkpoints/sft-tiny")
IGNORE_INDEX = -100

NUM_TRAIN_EXAMPLES = 32
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
MAX_STEPS = 5
MAX_LENGTH = 512
SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def tokenize_example(
    example: dict,
    tokenizer,
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

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    if not (labels != IGNORE_INDEX).any():
        raise ValueError(
            "Truncation removed all supervised assistant tokens. "
            f"original_length={original_length}, "
            f"prompt_length={prompt_length}, "
            f"max_length={MAX_LENGTH}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def make_collate_fn(tokenizer):
    def collate_fn(
        examples: list[dict],
    ) -> dict[str, torch.Tensor]:
        tokenized = [tokenize_example(example, tokenizer) for example in examples]

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


def inspect_batch(batch: dict[str, torch.Tensor]) -> None:
    print("\nBatch shapes:")
    print(f"input_ids:      {batch['input_ids'].shape}")
    print(f"attention_mask: {batch['attention_mask'].shape}")
    print(f"labels:         {batch['labels'].shape}")

    num_supervised = (batch["labels"] != IGNORE_INDEX).sum().item()
    num_positions = batch["labels"].numel()

    print(f"Supervised tokens: {num_supervised}/{num_positions}")

    assert num_supervised > 0
    assert num_supervised < num_positions


def main() -> None:
    torch.manual_seed(SEED)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Random seed: {SEED}")

    full_dataset = load_from_disk(DATASET_PATH)

    num_examples = min(
        NUM_TRAIN_EXAMPLES,
        len(full_dataset),
    )

    dataset = full_dataset.select(
        range(num_examples),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
    )

    model.to(device)
    model.train()

    data_generator = torch.Generator()
    data_generator.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        generator=data_generator,
    )

    print(f"Dataset size: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nTraining:")

    for step, batch in enumerate(dataloader):
        if step == 0:
            inspect_batch(batch)

        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"

        loss.backward()
        optimizer.step()

        print(f"step={step:03d} loss={loss.item():.4f}")

        if step + 1 >= MAX_STEPS:
            break

    print("\nSaving checkpoint:")

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    print(f"Saved checkpoint to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
