from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_post_training_lab.sft import IGNORE_INDEX, make_collate_fn

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "data/processed/gsm8k/train"
CHECKPOINT_DIR = Path("checkpoints/sft-tiny")

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
        collate_fn=make_collate_fn(
            tokenizer,
            MAX_LENGTH,
        ),
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
