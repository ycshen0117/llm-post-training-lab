import argparse
import json
import math
import tomllib
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_post_training_lab.sft import IGNORE_INDEX, make_collate_fn


def load_config(path: Path, parser: argparse.ArgumentParser) -> dict:
    try:
        with path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        parser.error(f"Cannot read configuration {path}: {error}")

    expected_types = {
        "model_id": (str,),
        "dataset_path": (str,),
        "checkpoint_dir": (str,),
        "num_train_examples": (int,),
        "batch_size": (int,),
        "learning_rate": (int, float),
        "max_steps": (int,),
        "max_length": (int,),
        "seed": (int,),
    }

    for name, value in config.items():
        if name not in expected_types:
            parser.error(f"Unknown configuration key in {path}: {name}")

        if type(value) not in expected_types[name]:
            expected = " or ".join(t.__name__ for t in expected_types[name])
            parser.error(f"{path}: {name} must be {expected}.")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny local SFT smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML experiment configuration.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model ID or local model directory.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/processed/gsm8k/train",
        help="Path to the processed training dataset.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/sft-tiny"),
        help="Directory for saving the model and tokenizer.",
    )
    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=32,
        help="Maximum number of training examples to select.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of examples per batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum optimizer steps within one pass through the dataset.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length after truncation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PyTorch and data shuffling.",
    )

    preliminary_args, _ = parser.parse_known_args()

    if preliminary_args.config is not None:
        config = load_config(preliminary_args.config, parser)
        parser.set_defaults(**config)

    args = parser.parse_args()

    if args.num_train_examples <= 0:
        parser.error("--num-train-examples must be positive.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive.")

    if args.max_length <= 0:
        parser.error("--max-length must be positive.")

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive.")

    return args


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
    args = parse_args()
    config_text = json.dumps(vars(args), indent=2, default=str)

    print("\nResolved configuration:")
    print(config_text)

    torch.manual_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    full_dataset = load_from_disk(args.dataset_path)

    num_examples = min(
        args.num_train_examples,
        len(full_dataset),
    )

    dataset = full_dataset.select(
        range(num_examples),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
    )

    model.to(device)
    model.train()

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(
            tokenizer,
            args.max_length,
        ),
        generator=data_generator,
    )

    print(f"Dataset size: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
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

        if step + 1 >= args.max_steps:
            break

    print("\nSaving checkpoint:")

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)

    config_path = args.checkpoint_dir / "run_config.json"
    config_path.write_text(config_text + "\n", encoding="utf-8")

    print(f"Saved checkpoint to {args.checkpoint_dir}")
    print(f"Saved run configuration to {config_path}")


if __name__ == "__main__":
    main()
