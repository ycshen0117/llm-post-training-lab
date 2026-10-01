import argparse
import json
import math
import time
import tomllib
from contextlib import nullcontext
from itertools import islice
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
        "gradient_accumulation_steps": (int,),
        "learning_rate": (int, float),
        "max_steps": (int,),
        "max_length": (int,),
        "seed": (int,),
        "device": (str,),
        "precision": (str,),
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
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of microbatches per optimizer update.",
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
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Training device. Explicit requests must be available.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Training precision. BF16 requires a supported CUDA GPU.",
    )

    preliminary_args, _ = parser.parse_known_args()

    if preliminary_args.config is not None:
        config = load_config(preliminary_args.config, parser)
        parser.set_defaults(**config)

    args = parser.parse_args()

    if args.device not in ("auto", "cpu", "mps", "cuda"):
        parser.error("Invalid device in configuration.")

    if args.precision not in ("fp32", "bf16"):
        parser.error("Invalid precision in configuration.")

    if args.num_train_examples <= 0:
        parser.error("--num-train-examples must be positive.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive.")

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive.")

    if args.max_length <= 0:
        parser.error("--max-length must be positive.")

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive.")

    return args


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    return torch.device(requested)


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


def train_sft(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_steps: int,
    gradient_accumulation_steps: int,
    precision: str,
) -> dict:
    batch_iterator = iter(dataloader)
    optimizer_steps = 0
    microbatches = 0
    examples = 0
    supervised_tokens = 0
    token_loss_sum = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    started_at = time.perf_counter()

    for step in range(max_steps):
        # Buffer only CPU batches. Each batch moves to the GPU separately.
        window = list(islice(batch_iterator, gradient_accumulation_steps))
        if not window:
            break

        # Causal LM loss predicts labels[1:] from the preceding positions.
        token_counts = [
            int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())
            for batch in window
        ]
        if any(count == 0 for count in token_counts):
            raise ValueError("A microbatch has no supervised next-token targets.")
        window_tokens = sum(token_counts)
        window_examples = 0
        window_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch, token_count in zip(window, token_counts, strict=True):
            if microbatches == 0:
                inspect_batch(batch)
            batch = {key: value.to(device) for key, value in batch.items()}
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )
            with amp_context:
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss).item():
                raise ValueError(f"Non-finite loss at microbatch {microbatches}.")

            # Normalize by tokens, including incomplete final windows.
            (loss * (token_count / window_tokens)).backward()
            window_loss_sum += loss.item() * token_count
            window_examples += batch["input_ids"].shape[0]
            microbatches += 1
            del outputs, loss, batch

        optimizer.step()
        optimizer_steps += 1
        examples += window_examples
        supervised_tokens += window_tokens
        token_loss_sum += window_loss_sum
        print(
            f"optimizer_step={step + 1:03d} microbatches={len(window)} "
            f"examples={window_examples} tokens={window_tokens} "
            f"loss={window_loss_sum / window_tokens:.4f}",
            flush=True,
        )

    if optimizer_steps == 0:
        raise ValueError("No optimizer updates were performed.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    train_seconds = time.perf_counter() - started_at

    return {
        "optimizer_steps": optimizer_steps,
        "microbatches": microbatches,
        "examples": examples,
        "supervised_tokens": supervised_tokens,
        "mean_token_loss": token_loss_sum / supervised_tokens,
        "train_seconds": train_seconds,
        "seconds_per_optimizer_step": train_seconds / optimizer_steps,
        "examples_per_second": examples / train_seconds,
        "supervised_tokens_per_second": supervised_tokens / train_seconds,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    config_text = json.dumps(vars(args), indent=2, default=str)

    print("\nResolved configuration:")
    print(config_text)

    torch.manual_seed(args.seed)
    device = get_device(args.device)

    if args.precision == "bf16":
        if device.type != "cuda":
            raise RuntimeError("BF16 training requires CUDA in this script.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA GPU does not support BF16.")

    print(f"Using device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Random seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Microbatch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(
        "Nominal effective batch size (single GPU): "
        f"{args.batch_size * args.gradient_accumulation_steps}"
    )
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
        dtype=torch.float32,
    )

    model.to(device)
    model.config.use_cache = False
    model.train()

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Parameter dtype: {next(model.parameters()).dtype}")

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
    available_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    print(f"Available optimizer steps in one pass: {available_steps}")
    if args.max_steps > available_steps:
        print("Dataset will end before max_steps; training stops after one pass.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    print("\nTraining:")

    metrics = train_sft(
        model,
        dataloader,
        optimizer,
        device,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        precision=args.precision,
    )
    print("\nTraining metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nSaving checkpoint:")

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)

    config_path = args.checkpoint_dir / "run_config.json"
    config_path.write_text(config_text + "\n", encoding="utf-8")

    metrics_path = args.checkpoint_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved checkpoint to {args.checkpoint_dir}")
    print(f"Saved run configuration to {config_path}")
    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
