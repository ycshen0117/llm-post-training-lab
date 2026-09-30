import argparse
from pathlib import Path

from datasets import load_dataset

from llm_post_training_lab.data import (
    add_chat_messages,
    preprocess_gsm8k_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GSM8K chat training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gsm8k"),
        help="Output directory. The training split is saved under its train subdirectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("openai/gsm8k", "main")

    train = preprocess_gsm8k_dataset(dataset["train"])
    train = add_chat_messages(train)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train"
    train.save_to_disk(train_path)

    print(f"Saved {len(train)} examples to {train_path}")


if __name__ == "__main__":
    main()
