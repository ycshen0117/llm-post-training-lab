from pathlib import Path

from datasets import load_dataset

from llm_post_training_lab.data import (
    add_chat_messages,
    preprocess_gsm8k_dataset,
)


OUTPUT_DIR = Path("data/processed/gsm8k")


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main")

    train = preprocess_gsm8k_dataset(dataset["train"])
    train = add_chat_messages(train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.save_to_disk(OUTPUT_DIR / "train")

    print(f"Saved {len(train)} examples to {OUTPUT_DIR / 'train'}")


if __name__ == "__main__":
    main()
