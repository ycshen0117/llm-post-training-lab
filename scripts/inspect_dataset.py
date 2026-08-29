import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main")

    train = dataset["train"]
    sample = train.select(range(5))

    output_path = Path("data/gsm8k_sample.jsonl")

    with output_path.open("w", encoding="utf-8") as file:
        for example in sample:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(sample)} examples to {output_path}")


if __name__ == "__main__":
    main()
