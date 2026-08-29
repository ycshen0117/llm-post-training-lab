from datasets import load_dataset

from llm_post_training_lab.data import preprocess_gsm8k_example


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main")

    example = dataset["train"][0]

    print("Raw example:")
    print(example)
    print()

    processed = preprocess_gsm8k_example(example)

    print("Processed example:")
    print(processed)


if __name__ == "__main__":
    main()
