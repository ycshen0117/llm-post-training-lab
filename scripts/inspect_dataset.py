from datasets import load_dataset


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main")

    print(dataset)
    print()
    print("First training example:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()
