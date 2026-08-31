from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
        {
            "role": "assistant",
            "content": "2 + 2 = 4.\n#### 4",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=128,
    )

    print("Formatted text:")
    print(text)
    print()
    print("Input IDs:")
    print(encoded["input_ids"])
    print()
    print("Tokens:")
    print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print()
    print("Sequence length:")
    print(len(encoded["input_ids"]))


if __name__ == "__main__":
    main()
