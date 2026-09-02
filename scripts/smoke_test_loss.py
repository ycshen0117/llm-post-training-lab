import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
IGNORE_INDEX = -100


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def inspect_labels(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    label_values = labels[0].tolist()

    print("\nToken-level supervision:")
    print(f"{'idx':>4}  {'token':<30}  {'label':>8}  loss?")
    print("-" * 60)

    for idx, (token, label) in enumerate(zip(tokens, label_values)):
        contributes_to_loss = label != IGNORE_INDEX

        print(f"{idx:>4}  {token:<30}  {label:>8}  {contributes_to_loss}")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    model.to(device)
    model.train()

    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
        {
            "role": "assistant",
            "content": "Paris.",
        },
    ]

    # Full training sequence:
    # user + assistant answer
    full_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    # Prompt prefix:
    # user + assistant-generation header
    prompt_encoding = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]

    prompt_length = prompt_encoding["input_ids"].shape[1]

    prompt_ids = prompt_encoding["input_ids"]
    full_prompt_prefix = full_encoding["input_ids"][:, :prompt_length]

    assert torch.equal(prompt_ids, full_prompt_prefix), (
        "Prompt tokenization is not a prefix of the full conversation."
    )

    labels = input_ids.clone()
    labels[:, :prompt_length] = IGNORE_INDEX

    inspect_labels(
        tokenizer,
        input_ids,
        labels,
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Prompt length: {prompt_length}")
    print(f"Assistant-only loss: {outputs.loss.item():.4f}")


if __name__ == "__main__":
    main()
