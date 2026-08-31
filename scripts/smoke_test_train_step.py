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

    full_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

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

    assert torch.equal(
        prompt_encoding["input_ids"],
        input_ids[:, :prompt_length],
    )

    labels = input_ids.clone()
    labels[:, :prompt_length] = IGNORE_INDEX

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
    )

    for step in range(10):
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        print(f"step={step:02d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
