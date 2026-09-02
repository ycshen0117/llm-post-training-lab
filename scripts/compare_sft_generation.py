from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_CHECKPOINT = Path("checkpoints/sft-tiny")

MAX_NEW_TOKENS = 128

PROMPTS = [
    "What is 17 + 25? Show your reasoning.",
    (
        "A box contains 12 red balls and 8 blue balls. "
        "How many balls are in the box altogether? "
        "Show your reasoning."
    ),
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def generate_responses(
    model_path: str | Path,
    prompts: list[str],
    device: torch.device,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
    )

    model.to(device)
    model.eval()

    responses = []

    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[1]

        generated_tokens = output_ids[
            0,
            prompt_length:,
        ]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        responses.append(response)

    return responses


def clear_device_cache(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if device.type == "mps":
        torch.mps.empty_cache()


def print_comparison(
    prompts: list[str],
    base_responses: list[str],
    sft_responses: list[str],
) -> None:
    for index, prompt in enumerate(prompts, start=1):
        base_response = base_responses[index - 1]
        sft_response = sft_responses[index - 1]

        print(f"\n{'=' * 72}")
        print(f"Example {index}")
        print(f"{'=' * 72}")

        print("\nPrompt:")
        print(prompt)

        print("\nBase response:")
        print(base_response)

        print("\nSFT response:")
        print(sft_response)


def main() -> None:
    if not SFT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SFT checkpoint does not exist: {SFT_CHECKPOINT}. "
            "Run scripts/train_sft_tiny.py first."
        )

    device = get_device()
    print(f"Using device: {device}")

    print(f"\nLoading base model: {BASE_MODEL_ID}")

    base_responses = generate_responses(
        BASE_MODEL_ID,
        PROMPTS,
        device,
    )

    clear_device_cache(device)

    print(f"\nLoading SFT checkpoint: {SFT_CHECKPOINT}")

    sft_responses = generate_responses(
        SFT_CHECKPOINT,
        PROMPTS,
        device,
    )

    clear_device_cache(device)

    print_comparison(
        PROMPTS,
        base_responses,
        sft_responses,
    )


if __name__ == "__main__":
    main()
