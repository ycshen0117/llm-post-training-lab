import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is 17 + 25? Show your reasoning.",
    (
        "A box contains 12 red balls and 8 blue balls. "
        "How many balls are in the box altogether? "
        "Show your reasoning."
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and SFT model responses on fixed prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model ID or local directory for the pre-SFT reference model.",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=Path("checkpoints/sft-tiny"),
        help="Local directory containing the SFT model and tokenizer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens per response.",
    )

    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive.")

    if not args.sft_checkpoint.is_dir():
        parser.error(
            f"--sft-checkpoint must be an existing directory: {args.sft_checkpoint}"
        )

    return args


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
    max_new_tokens: int,
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
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Max new tokens: {args.max_new_tokens}")

    print(f"\nLoading base model: {args.base_model_id}")

    base_responses = generate_responses(
        model_path=args.base_model_id,
        prompts=PROMPTS,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    clear_device_cache(device)

    print(f"\nLoading SFT checkpoint: {args.sft_checkpoint}")

    sft_responses = generate_responses(
        model_path=args.sft_checkpoint,
        prompts=PROMPTS,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    clear_device_cache(device)

    print_comparison(
        PROMPTS,
        base_responses,
        sft_responses,
    )


if __name__ == "__main__":
    main()
