import copy
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

IGNORE_INDEX = -100


@pytest.fixture(scope="module")
def training_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_sft_tiny.py"
    return runpy.run_path(str(path))


def make_model():
    """Instantiate a tiny real Qwen model locally, without downloading weights."""
    config = Qwen2Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
    )
    return AutoModelForCausalLM.from_config(config)


def make_examples():
    examples = []
    # Different answer lengths expose incorrect averaging of microbatch means.
    for response_length in (1, 3, 2, 4, 1):
        ids = [1, 2] + [3, 4, 5, 6][:response_length]
        padding = 6 - len(ids)
        examples.append(
            {
                "input_ids": torch.tensor(ids + [0] * padding),
                "attention_mask": torch.tensor([1] * len(ids) + [0] * padding),
                "labels": torch.tensor(
                    [IGNORE_INDEX, IGNORE_INDEX] + ids[2:] + [IGNORE_INDEX] * padding
                ),
            }
        )
    return examples


def run_training(training_script, model, examples, batch_size, accumulation, max_steps):
    return training_script["train_sft"](
        model,
        DataLoader(examples, batch_size=batch_size, shuffle=False),
        # SGD isolates gradient equivalence from Adam's near-zero-gradient sensitivity.
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.device("cpu"),
        max_steps=max_steps,
        gradient_accumulation_steps=accumulation,
        precision="fp32",
    )


def test_accumulation_matches_large_batches_and_flushes_remainder(training_script):
    torch.manual_seed(42)
    accumulated = make_model()
    reference = copy.deepcopy(accumulated)
    examples = make_examples()

    metrics = run_training(training_script, accumulated, examples, 1, 2, 10)
    reference_metrics = run_training(training_script, reference, examples, 2, 1, 10)

    # Five microbatches form two full updates and one final partial update.
    assert metrics["optimizer_steps"] == 3
    assert metrics["microbatches"] == 5
    assert metrics["examples"] == 5
    assert metrics["supervised_tokens"] == 11
    assert metrics["mean_token_loss"] == pytest.approx(
        reference_metrics["mean_token_loss"], rel=1e-6
    )
    assert metrics["train_seconds"] > 0
    assert metrics["peak_allocated_gib"] is None
    for parameter, expected in zip(
        accumulated.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


def test_max_steps_limits_optimizer_updates(training_script):
    torch.manual_seed(42)
    accumulated = make_model()
    reference = copy.deepcopy(accumulated)
    examples = make_examples()

    metrics = run_training(training_script, accumulated, examples, 1, 2, 1)
    run_training(training_script, reference, examples[:2], 2, 1, 1)

    assert metrics["optimizer_steps"] == 1
    assert metrics["microbatches"] == 2
    assert metrics["examples"] == 2
    for parameter, expected in zip(
        accumulated.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_accumulation(training_script, monkeypatch, value):
    monkeypatch.setattr(
        sys, "argv", ["train_sft_tiny.py", "--gradient-accumulation-steps", value]
    )
    with pytest.raises(SystemExit) as error:
        training_script["parse_args"]()
    assert error.value.code == 2


def test_accumulation_config_and_cli_precedence(training_script, monkeypatch, tmp_path):
    config = tmp_path / "sft.toml"
    config.write_text("gradient_accumulation_steps = 3\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_sft_tiny.py", "--config", str(config)])
    assert training_script["parse_args"]().gradient_accumulation_steps == 3
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft_tiny.py",
            "--config",
            str(config),
            "--gradient-accumulation-steps",
            "4",
        ],
    )
    assert training_script["parse_args"]().gradient_accumulation_steps == 4


def test_main_saves_reloadable_checkpoint_and_actual_metrics(
    training_script, monkeypatch, tmp_path
):
    model_dir = tmp_path / "model"
    make_model().save_pretrained(model_dir)
    backend = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "user": 2,
                "assistant": 3,
                "question": 4,
                "answer": 5,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        chat_template=(
            "{% for message in messages %}"
            "{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )
    tokenizer.save_pretrained(model_dir)
    data_dir = tmp_path / "data"
    Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer " * length},
                ]
                for length in (1, 3, 2, 4, 1)
            ]
        }
    ).save_to_disk(data_dir)
    checkpoint = tmp_path / "checkpoint"
    # Use our local word tokenizer instead of AutoTokenizer's Qwen-specific loader.
    monkeypatch.setattr(
        training_script["AutoTokenizer"], "from_pretrained", lambda _: tokenizer
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft_tiny.py",
            "--model-id",
            str(model_dir),
            "--dataset-path",
            str(data_dir),
            "--checkpoint-dir",
            str(checkpoint),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "2",
            "--max-steps",
            "3",
        ],
    )
    training_script["main"]()
    metrics = json.loads((checkpoint / "train_metrics.json").read_text())
    assert metrics["optimizer_steps"] == 3
    assert metrics["microbatches"] == 5
    assert metrics["examples"] == 5
    assert metrics["supervised_tokens"] == 11
    config = json.loads((checkpoint / "run_config.json").read_text())
    assert config["gradient_accumulation_steps"] == 2
    assert config["checkpoint_dir"] == str(checkpoint)
    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint)
    assert reloaded.config.vocab_size == 8
    assert reloaded.dtype == torch.float32
