import pytest
from datasets import Dataset

from llm_post_training_lab.data import (
    preprocess_gsm8k_dataset,
    preprocess_gsm8k_example,
    to_chat_messages,
)


def test_preprocess_gsm8k_example() -> None:
    example = {
        "question": "What is 2 + 2? ",
        "answer": " 2 + 2 = 4.\n#### 4 ",
    }

    result = preprocess_gsm8k_example(example)

    assert result == {
        "prompt": "What is 2 + 2?",
        "response": "2 + 2 = 4.\n#### 4",
    }


def test_preprocess_gsm8k_example_rejects_empty_question() -> None:
    example = {
        "question": "   ",
        "answer": "#### 4",
    }

    with pytest.raises(ValueError, match="empty question"):
        preprocess_gsm8k_example(example)


def test_preprocess_gsm8k_dataset() -> None:
    dataset = Dataset.from_dict(
        {
            "question": ["What is 1 + 1?"],
            "answer": ["1 + 1 = 2.\n#### 2"],
        }
    )

    processed = preprocess_gsm8k_dataset(dataset)

    assert processed.column_names == ["prompt", "response"]
    assert processed[0]["prompt"] == "What is 1 + 1?"
    assert processed[0]["response"] == "1 + 1 = 2.\n#### 2"


def test_to_chat_messages() -> None:
    example = {
        "prompt": "What is 2 + 2?",
        "response": "2 + 2 = 4.\n#### 4",
    }

    result = to_chat_messages(example)

    assert result == {
        "messages": [
            {
                "role": "user",
                "content": "What is 2 + 2?",
            },
            {
                "role": "assistant",
                "content": "2 + 2 = 4.\n#### 4",
            },
        ]
    }
