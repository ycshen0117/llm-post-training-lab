from datasets import Dataset


def preprocess_gsm8k_example(example: dict[str, str]) -> dict[str, str]:
    prompt = example["question"].strip()
    response = example["answer"].strip()

    if not prompt:
        raise ValueError("GSM8K example has an empty question")

    if not response:
        raise ValueError("GSM8K example has an empty answer")

    return {
        "prompt": prompt,
        "response": response,
    }


def preprocess_gsm8k_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(
        preprocess_gsm8k_example,
        remove_columns=dataset.column_names,
    )
