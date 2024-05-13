from datasets import load_dataset

from lema.datasets.alpaca import alpaca_preprocessing_fn  # TODO: pull from registry


def build_prompt_generation_fn(function_name, tokenizer):
    """Build a prompt generation function."""

    # TODO: pull from registry
    if function_name == "alpaca":
        return alpaca_preprocessing_fn(tokenizer)

    raise ValueError(f"Unknown prompt generation function: {function_name}")


def build_dataset(dataset_name, preprocessing_function_name, tokenizer, **kwargs):
    """Build a dataset for training."""

    # TODO: should return all splits
    dataset = load_dataset(dataset_name, split="train")

    preprocessing_fn = build_prompt_generation_fn(
        preprocessing_function_name, tokenizer
    )

    dataset = dataset.map(preprocessing_fn, batched=True, **kwargs)

    return dataset
