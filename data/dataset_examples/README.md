## Supported Dataset Formats

1. **Oumi Conversation Format**
   Oumi natively supports the `conversation` format, used internally across SFT, Preference Tuning, and Vision-Language datasets. Each example consists of a JSON object with a list of messages, supporting both single and multi-turn conversations. Optionally, `metadata` can be included to provide additional details. The primary class for loading such datasets is `oumi.datasets.TextSftJsonLinesDataset`.
   - A short example of a conversation-formatted dataset is shown [here](./oumi_format.json), stored in a `json` file.
   - The same dataset can also be stored as a `jsonl` file – see [here](./oumi_format.jsonl).

2. **Alpaca Format**
   The `alpaca instruction` format offers broader compatibility with existing libraries but lacks support for multi-turn conversations and per-example metadata.
   - A small dataset in `alpaca instruction` format is stored as a `json` file [here](./alpaca_format.json), and as a `jsonl` file [here](./alpaca_format.jsonl).


## Example Usage

Below is an example of how to load and iterate over the supported dataset formats using `TextSftJsonLinesDataset` from the `oumi.datasets` module.

```python
from pathlib import Path
from oumi.datasets import TextSftJsonLinesDataset

# Iterate through different dataset formats and file types
for dataset_format in ["alpaca", "oumi"]:
    for file_extension in ["json", "jsonl"]:
        example_dataset_path = Path(f"./{dataset_format}_format.{file_extension}")

        # Load the dataset
        dataset = TextSftJsonLinesDataset(dataset_path=example_dataset_path)

        # Iterate through the dataset and print conversations
        for i in range(len(dataset)):
            print(dataset.conversation(i))
```
