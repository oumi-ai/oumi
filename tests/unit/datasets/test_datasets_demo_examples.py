from pathlib import Path

from oumi.datasets import TextSftJsonLinesDataset


def test_data_format_loading():
    """Tests demo examples are correctly loaded in both json and jsonl formats."""
    current_dir = Path(__file__).resolve().parent
    data_top_dir = current_dir / "../../../data/dataset_examples"

    for format in ["alpaca", "oumi"]:
        all_data = []
        for ending in ["json", "jsonl"]:
            current = TextSftJsonLinesDataset(
                dataset_path=data_top_dir / (format + "_format." + ending)
            )
            all_data.append(current)

        json_datum = all_data[0]
        jsonl_datum = all_data[1]
        for i in range(len(json_datum)):
            assert json_datum.conversation(i) == jsonl_datum.conversation(i), (
                "Data from json and jsonl files should be the same by construction."
            )
