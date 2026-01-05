import datasets
import pandas as pd
import pytest

from oumi.core.datasets.base_map_dataset import BaseMapDataset


class MyTestDataset(BaseMapDataset):
    def __init__(self, *args, **kwargs):
        self._data = self._default_mock_data()
        super().__init__(*args, **kwargs)

    def _load_data(self):
        return self.mock_data

    def transform(self, sample: pd.Series) -> dict:
        result = sample.to_dict()
        computed_dict = {}
        for key, value in result.items():
            if isinstance(value, str | list):
                computed_dict[f"{key}_len"] = len(value)
        result.update(computed_dict)
        return result

    def _default_mock_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"text": "This is a test sentence."},
                {"text": "Another example sentence for testing."},
                {"text": "A third sentence to ensure we have enough data."},
            ]
        )


@pytest.mark.parametrize(
    "stream",
    [
        False,
        True,
    ],
)
def test_to_hf_basic(stream: bool):
    custom_dataset = MyTestDataset(dataset_name=f"foo_stream_{stream}")
    dataset = custom_dataset.to_hf(return_iterable=stream)

    if stream:
        assert isinstance(dataset, datasets.IterableDataset)
    else:
        assert isinstance(dataset, datasets.Dataset)


def test_load_xlsx_dataset(tmp_path):
    """Test loading XLSX dataset."""
    # Create test XLSX file
    test_data = pd.DataFrame(
        {
            "text": ["sample1", "sample2", "sample3"],
            "label": [0, 1, 0],
        }
    )
    xlsx_file = tmp_path / "test.xlsx"
    test_data.to_excel(xlsx_file, index=False, engine="openpyxl")

    # Create a test dataset class
    class TestXlsxDataset(BaseMapDataset):
        def __init__(self, **kwargs):
            super().__init__(dataset_name="test", dataset_path=str(xlsx_file), **kwargs)
            self._data = self._load_data()

        def transform(self, sample):
            return {"text": sample["text"], "label": sample["label"]}

    # Test loading
    dataset = TestXlsxDataset()
    assert len(dataset) == 3
    assert dataset[0]["text"] == "sample1"
    assert dataset[1]["label"] == 1
