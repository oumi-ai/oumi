from pathlib import Path

import jsonlines
import pytest

from oumi.utils.io_utils import get_oumi_root_directory, load_jsonlines, save_jsonlines


@pytest.fixture
def sample_data():
    return [
        {"name": "Space Needle", "height": 184},
        {"name": "Pike Place Market", "founded": 1907},
        {"name": "Seattle Aquarium", "opened": 1977},
    ]


@pytest.mark.parametrize(
    "filename", ["train.py", "evaluate.py", "infer.py", "judge.py"]
)
def test_get_oumi_root_directory(filename):
    root_dir = get_oumi_root_directory()
    file_path = root_dir / filename
    assert file_path.exists(), f"{file_path} does not exist in the root directory."


def test_load_jsonlines_successful(tmp_path, sample_data):
    file_path = tmp_path / "test.jsonl"
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(sample_data)

    result = load_jsonlines(file_path)
    assert result == sample_data


def test_load_jsonlines_file_not_found():
    with pytest.raises(FileNotFoundError, match="Provided path does not exist"):
        load_jsonlines("non_existent_file.jsonl")


def test_load_jsonlines_directory_path(tmp_path):
    with pytest.raises(
        ValueError, match="Provided path is a directory, expected a file"
    ):
        load_jsonlines(tmp_path)


def test_load_jsonlines_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.jsonl"
    with open(file_path, "w") as f:
        f.write('{"valid": "json"}\n{"invalid": json}\n')

    with pytest.raises(jsonlines.InvalidLineError):
        load_jsonlines(file_path)


def test_save_jsonlines_successful(tmp_path, sample_data):
    file_path = tmp_path / "output.jsonl"
    save_jsonlines(file_path, sample_data)

    with jsonlines.open(file_path) as reader:
        saved_data = list(reader)
    assert saved_data == sample_data


def test_save_jsonlines_io_error(tmp_path, sample_data, monkeypatch):
    file_path = tmp_path / "test.jsonl"

    # Mock the open function to raise an OSError
    def mock_open(*args, **kwargs):
        raise OSError("Mocked IO error")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(OSError):
        save_jsonlines(file_path, sample_data)


def test_load_and_save_with_path_object(tmp_path, sample_data):
    file_path = Path(tmp_path) / "test_path.jsonl"

    save_jsonlines(file_path, sample_data)
    loaded_data = load_jsonlines(file_path)

    assert loaded_data == sample_data
