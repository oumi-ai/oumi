from pathlib import Path

import jsonlines
import pandas as pd
import pytest

from oumi.utils.io_utils import (
    get_oumi_root_directory,
    load_jsonlines,
    load_xlsx_all_sheets,
    save_jsonlines,
)


@pytest.fixture
def sample_data():
    return [
        {"name": "Space Needle", "height": 184},
        {"name": "Pike Place Market", "founded": 1907},
        {"name": "Seattle Aquarium", "opened": 1977},
    ]


@pytest.mark.parametrize("filename", ["train.py", "evaluate.py", "infer.py"])
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


# Tests for load_xlsx_all_sheets
@pytest.fixture
def sample_xlsx_single_sheet(tmp_path):
    """Create a temporary XLSX file with a single sheet."""
    file_path = tmp_path / "single_sheet.xlsx"
    xl_df = pd.DataFrame(
        {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NY", "LA"]}
    )
    xl_df.to_excel(file_path, sheet_name="Sheet1", index=False, engine="openpyxl")
    return file_path


@pytest.fixture
def sample_xlsx_multiple_sheets(tmp_path):
    """Create a temporary XLSX file with multiple sheets."""
    file_path = tmp_path / "multiple_sheets.xlsx"

    # Create multiple sheets with different data
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df1 = pd.DataFrame(
            {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NY", "LA"]}
        )
        df2 = pd.DataFrame(
            {"name": ["Charlie", "Diana"], "age": [35, 40], "city": ["SF", "CHI"]}
        )
        df3 = pd.DataFrame(
            {"name": ["Eve", "Frank"], "age": [45, 50], "city": ["SEA", "BOS"]}
        )
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Sheet2", index=False)
        df3.to_excel(writer, sheet_name="Sheet3", index=False)

    return file_path


@pytest.fixture
def expected_combined_dataframe():
    """Expected DataFrame when all sheets are combined."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
            "age": [25, 30, 35, 40, 45, 50],
            "city": ["NY", "LA", "SF", "CHI", "SEA", "BOS"],
        }
    )


def test_load_xlsx_all_sheets_single_sheet(sample_xlsx_single_sheet):
    """Test loading an XLSX file with a single sheet."""
    result = load_xlsx_all_sheets(sample_xlsx_single_sheet)

    expected = pd.DataFrame(
        {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NY", "LA"]}
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_load_xlsx_all_sheets_multiple_sheets(
    sample_xlsx_multiple_sheets, expected_combined_dataframe
):
    """Test loading an XLSX file with multiple sheets and concatenating them."""
    result = load_xlsx_all_sheets(sample_xlsx_multiple_sheets)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected_combined_dataframe
    )


def test_load_xlsx_all_sheets_file_not_found():
    """Test that FileNotFoundError is raised when file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_xlsx_all_sheets("non_existent_file.xlsx")


@pytest.fixture
def sample_xlsx_empty_sheets(tmp_path):
    """Create a temporary XLSX file with empty sheets."""
    file_path = tmp_path / "empty_sheets.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df_empty = pd.DataFrame()
        df_empty.to_excel(writer, sheet_name="EmptySheet", index=False)

    return file_path


def test_load_xlsx_all_sheets_empty_file(sample_xlsx_empty_sheets):
    """Test loading an XLSX file with empty sheets."""
    result = load_xlsx_all_sheets(sample_xlsx_empty_sheets)

    # Should return an empty DataFrame
    assert result.empty or len(result) == 0


@pytest.fixture
def sample_xlsx_different_columns(tmp_path):
    """Create an XLSX file with sheets having different columns."""
    file_path = tmp_path / "different_columns.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df1 = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df2 = pd.DataFrame({"name": ["Charlie", "Diana"], "salary": [50000, 60000]})
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Sheet2", index=False)

    return file_path


def test_load_xlsx_all_sheets_different_columns(sample_xlsx_different_columns):
    """Test loading sheets with different column structures."""
    result = load_xlsx_all_sheets(sample_xlsx_different_columns)

    # pandas concat will fill missing values with NaN
    assert len(result) == 4  # Total of 4 rows
    assert "name" in result.columns
    assert "age" in result.columns
    assert "salary" in result.columns

    # Check that NaN values are present where columns don't match
    assert pd.isna(result.iloc[0]["salary"])  # First sheet has no salary
    assert pd.isna(result.iloc[2]["age"])  # Second sheet has no age
