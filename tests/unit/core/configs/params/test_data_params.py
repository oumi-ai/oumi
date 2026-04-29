import dataclasses
from pathlib import Path

import pytest

from oumi.core.configs.params.data_params import DatasetParams
from oumi.exceptions import OumiConfigError


def _get_invalid_field_name_lists() -> list[list[str]]:
    all_fields: set[str] = {f.name for f in dataclasses.fields(DatasetParams())}
    result = [[field_name] for field_name in all_fields]
    result.extend([["valid_kwarg", field_name] for field_name in all_fields][:3])
    return result


def _get_test_name_for_invalid_field_name_list(x):
    assert isinstance(x, list)
    return "--".join(x)


@pytest.mark.parametrize(
    "field_names",
    _get_invalid_field_name_lists(),
    ids=_get_test_name_for_invalid_field_name_list,
)
def test_dataset_params_reserved_kwargs(field_names: list[str]):
    invalid_names = {f.name for f in dataclasses.fields(DatasetParams())}.intersection(
        field_names
    )
    with pytest.raises(
        ValueError,
        match=(
            "dataset_kwargs attempts to override the following reserved fields: "
            f"{invalid_names}"
        ),
    ):
        DatasetParams(
            dataset_name="DUMMY-NON-EXISTENT",
            dataset_kwargs={field_name: "foo_value" for field_name in field_names},
        )


def test_dataset_params_finalize_nonexistent_path():
    params = DatasetParams(
        dataset_name="some_dataset",
        dataset_path="/nonexistent/path/to/data.jsonl",
    )
    with pytest.raises(
        OumiConfigError,
        match="dataset_path '/nonexistent/path/to/data.jsonl' does not exist",
    ):
        params.finalize_and_validate()


def test_dataset_params_finalize_existing_path(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("{}\n")
    params = DatasetParams(
        dataset_name="some_dataset",
        dataset_path=str(data_file),
    )
    params.finalize_and_validate()


def test_dataset_params_finalize_no_path():
    params = DatasetParams(dataset_name="some_dataset")
    params.finalize_and_validate()


def test_dataset_params_finalize_existing_directory(tmp_path: Path):
    params = DatasetParams(
        dataset_name="some_dataset",
        dataset_path=str(tmp_path),
    )
    params.finalize_and_validate()
