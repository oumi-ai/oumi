import pytest
import torch

from oumi.core.datasets.base_iterable_dataset import _convert_tensors_for_arrow


def test_convert_scalar_tensor_returns_python_int():
    tensor = torch.tensor(1212)
    result = _convert_tensors_for_arrow(tensor)
    assert result == 1212
    assert type(result) is int


def test_convert_scalar_float_tensor_returns_python_float():
    tensor = torch.tensor(3.14)
    result = _convert_tensors_for_arrow(tensor)
    assert isinstance(result, float)
    assert result == pytest.approx(3.14)


def test_convert_1d_tensor_returns_python_list():
    tensor = torch.tensor([1, 2, 3])
    result = _convert_tensors_for_arrow(tensor)
    assert result == [1, 2, 3]
    assert isinstance(result, list)
    assert all(type(x) is int for x in result)


def test_convert_2d_tensor_returns_nested_list():
    tensor = torch.tensor([[1, 2], [3, 4]])
    result = _convert_tensors_for_arrow(tensor)
    assert result == [[1, 2], [3, 4]]


def test_convert_dict_with_tensors():
    item = {
        "input_ids": torch.tensor([10, 20, 30]),
        "labels": torch.tensor([1]),
    }
    result = _convert_tensors_for_arrow(item)
    assert result == {"input_ids": [10, 20, 30], "labels": [1]}
    assert isinstance(result["input_ids"], list)


def test_convert_nested_dict():
    item = {"outer": {"inner": torch.tensor([5, 6])}}
    result = _convert_tensors_for_arrow(item)
    assert result == {"outer": {"inner": [5, 6]}}


def test_convert_list_of_tensors():
    item = [torch.tensor(1), torch.tensor(2)]
    result = _convert_tensors_for_arrow(item)
    assert result == [1, 2]
    assert all(type(x) is int for x in result)


def test_convert_plain_python_objects_unchanged():
    assert _convert_tensors_for_arrow(42) == 42
    assert _convert_tensors_for_arrow("hello") == "hello"
    assert _convert_tensors_for_arrow([1, 2]) == [1, 2]
    assert _convert_tensors_for_arrow({"a": 1}) == {"a": 1}


def test_convert_mixed_dict_with_tensors_and_plain_values():
    item = {
        "input_ids": torch.tensor([10, 20]),
        "text": "hello",
        "count": 5,
    }
    result = _convert_tensors_for_arrow(item)
    assert result == {"input_ids": [10, 20], "text": "hello", "count": 5}
