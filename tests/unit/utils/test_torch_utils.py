import numpy as np
import pytest
import torch

from oumi.utils.torch_utils import (
    convert_to_list_of_tensors,
    pad_sequences,
    pad_sequences_left_side,
    pad_sequences_right_side,
)


def test_convert_to_list_of_tensors_empty_list():
    result = convert_to_list_of_tensors([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_convert_to_list_of_tensors_from_lists():
    result = convert_to_list_of_tensors([[1, 2, 3, 4], [5], [6, 7]])
    assert isinstance(result, list)
    assert len(result) == 3

    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == (4,)
    assert result[0].dtype == torch.int64
    assert np.all(result[0].numpy() == np.asarray([1, 2, 3, 4]))

    assert result[1].shape == (1,)
    assert result[1].dtype == torch.int64
    assert np.all(result[1].numpy() == np.asarray([5]))

    assert result[2].shape == (2,)
    assert result[2].dtype == torch.int64
    assert np.all(result[2].numpy() == np.asarray([6, 7]))


def test_convert_to_list_of_tensors_from_ndarrays():
    result = convert_to_list_of_tensors(
        [np.asarray([1, 2, 3, 4]), np.asarray([5]), np.asarray([6, 7])]
    )
    assert isinstance(result, list)
    assert len(result) == 3

    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == (4,)
    assert result[0].dtype == torch.int64
    assert np.all(result[0].numpy() == np.asarray([1, 2, 3, 4]))

    assert result[1].shape == (1,)
    assert result[1].dtype == torch.int64
    assert np.all(result[1].numpy() == np.asarray([5]))

    assert result[2].shape == (2,)
    assert result[2].dtype == torch.int64
    assert np.all(result[2].numpy() == np.asarray([6, 7]))


def test_convert_to_list_of_tensors_from_tensors():
    result = convert_to_list_of_tensors(
        [
            torch.from_numpy(np.asarray([1, 2, 3, 4])),
            torch.from_numpy(np.asarray([5])),
            torch.from_numpy(np.asarray([6, 7])),
        ]
    )
    assert isinstance(result, list)
    assert len(result) == 3

    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == (4,)
    assert result[0].dtype == torch.int64
    assert np.all(result[0].numpy() == np.asarray([1, 2, 3, 4]))

    assert result[1].shape == (1,)
    assert result[1].dtype == torch.int64
    assert np.all(result[1].numpy() == np.asarray([5]))

    assert result[2].shape == (2,)
    assert result[2].dtype == torch.int64
    assert np.all(result[2].numpy() == np.asarray([6, 7]))


@pytest.mark.parametrize(
    "padding_value",
    [0, -100, 7],
)
def test_pad_sequences_right_side(padding_value: int):
    test_sequences = [[1, 2, 3, 4], [5], [6, 7]]
    result = pad_sequences_right_side(test_sequences, padding_value=padding_value)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 4)

    pad = padding_value
    assert np.all(
        result.numpy()
        == np.asarray([[1, 2, 3, 4], [5, pad, pad, pad], [6, 7, pad, pad]])
    )

    assert np.all(
        result.numpy()
        == pad_sequences(
            test_sequences, padding_side="right", padding_value=padding_value
        ).numpy()
    )


@pytest.mark.parametrize(
    "padding_value",
    [0, -100, 7],
)
def test_pad_sequences_left_side(padding_value: int):
    test_sequences = [[1, 2, 3, 4], [5], [6, 7]]
    result = pad_sequences_left_side(test_sequences, padding_value=padding_value)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 4)

    pad = padding_value
    assert np.all(
        result.numpy()
        == np.asarray([[1, 2, 3, 4], [pad, pad, pad, 5], [pad, pad, 6, 7]])
    )

    assert np.all(
        result.numpy()
        == pad_sequences(
            test_sequences, padding_side="left", padding_value=padding_value
        ).numpy()
    )
