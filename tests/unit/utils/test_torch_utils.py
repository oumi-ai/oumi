import numpy as np
import pytest
import torch

from oumi.utils.torch_utils import (
    convert_to_list_of_tensors,
    create_ones_like,
    get_first_dim_len,
    get_torch_dtype,
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


def test_create_ones_from_empty():
    result = create_ones_like([])
    assert isinstance(result, list) and len(result) == 0

    result = create_ones_like(np.asarray([]))
    assert isinstance(result, np.ndarray) and result.shape == (0,)

    result = create_ones_like(torch.Tensor(np.asarray([])))
    assert isinstance(result, torch.Tensor) and result.shape == (0,)


def test_create_ones_from_none():
    with pytest.raises(ValueError, match="Unsupported type"):
        create_ones_like(None)


def test_create_ones_from_primitive():
    with pytest.raises(ValueError, match="Unsupported type"):
        create_ones_like(1)
    with pytest.raises(ValueError, match="Unsupported type"):
        create_ones_like(2.0)
    with pytest.raises(ValueError, match="Unsupported type"):
        create_ones_like("zzz")


def test_create_ones_like_inhomogeneous_shape():
    with pytest.raises(
        ValueError,
        match=(
            "setting an array element with a sequence. "
            "The requested array has an inhomogeneous shape after"
        ),
    ):
        create_ones_like([2, 3, [4, 5]])

    with pytest.raises(
        ValueError,
        match=(
            "setting an array element with a sequence. "
            "The requested array has an inhomogeneous shape after"
        ),
    ):
        create_ones_like([2, 3, np.asarray([4, 5]), 6])

    with pytest.raises(
        ValueError,
        match=(
            "setting an array element with a sequence. "
            "The requested array has an inhomogeneous shape after"
        ),
    ):
        create_ones_like([2, torch.Tensor([4, 5]), 7])


def test_create_ones_like_different_types():
    with pytest.raises(
        ValueError,
        match=("Sequence contains elements of different types"),
    ):
        create_ones_like([[4, 5], 1])

    with pytest.raises(
        ValueError,
        match=("Sequence contains elements of different types"),
    ):
        create_ones_like([[4, 5], np.asarray([6, 7])])

    with pytest.raises(
        ValueError,
        match=("Sequence contains elements of different types"),
    ):
        create_ones_like([np.asarray([6, 7]), torch.Tensor([8, 9])])

    with pytest.raises(
        ValueError,
        match=("Sequence contains elements of different types"),
    ):
        create_ones_like([torch.Tensor([8, 9]), [1, 2]])

    with pytest.raises(
        ValueError,
        match=("Sequence contains elements of different types"),
    ):
        create_ones_like([torch.Tensor([8, 9]), "str"])


def test_create_ones_like_success_list():
    result = create_ones_like([2])
    assert isinstance(result, list)
    assert np.all(np.asarray(result) == [1])

    result = create_ones_like([2, 3, 4])
    assert isinstance(result, list)
    assert np.all(np.asarray(result) == [1, 1, 1])

    result = create_ones_like([[2, 3], [4, 5]])
    assert isinstance(result, list)
    assert np.all(np.asarray(result) == np.asarray([[1, 1], [1, 1]]))


def test_create_ones_like_success_numpy():
    result = create_ones_like(np.asarray([2]))
    assert isinstance(result, np.ndarray)
    assert np.all(result == [1])

    result = create_ones_like(np.asarray([2, 3, 4]))
    assert isinstance(result, np.ndarray)
    assert np.all(np.asarray(result) == [1, 1, 1])

    result = create_ones_like([np.asarray([2, 3]), np.asarray([4, 5, 6])])
    assert isinstance(result, list)
    assert isinstance(result[0], np.ndarray)
    assert np.all(result[0] == np.asarray([1, 1]))
    assert isinstance(result[1], np.ndarray)
    assert np.all(result[1] == np.asarray([1, 1, 1]))


def test_create_ones_like_success_tensor():
    result = create_ones_like(torch.Tensor([2]))
    assert isinstance(result, torch.Tensor)
    assert np.all(result.numpy() == np.asarray([1]))

    result = create_ones_like(torch.Tensor([2, 3, 4]))
    assert isinstance(result, torch.Tensor)
    assert np.all(result.numpy() == np.asarray([1, 1, 1]))

    result = create_ones_like([torch.Tensor([2, 3]), torch.Tensor([4, 5, 6])])
    assert isinstance(result, list)
    assert isinstance(result[0], torch.Tensor)
    assert np.all(result[0].numpy() == np.asarray([1, 1]))
    assert isinstance(result[1], torch.Tensor)
    assert np.all(result[1].numpy() == np.asarray([1, 1, 1]))


@pytest.mark.parametrize(
    "dtype_str, expected_dtype",
    [
        ("f64", torch.float64),
        ("float64", torch.float64),
        ("double", torch.float64),
        ("f32", torch.float32),
        ("float32", torch.float32),
        ("float", torch.float32),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("f16", torch.float16),
        ("float16", torch.float16),
        ("half", torch.float16),
        ("uint8", torch.uint8),
    ],
)
def test_get_torch_dtype(dtype_str, expected_dtype):
    result = get_torch_dtype(dtype_str)
    assert result == expected_dtype


def test_get_torch_dtype_invalid():
    with pytest.raises(ValueError, match="Unsupported torch dtype: invalid_dtype"):
        get_torch_dtype("invalid_dtype")


def test_get_first_dim_len_list():
    assert get_first_dim_len([]) == 0
    assert get_first_dim_len([1]) == 1
    assert get_first_dim_len([1, 2, 3]) == 3
    assert get_first_dim_len([1, [[6, 9], 7], "abc"]) == 3
    assert get_first_dim_len([[[6, 9], 7], "abc"]) == 2


def test_get_first_dim_len_numpy_array():
    assert get_first_dim_len(np.asarray([])) == 0
    assert get_first_dim_len(np.asarray([1])) == 1
    assert get_first_dim_len(np.asarray([1, 2, 3])) == 3
    assert get_first_dim_len(np.asarray([[1, 2, 3]])) == 1
    assert get_first_dim_len(np.asarray([[1, 2, 3], [1, 2, 3]])) == 2
    assert get_first_dim_len(np.asarray([["a1", "a2", "a3"], ["x1", "x2", "x3"]])) == 2


def test_get_first_dim_len_torch_tensor():
    assert get_first_dim_len(torch.from_numpy(np.asarray([]))) == 0
    assert get_first_dim_len(torch.from_numpy(np.asarray([1]))) == 1
    assert get_first_dim_len(torch.from_numpy(np.asarray([1, 2, 3]))) == 3
    assert get_first_dim_len(torch.from_numpy(np.asarray([[1, 2, 3]]))) == 1
    assert get_first_dim_len(torch.from_numpy(np.asarray([[1, 2, 3], [1, 2, 3]]))) == 2


def test_get_first_dim_len_bad_input_type():
    with pytest.raises(ValueError, match="Unsupported type"):
        get_first_dim_len(None)
    with pytest.raises(ValueError, match="Unsupported type"):
        get_first_dim_len("hello")
    with pytest.raises(ValueError, match="Unsupported type"):
        get_first_dim_len(123)
    with pytest.raises(ValueError, match="Unsupported type"):
        get_first_dim_len(float(123))
    with pytest.raises(ValueError, match="Unsupported type"):
        get_first_dim_len(test_get_first_dim_len_bad_input_type)
