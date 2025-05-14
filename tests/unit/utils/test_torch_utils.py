import re

import numpy as np
import pytest
import torch

from oumi.utils.torch_utils import (
    ModelParameterCount,
    convert_to_list_of_tensors,
    create_ones_like,
    estimate_sample_dict_size_in_bytes,
    freeze_model_layers,
    get_dtype_size_in_bytes,
    get_first_dim_len,
    get_torch_dtype,
    pad_sequences,
    pad_sequences_left_side,
    pad_sequences_right_side,
    pad_to_max_dim_and_stack,
)


def test_valid_model_parameter_count():
    mpc = ModelParameterCount(
        all_params=1000, trainable_params=800, embedding_params=200
    )
    assert mpc.trainable_params_percent == 80.0
    assert mpc.frozen_params_percent == 20.0


def test_model_parameter_count_zero_params():
    mpc = ModelParameterCount(all_params=0, trainable_params=0, embedding_params=0)
    assert mpc.trainable_params_percent == 0.0
    assert mpc.frozen_params_percent == 100.0


@pytest.mark.parametrize(
    "all_params, trainable_params, embedding_params, error_field",
    [
        (-1, 0, 0, "all_params"),
        (100, -5, 0, "trainable_params"),
        (100, 0, -3, "embedding_params"),
    ],
)
def test_model_parameter_count_negative_values(
    all_params, trainable_params, embedding_params, error_field
):
    with pytest.raises(ValueError, match=rf"`{error_field}`.*must be >= 0\."):
        ModelParameterCount(all_params, trainable_params, embedding_params)


def test_model_parameter_relative_sizes():
    with pytest.raises(
        ValueError,
        match=(
            r"`trainable_params` \(200\) cannot be greater than "
            r"`all_params` \(100\)\."
        ),
    ):
        ModelParameterCount(all_params=100, trainable_params=200, embedding_params=50)

    with pytest.raises(
        ValueError,
        match=(
            r"`embedding_params` \(150\) cannot be greater than "
            r"`all_params` \(100\)\."
        ),
    ):
        ModelParameterCount(all_params=100, trainable_params=80, embedding_params=150)


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


def test_pad_sequences_invalid_side():
    test_sequences = [[1, 2, 3, 4], [5], [6, 7]]
    with pytest.raises(ValueError, match="Unsupported padding side: 'bottom'"):
        pad_sequences(test_sequences, padding_side="bottom", padding_value=1)


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

    # Verify `stack_and_pad_to_max_dim()` returns exact same result for 1D sequences.
    assert np.all(
        result.numpy()
        == pad_to_max_dim_and_stack(
            test_sequences, padding_side="right", padding_value=padding_value
        ).numpy()
    )
    assert np.all(
        result.numpy()
        == pad_to_max_dim_and_stack(test_sequences, padding_value=padding_value).numpy()
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

    # Verify `stack_and_pad_to_max_dim()` returns exact same result for 1D sequences.
    assert np.all(
        result.numpy()
        == pad_to_max_dim_and_stack(
            test_sequences, padding_side="left", padding_value=padding_value
        ).numpy()
    )


def test_stack_and_pad_to_max_dim_invalid_side():
    test_sequences = [[1, 2, 3, 4], [5], [6, 7]]
    with pytest.raises(ValueError, match="Unsupported padding side: 'top'"):
        pad_to_max_dim_and_stack(test_sequences, padding_side="top", padding_value=1)


def test_stack_and_pad_to_max_dim_incompatible_dimensionality():
    test_sequences = [[1, 2, 3, 4], [5], [[6, 7], [8, 9]]]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Tensors have different number of dimensions: 1 vs 2! "
            "Shapes: torch.Size([4]), torch.Size([2, 2])"
        ),
    ):
        pad_to_max_dim_and_stack(test_sequences, padding_value=1)


@pytest.mark.parametrize(
    "padding_value",
    [0, -100, 7],
)
def test_stack_and_pad_to_max_dim_right_side(padding_value):
    test_sequences = [torch.ones([5, 1, 2]), torch.full([3, 2, 1], 2)]
    result = pad_to_max_dim_and_stack(test_sequences, padding_value=padding_value)
    assert result.shape == (2, 5, 2, 2)

    expected = torch.full((2, 5, 2, 2), padding_value)
    expected[0, :, :1, :] = 1
    expected[1, :3, :, :1] = 2
    assert np.all(result.numpy() == expected.numpy()), (
        f"result: {result} expected: {expected}"
    )

    assert np.all(
        expected.numpy()
        == pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=3,
            padding_side="right",
            padding_value=padding_value,
        ).numpy()
    )

    assert np.all(
        expected.numpy()
        == pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=-1,
            padding_value=padding_value,
        ).numpy()
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Too many dimensions with variable size. "
            "Got: 3 variable size dimensions. "
            "Maximum allowed: 1"
        ),
    ):
        pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=1,
            padding_side="right",
            padding_value=padding_value,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Too many dimensions with variable size. "
            "Got: 3 variable size dimensions. "
            "Maximum allowed: 2"
        ),
    ):
        pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=2,
            padding_value=padding_value,
        )


@pytest.mark.parametrize(
    "padding_value",
    [0, -100, 7],
)
def test_stack_and_pad_to_max_dim_left_side(padding_value):
    test_sequences = [
        torch.ones([5, 1, 2, 1]),
        torch.full([3, 2, 1, 1], 2),
        torch.full([1, 1, 1, 1], -3),
    ]
    result = pad_to_max_dim_and_stack(
        test_sequences,
        padding_side="left",
        padding_value=padding_value,
    )
    assert result.shape == (3, 5, 2, 2, 1)

    expected = torch.full((3, 5, 2, 2, 1), padding_value)
    expected[0, :, -1:, :, :] = 1
    expected[1, -3:, :, -1:, :] = 2
    expected[2, -1:, -1:, -1:, :] = -3
    assert np.all(result.numpy() == expected.numpy()), (
        f"result: {result} expected: {expected}"
    )

    assert np.all(
        expected.numpy()
        == pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=3,
            padding_side="left",
            padding_value=padding_value,
        ).numpy()
    )

    assert np.all(
        expected.numpy()
        == pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=-1,
            padding_side="left",
            padding_value=padding_value,
        ).numpy()
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Too many dimensions with variable size. "
            "Got: 3 variable size dimensions. "
            "Maximum allowed: 1"
        ),
    ):
        pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=1,
            padding_side="left",
            padding_value=padding_value,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Too many dimensions with variable size. "
            "Got: 3 variable size dimensions. "
            "Maximum allowed: 2"
        ),
    ):
        pad_to_max_dim_and_stack(
            test_sequences,
            max_variable_sized_dims=2,
            padding_side="left",
            padding_value=padding_value,
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


def test_get_dtype_size_in_bytes_str():
    assert get_dtype_size_in_bytes("f64") == 8
    assert get_dtype_size_in_bytes("float64") == 8
    assert get_dtype_size_in_bytes("double") == 8
    assert get_dtype_size_in_bytes("f32") == 4
    assert get_dtype_size_in_bytes("float32") == 4
    assert get_dtype_size_in_bytes("float") == 4
    assert get_dtype_size_in_bytes("bf16") == 2
    assert get_dtype_size_in_bytes("bfloat16") == 2
    assert get_dtype_size_in_bytes("f16") == 2
    assert get_dtype_size_in_bytes("float16") == 2
    assert get_dtype_size_in_bytes("half") == 2
    assert get_dtype_size_in_bytes("uint8") == 1


def test_get_dtype_size_in_bytes_torch():
    assert get_dtype_size_in_bytes(torch.float64) == 8
    assert get_dtype_size_in_bytes(torch.float32) == 4
    assert get_dtype_size_in_bytes(torch.float16) == 2
    assert get_dtype_size_in_bytes(torch.uint8) == 1

    assert get_dtype_size_in_bytes(torch.uint64) == 8
    assert get_dtype_size_in_bytes(torch.int64) == 8
    assert get_dtype_size_in_bytes(torch.uint32) == 4
    assert get_dtype_size_in_bytes(torch.int32) == 4
    assert get_dtype_size_in_bytes(torch.uint16) == 2
    assert get_dtype_size_in_bytes(torch.int16) == 2
    assert get_dtype_size_in_bytes(torch.uint8) == 1
    assert get_dtype_size_in_bytes(torch.int8) == 1


def test_get_dtype_size_in_bytes_numpy():
    assert get_dtype_size_in_bytes(np.float64) == 8
    assert get_dtype_size_in_bytes(torch.float32) == 4
    assert get_dtype_size_in_bytes(torch.float16) == 2
    assert get_dtype_size_in_bytes(torch.uint8) == 1

    assert get_dtype_size_in_bytes(torch.uint64) == 8
    assert get_dtype_size_in_bytes(torch.int64) == 8
    assert get_dtype_size_in_bytes(torch.uint32) == 4
    assert get_dtype_size_in_bytes(torch.int32) == 4
    assert get_dtype_size_in_bytes(torch.uint16) == 2
    assert get_dtype_size_in_bytes(torch.int16) == 2
    assert get_dtype_size_in_bytes(torch.uint8) == 1
    assert get_dtype_size_in_bytes(torch.int8) == 1


def test_estimate_sample_dict_size_in_bytes():
    assert estimate_sample_dict_size_in_bytes({"hi": [1] * 100}) == 402
    assert estimate_sample_dict_size_in_bytes({"hi": [1] * 40, "bye": [1] * 60}) == 405
    assert estimate_sample_dict_size_in_bytes({"hi": "Wir müssen"}) == 13
    assert estimate_sample_dict_size_in_bytes({"hi": "Мы должны"}) == 19
    assert estimate_sample_dict_size_in_bytes({"Мы": "должны"}) == 16
    assert estimate_sample_dict_size_in_bytes({"hi": "there"}) == 7

    assert estimate_sample_dict_size_in_bytes({"hi": ["there"]}) == 7

    # Numpy
    assert (
        estimate_sample_dict_size_in_bytes(
            {"foo": np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int64)}
        )
        == 3 + 6 * 8
    )
    assert (
        estimate_sample_dict_size_in_bytes(
            {"foo": np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int32)}
        )
        == 3 + 6 * 4
    )

    # Torch
    assert (
        estimate_sample_dict_size_in_bytes(
            {
                "foo": torch.from_numpy(
                    np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int64)
                )
            }
        )
        == 3 + 6 * 8
    )
    assert (
        estimate_sample_dict_size_in_bytes(
            {
                "foo": torch.from_numpy(
                    np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
                )
            }
        )
        == 3 + 6 * 4
    )

    # Mixed
    assert (
        estimate_sample_dict_size_in_bytes(
            {
                "torch": torch.from_numpy(
                    np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int64)
                ),
                "numpy": np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int32),
                "list": [[1, 2, 3], [1, 2, 3]],
            }
        )
        == 14 + 6 * 8 + 6 * 4 + 6 * 4
    )


class LittleNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.foo_flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential()
        self.linear_relu_stack.add_module("linearA", torch.nn.Linear(14, 16))
        self.linear_relu_stack.add_module("reluA", torch.nn.ReLU())
        self.linear_relu_stack.add_module("linearB", torch.nn.Linear(16, 12))
        self.linear_relu_stack.add_module("reluB", torch.nn.ReLU())
        self.linear_relu_stack.add_module("linearC", torch.nn.Linear(12, 10))
        nested_stack = torch.nn.Sequential()
        self.linear_relu_stack.add_module("nested", nested_stack)
        nested_stack.add_module("linearX", torch.nn.Linear(10, 8))
        nested_stack.add_module("linearY", torch.nn.Linear(8, 8))
        nested_stack.add_module("linearZ", torch.nn.Linear(8, 4))
        self.bar_softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.foo_flatten(x)
        logits = self.linear_relu_stack(x)
        return self.bar_softmax(logits)


def _create_test_neural_network() -> torch.nn.Module:
    return LittleNeuralNetwork().to("cpu")


def test_freeze_model_layers():
    assert freeze_model_layers(_create_test_neural_network(), []) == 0
    assert (
        freeze_model_layers(_create_test_neural_network(), ["linear_relu_stack"]) == 1
    )
    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            ["linear_relu_stack", "foo_flatten", "bar_softmax"],
        )
        == 3
    ), "Basic layers not handled correctly"

    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            [
                "bar_softmax",
                "foo_flatten",
                "linear_relu_stack.linearA",
                "linear_relu_stack.linearB",
            ],
        )
        == 4
    ), "Basic nested layers not handled correctly"

    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            [
                "bar_softmax",
                "bar_softmax",
                "foo_flatten",
                "linear_relu_stack.linearA",
                "linear_relu_stack.linearB",
                "linear_relu_stack.linearB",
            ],
        )
        == 4
    ), "Duplicates not handled correctly"

    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            [
                "bar_softmax",
                "foo_flatten",
                "linear_relu_stack",
                "linear_relu_stack.linearA",  # has no effect as parent is frozen.
                "linear_relu_stack.linearB",  # has no effect as parent is frozen.
            ],
        )
        == 3
    ), "Redundant children not handled correctly"

    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            [
                "bar_softmax",
                "foo_flatten_DOESNT_EXIST",
                "linear_relu_stack.linearA",
                "linear_relu_stack.linearDOESNT_EXIST",
            ],
        )
        == 2
    ), "Non-existent entries not handled correctly"

    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            [
                "bar_softmax",
                "foo_flatten",
                "linear_relu_stack.linearA",
                "linear_relu_stack.linearB",
                "linear_relu_stack.nested.linearX",
                "linear_relu_stack.nested.linearZ",
            ],
        )
        == 6
    ), "Multiple nested layers not handled correctly"

    assert (
        freeze_model_layers(
            _create_test_neural_network(),
            [
                "bar_softmax",
                "foo_flatten",
                "linear_relu_stack.linearA",
                "linear_relu_stack.linearB",
                "linear_relu_stack.nested.linearX",  # no effect as parent is frozen.
                "linear_relu_stack.nested.linearY",  # no effect as parent is frozen.
                "linear_relu_stack.nested.linearZ",  # no effect as parent is frozen.
                "linear_relu_stack.nested",
            ],
        )
        == 5
    ), "Redunant multiple nested layers not handled correctly"
