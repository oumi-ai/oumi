import math
from typing import NamedTuple, Optional

import pytest
import torch

from oumi.performance.mfu import (
    calculate_mfu,
    calculate_mfu_from_model_flops_per_second,
)

_MFU_ABS_TOLERANCE = 1e-3


class MfuTestParams(NamedTuple):
    device_name: str
    num_devices: int
    dtype: torch.dtype
    num_params: int
    num_tokens: int
    delta_time_seconds: float
    expected_mfu: float
    num_layers: Optional[int]
    num_attention_heads: Optional[int]
    attention_head_size: Optional[int]
    sequence_length: Optional[int]
    add_rematerialization: bool


@pytest.mark.parametrize(
    "params",
    [
        MfuTestParams(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=int(124e6),
            num_tokens=178000,
            delta_time_seconds=1.0,
            expected_mfu=0.424,
            num_layers=None,
            num_attention_heads=None,
            attention_head_size=None,
            sequence_length=None,
            add_rematerialization=False,
        ),  # nanogpt, model only
        MfuTestParams(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=int(124e6),
            num_tokens=178000,
            delta_time_seconds=1.0,
            expected_mfu=0.489,
            num_layers=12,
            num_attention_heads=12,
            attention_head_size=64,
            sequence_length=1024,
            add_rematerialization=False,
        ),  # nanogpt, model + attention
        MfuTestParams(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=2240,
            dtype=torch.bfloat16,
            num_params=int(530e9),
            num_tokens=65400,
            delta_time_seconds=1.0,
            expected_mfu=0.298,
            num_layers=None,
            num_attention_heads=None,
            attention_head_size=None,
            sequence_length=None,
            add_rematerialization=False,
        ),  # MT-NLG 530B, model only
        MfuTestParams(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=2240,
            dtype=torch.bfloat16,
            num_params=int(530e9),
            num_tokens=65400,
            delta_time_seconds=1.0,
            expected_mfu=0.306,
            num_layers=105,
            num_attention_heads=128,
            attention_head_size=256,
            sequence_length=2048,
            add_rematerialization=False,
        ),  # MT-NLG 530B, model + attention
        MfuTestParams(
            device_name="TPUv4",
            num_devices=6144,
            dtype=torch.bfloat16,
            num_params=int(540e9),
            num_tokens=238300,
            delta_time_seconds=1.0,
            expected_mfu=0.457,
            num_layers=None,
            num_attention_heads=48,
            attention_head_size=256,
            sequence_length=2048,
            add_rematerialization=False,
        ),  # PaLM 540B, model only
        MfuTestParams(
            device_name="TPUv4",
            num_devices=6144,
            dtype=torch.bfloat16,
            num_params=int(540e9),
            num_tokens=238300,
            delta_time_seconds=1.0,
            expected_mfu=0.462,
            num_layers=118,
            num_attention_heads=48,
            attention_head_size=256,
            sequence_length=2048,
            add_rematerialization=False,
        ),  # PaLM 540B, model + attention
        MfuTestParams(
            device_name="TPUv4",
            num_devices=6144,
            dtype=torch.bfloat16,
            num_params=int(540e9),
            num_tokens=238300,
            delta_time_seconds=1.0,
            expected_mfu=0.578,
            num_layers=118,
            num_attention_heads=48,
            attention_head_size=256,
            sequence_length=2048,
            add_rematerialization=True,
        ),  # PaLM 540B, model + attention + rematerialization
    ],
)
def test_mfu_parametric(params: MfuTestParams):
    mfu = calculate_mfu(
        device_name=params.device_name,
        num_devices=params.num_devices,
        dtype=params.dtype,
        num_params=params.num_params,
        num_tokens=params.num_tokens,
        delta_time_seconds=params.delta_time_seconds,
        num_layers=params.num_layers,
        num_attention_heads=params.num_attention_heads,
        attention_head_size=params.attention_head_size,
        sequence_length=params.sequence_length,
        add_rematerialization=params.add_rematerialization,
    )

    assert math.isclose(mfu, params.expected_mfu, abs_tol=_MFU_ABS_TOLERANCE)


class MfuFromModelFlopsTestParams(NamedTuple):
    device_name: str
    num_devices: int
    dtype: torch.dtype
    model_tflops_per_second_on_all_devices: float
    expected_mfu: float


@pytest.mark.parametrize(
    "params",
    [
        MfuFromModelFlopsTestParams(
            device_name="NVIDIA A100-SXM4-40GB",
            num_devices=1,
            dtype=torch.bfloat16,
            model_tflops_per_second_on_all_devices=156,
            expected_mfu=0.5,
        ),
        MfuFromModelFlopsTestParams(
            device_name="NVIDIA A100-SXM4-40GB",
            num_devices=2,
            dtype=torch.float16,
            model_tflops_per_second_on_all_devices=156,
            expected_mfu=0.25,
        ),
        MfuFromModelFlopsTestParams(
            device_name="NVIDIA GeForce RTX 3090",
            num_devices=1,
            dtype=torch.float32,
            model_tflops_per_second_on_all_devices=35.6,
            expected_mfu=1.0,
        ),
        MfuFromModelFlopsTestParams(
            device_name="NVIDIA GeForce RTX 3090",
            num_devices=4,
            dtype=torch.float32,
            model_tflops_per_second_on_all_devices=35.6,
            expected_mfu=0.25,
        ),
    ],
)
def test_calculate_mfu_from_model_flops_per_second_parametric(
    params: MfuFromModelFlopsTestParams,
):
    mfu = calculate_mfu_from_model_flops_per_second(
        device_name=params.device_name,
        num_devices=params.num_devices,
        dtype=params.dtype,
        model_flops_per_second_on_all_devices=(
            params.model_tflops_per_second_on_all_devices * 1e12
        ),
    )
    assert math.isclose(mfu, params.expected_mfu, abs_tol=_MFU_ABS_TOLERANCE)


def test_mfu_bad_device():
    with pytest.raises(NotImplementedError) as exception_info:
        calculate_mfu(
            device_name="BadDevice",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )
    assert "BadDevice" in str(exception_info.value)


def test_mfu_bad_dtype():
    with pytest.raises(NotImplementedError) as exception_info:
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.int8,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )
    assert "torch.int8" in str(exception_info.value)


def test_mfu_bad_num_devices():
    with pytest.raises(ValueError) as exception_info:
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=0,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )
    assert "Must have a positive number of devices" in str(exception_info.value)


def test_mfu_bad_num_tokens():
    with pytest.raises(ValueError) as exception_info:
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=0,
            delta_time_seconds=1.0,
        )
    assert "Must have a positive number of tokens" in str(exception_info.value)


def test_mfu_bad_delta_time_seconds():
    with pytest.raises(ValueError) as exception_info:
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=0,
        )
    assert "Must have a positive delta time" in str(exception_info.value)


def test_mfu_bad_num_params():
    with pytest.raises(ValueError) as exception_info:
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=0,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )
    assert "Must have a positive number of model params" in str(exception_info.value)


def test_mfu_from_model_flops_per_second_bad_num_devices():
    with pytest.raises(ValueError) as exception_info:
        calculate_mfu_from_model_flops_per_second(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=0,
            dtype=torch.bfloat16,
            model_flops_per_second_on_all_devices=1e9,
        )
    assert "Must have a positive number of devices" in str(exception_info.value)
