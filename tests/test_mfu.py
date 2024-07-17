from typing import Optional

import pytest
import torch

from lema.evaluation.mfu import calculate_mfu


@pytest.mark.parametrize(
    "device_name,num_devices,dtype,num_params,num_tokens,delta_time_seconds,expected_mfu,num_layers,num_attention_heads,attention_head_size,sequence_length,add_rematerialization",
    [
        (
            "NVIDIA A100-SXM4-80GB",
            1,
            torch.bfloat16,
            124e6,
            178000,
            1.0,
            0.424,
            None,
            None,
            None,
            None,
            False,
        ),  # nanogpt, model only
        (
            "NVIDIA A100-SXM4-80GB",
            1,
            torch.bfloat16,
            124e6,
            178000,
            1.0,
            0.489,
            12,
            12,
            64,
            1024,
            False,
        ),  # nanogpt, model + attention
        (
            "NVIDIA A100-SXM4-80GB",
            2240,
            torch.bfloat16,
            530e9,
            65400,
            1.0,
            0.298,
            None,
            None,
            None,
            None,
            False,
        ),  # MT-NLG 530B, model only
        (
            "NVIDIA A100-SXM4-80GB",
            2240,
            torch.bfloat16,
            530e9,
            65400,
            1.0,
            0.306,
            105,
            128,
            256,
            2048,
            False,
        ),  # MT-NLG 530B, model + attention
        (
            "TPUv4",
            6144,
            torch.bfloat16,
            540e9,
            238300,
            1.0,
            0.457,
            None,
            48,
            256,
            2048,
            False,
        ),  # PaLM 540B, model only
        (
            "TPUv4",
            6144,
            torch.bfloat16,
            540e9,
            238300,
            1.0,
            0.462,
            118,
            48,
            256,
            2048,
            False,
        ),  # PaLM 540B, model + attention
        (
            "TPUv4",
            6144,
            torch.bfloat16,
            540e9,
            238300,
            1.0,
            0.578,
            118,
            48,
            256,
            2048,
            True,
        ),  # PaLM 540B, model + attention + rematerialization
    ],
)
def test_mfu_parametric(
    device_name: str,
    num_devices: int,
    dtype: torch.dtype,
    num_params: int,
    num_tokens: int,
    delta_time_seconds: float,
    expected_mfu: float,
    num_layers: Optional[int],
    num_attention_heads: Optional[int],
    attention_head_size: Optional[int],
    sequence_length: Optional[int],
    add_rematerialization: bool,
):
    mfu = calculate_mfu(
        device_name=device_name,
        num_devices=num_devices,
        dtype=dtype,
        num_params=num_params,
        num_tokens=num_tokens,
        delta_time_seconds=delta_time_seconds,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        attention_head_size=attention_head_size,
        sequence_length=sequence_length,
        add_rematerialization=add_rematerialization,
    )

    assert abs(mfu - expected_mfu) < 2e-3


def test_mfu_bad_device():
    with pytest.raises(NotImplementedError):
        calculate_mfu(
            device_name="BadDevice",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )


def test_mfu_bad_dtype():
    with pytest.raises(NotImplementedError):
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.int8,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )


def test_mfu_bad_num_devices():
    with pytest.raises(ValueError):
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=0,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )


def test_mfu_bad_num_tokens():
    with pytest.raises(ValueError):
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=0,
            delta_time_seconds=1.0,
        )


def test_mfu_bad_delta_time_seconds():
    with pytest.raises(ValueError):
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=124,
            num_tokens=178000,
            delta_time_seconds=0,
        )


def test_mfu_bad_num_params():
    with pytest.raises(ValueError):
        calculate_mfu(
            device_name="NVIDIA A100-SXM4-80GB",
            num_devices=1,
            dtype=torch.bfloat16,
            num_params=0,
            num_tokens=178000,
            delta_time_seconds=1.0,
        )
