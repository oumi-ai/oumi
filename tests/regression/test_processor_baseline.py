# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Baseline tests for processor outputs before refactoring.

These tests capture the current behavior of processors to ensure
the refactoring doesn't change outputs or break functionality.
"""

import json
from typing import Any

import pytest
import torch
import transformers

from oumi.builders import build_processor, build_tokenizer


def serialize_processor_output(output: transformers.BatchFeature) -> dict[str, Any]:
    """Serialize processor output for comparison.

    We don't compare exact tensor values (floating point issues),
    but rather the structure, shapes, and key properties.
    """
    result = {
        "keys": sorted(output.keys()),
        "shapes": {},
        "dtypes": {},
    }

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            result["shapes"][key] = list(value.shape)
            result["dtypes"][key] = str(value.dtype)
        elif isinstance(value, list):
            result["shapes"][key] = f"list[{len(value)}]"
        else:
            result["shapes"][key] = str(type(value).__name__)

    return result


@pytest.mark.slow
@pytest.mark.requires_download
def test_processor_single_image_baseline(test_image, tmp_path):
    """Test processor with single image - baseline behavior."""
    from oumi.core.configs import ModelParams

    # Use a small, fast model for baseline
    model_name = "llava-hf/llava-1.5-7b-hf"

    # Build tokenizer and processor
    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    processor = build_processor(
        model_name,
        tokenizer,
        trust_remote_code=False,
    )

    # Test basic processor properties
    assert processor is not None
    assert callable(processor)
    assert processor.processor_name == model_name
    assert processor.tokenizer is not None
    assert processor.image_processor is not None

    # Process single image
    result = processor(
        images=[test_image],
        text=["What is in this image?"],
        return_tensors="pt",
        padding=True,
    )

    # Verify result structure
    assert isinstance(result, transformers.BatchEncoding)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "pixel_values" in result

    # Save baseline structure
    baseline = serialize_processor_output(result)
    baseline_file = tmp_path / "processor_single_image_baseline.json"
    baseline_file.write_text(json.dumps(baseline, indent=2))

    # Verify expected shapes (for llava-1.5-7b)
    assert result["input_ids"].dim() == 2  # [batch_size, seq_len]
    assert result["pixel_values"].dim() == 4  # [batch_size, channels, height, width]
    assert result["input_ids"].shape[0] == 1  # batch size 1


@pytest.mark.slow
@pytest.mark.requires_download
def test_processor_chat_template_baseline(test_image):
    """Test processor apply_chat_template - baseline behavior."""
    from oumi.core.configs import ModelParams
    from oumi.core.types.conversation import ContentItem, Message, Role, Type

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    processor = build_processor(
        model_name,
        tokenizer,
        trust_remote_code=False,
    )

    # Create message with image
    messages = [
        Message(
            role=Role.USER,
            content=[
                ContentItem(type=Type.TEXT, content="What is this?"),
            ],
        )
    ]

    # Apply chat template
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=False
    )

    # Verify output is string
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "What is this?" in prompt


@pytest.mark.slow
@pytest.mark.requires_download
def test_processor_properties_baseline():
    """Test processor properties - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    processor = build_processor(
        model_name,
        tokenizer,
        trust_remote_code=False,
    )

    # Test key properties exist
    assert hasattr(processor, "processor_name")
    assert hasattr(processor, "tokenizer")
    assert hasattr(processor, "image_processor")
    assert hasattr(processor, "chat_template")
    assert hasattr(processor, "label_ignore_index")

    # Test property values
    assert processor.processor_name == model_name
    assert processor.tokenizer is not None
    assert processor.image_processor is not None
    assert processor.chat_template is not None
    assert processor.label_ignore_index is not None

    # Test image token properties
    assert hasattr(processor, "image_token")
    assert hasattr(processor, "image_token_id")


@pytest.mark.slow
@pytest.mark.requires_download
def test_processor_kwargs_applied_baseline():
    """Test that processor_kwargs from InternalModelConfig are applied."""
    from oumi.core.configs import ModelParams
    from oumi.core.configs.internal.supported_models import (
        find_internal_model_config_using_model_name,
    )

    model_name = "llava-hf/llava-1.5-7b-hf"

    # Get internal config
    config = find_internal_model_config_using_model_name(
        model_name, trust_remote_code=False
    )

    # Verify config has processor_kwargs
    assert config is not None
    assert hasattr(config, "processor_kwargs")
    assert len(config.processor_kwargs) > 0

    # Build processor
    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    processor = build_processor(
        model_name,
        tokenizer,
        trust_remote_code=False,
    )

    # Processor should be built successfully with kwargs applied
    assert processor is not None


@pytest.mark.slow
@pytest.mark.requires_download
def test_processor_batch_processing_baseline(test_image):
    """Test processor with batched inputs - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    processor = build_processor(
        model_name,
        tokenizer,
        trust_remote_code=False,
    )

    # Process batch of images
    result = processor(
        images=[test_image, test_image],
        text=["Image 1", "Image 2"],
        return_tensors="pt",
        padding=True,
    )

    # Verify batch dimension
    assert result["input_ids"].shape[0] == 2
    assert result["pixel_values"].shape[0] == 2


def test_processor_build_empty_name():
    """Test that building processor with empty name raises ValueError."""
    from oumi.builders import build_tokenizer
    from oumi.core.configs import ModelParams

    # Create tokenizer
    model_params = ModelParams(model_name="openai-community/gpt2")
    tokenizer = build_tokenizer(model_params)

    with pytest.raises(ValueError, match="Empty model name"):
        build_processor("", tokenizer, trust_remote_code=False)
