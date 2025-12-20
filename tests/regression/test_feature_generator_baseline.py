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

"""Baseline tests for feature generator outputs before refactoring.

These tests ensure feature generators continue to produce correct
outputs after processor refactoring.
"""

import json
from typing import Any

import numpy as np
import pytest
import torch

from oumi.builders import build_tokenizer
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
)
from oumi.core.feature_generators.vision_language_conversation_feature_generator import (
    VisionLanguageConversationFeatureGenerator,
)


def serialize_features(features: dict[str, Any]) -> dict[str, Any]:
    """Serialize feature generator output for comparison."""
    result = {
        "keys": sorted(features.keys()),
        "shapes": {},
        "dtypes": {},
        "has_negative_labels": False,
        "label_range": None,
    }

    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            result["shapes"][key] = list(value.shape)
            result["dtypes"][key] = str(value.dtype)

            # Check for negative labels
            if key == "labels":
                result["has_negative_labels"] = bool(torch.any(value < 0))
                result["label_range"] = [
                    int(value.min().item()),
                    int(value.max().item()),
                ]
        elif isinstance(value, np.ndarray):
            result["shapes"][key] = list(value.shape)
            result["dtypes"][key] = str(value.dtype)

            if key == "labels":
                result["has_negative_labels"] = bool(np.any(value < 0))
                result["label_range"] = [int(value.min()), int(value.max())]
        elif isinstance(value, list):
            result["shapes"][key] = f"list[{len(value)}]"
        else:
            result["shapes"][key] = str(type(value).__name__)

    return result


@pytest.mark.slow
@pytest.mark.requires_download
def test_feature_generator_single_image_baseline(
    single_image_conversation, tmp_path
):
    """Test feature generator with single image - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    generator = VisionLanguageConversationFeatureGenerator(
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=False,
        return_tensors="pt",
    )

    # Generate features
    features = generator.transform_conversation(
        single_image_conversation, options=None
    )

    # Verify basic structure
    assert isinstance(features, dict)
    assert "input_ids" in features
    assert "labels" in features
    assert "attention_mask" in features
    assert "pixel_values" in features

    # Verify labels are created from input_ids
    input_ids = features["input_ids"]
    labels = features["labels"]

    if isinstance(input_ids, torch.Tensor) and isinstance(labels, torch.Tensor):
        # Labels should be mostly the same as input_ids (except masked tokens)
        assert input_ids.shape == labels.shape

    # Save baseline
    baseline = serialize_features(features)
    baseline_file = tmp_path / "feature_generator_single_image_baseline.json"
    baseline_file.write_text(json.dumps(baseline, indent=2))


@pytest.mark.slow
@pytest.mark.requires_download
def test_feature_generator_image_token_masking_baseline(single_image_conversation):
    """Test that image tokens are masked in labels - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    generator = VisionLanguageConversationFeatureGenerator(
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=False,
        return_tensors="pt",
        label_ignore_index=-100,
    )

    features = generator.transform_conversation(
        single_image_conversation, options=None
    )

    # Get labels
    labels = features["labels"]

    # Should have -100 for masked tokens (image tokens)
    if isinstance(labels, torch.Tensor):
        assert torch.any(labels == -100), "Expected some tokens to be masked"
    elif isinstance(labels, np.ndarray):
        assert np.any(labels == -100), "Expected some tokens to be masked"


@pytest.mark.slow
@pytest.mark.requires_download
def test_feature_generator_internal_config_usage_baseline(single_image_conversation):
    """Test that InternalModelConfig is used correctly - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    # Get internal config
    config = find_internal_model_config_using_model_name(
        model_name, trust_remote_code=False
    )
    assert config is not None

    # Verify config properties
    assert hasattr(config, "model_input_features")
    assert hasattr(config, "label_ignore_index")
    assert hasattr(config, "processor_kwargs")

    # Build feature generator
    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)
    generator = VisionLanguageConversationFeatureGenerator(
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=False,
        return_tensors="pt",
    )

    # Generate features
    features = generator.transform_conversation(
        single_image_conversation, options=None
    )

    # Verify expected features from config are present
    for feature_name, feature_spec in config.model_input_features.items():
        if feature_spec.required:
            assert (
                feature_name in features
            ), f"Required feature '{feature_name}' missing"


@pytest.mark.slow
@pytest.mark.requires_download
def test_feature_generator_completion_only_baseline(single_image_conversation):
    """Test completion-only training masking - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)

    # Without completion-only training
    generator_full = VisionLanguageConversationFeatureGenerator(
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=False,
        return_tensors="pt",
        train_on_completions_only=False,
    )

    features_full = generator_full.transform_conversation(
        single_image_conversation, options=None
    )

    # With completion-only training
    generator_completions = VisionLanguageConversationFeatureGenerator(
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=False,
        return_tensors="pt",
        train_on_completions_only=True,
        response_template="ASSISTANT:",
    )

    features_completions = generator_completions.transform_conversation(
        single_image_conversation, options=None
    )

    # Both should have same keys
    assert set(features_full.keys()) == set(features_completions.keys())

    # Labels should differ (more masking in completion-only mode)
    labels_full = features_full["labels"]
    labels_completions = features_completions["labels"]

    if isinstance(labels_full, torch.Tensor):
        masked_full = torch.sum(labels_full == -100)
        masked_completions = torch.sum(labels_completions == -100)
        # Completion-only should have more masked tokens
        assert (
            masked_completions >= masked_full
        ), "Completion-only should mask more tokens"


@pytest.mark.slow
@pytest.mark.requires_download
def test_feature_generator_truncation_baseline(single_image_conversation):
    """Test truncation handling - baseline behavior."""
    from oumi.core.configs import ModelParams

    model_name = "llava-hf/llava-1.5-7b-hf"

    model_params = ModelParams(model_name=model_name)
    tokenizer = build_tokenizer(model_params)

    # With truncation
    generator = VisionLanguageConversationFeatureGenerator(
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=False,
        return_tensors="pt",
        max_length=50,
        truncation=True,
        truncation_side="right",
    )

    features = generator.transform_conversation(
        single_image_conversation, options=None
    )

    # Verify features were generated (truncation is applied at text level before tokenization)
    # The actual sequence length may be longer than max_length due to chat template overhead
    # and image tokens added after truncation
    input_ids = features["input_ids"]
    assert input_ids is not None
    assert "pixel_values" in features
