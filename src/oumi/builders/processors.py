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

import functools
from typing import Any, Optional

import transformers

import oumi.core.constants as constants
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.processors.default_processor import DefaultProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


def build_processor(
    processor_name: str,
    tokenizer: BaseTokenizer,
    *,
    processor_kwargs: Optional[dict[str, Any]] = None,
    trust_remote_code: bool = False,
) -> BaseProcessor:
    """Builds a processor.

    Args:
        processor_name: A name of the processor (usually, equals to a model name).
        tokenizer: A tokenizer to use with the processor.
        processor_kwargs: A dictionary of processor-specific parameters.
            These parameters are passed to the processor constructor.
            They can override model-specific parameters.
        trust_remote_code: Whether to allow loading remote code for this processor
            Some processors come with downloadable executable Python files,
            which can be a potential security risk, unless it's from a trusted source.

    Returns:
        BaseProcessor: The newly created processor.
    """
    if not processor_name:
        raise ValueError("Empty model name.")

    model_config = find_internal_model_config_using_model_name(
        processor_name, trust_remote_code=trust_remote_code
    )

    # Initialize model-specific params.
    label_ignore_index: Optional[int] = constants.LABEL_IGNORE_INDEX
    ignore_features: Optional[list[str]] = None
    effective_processor_kwargs = {}
    if model_config is not None:
        label_ignore_index = model_config.label_ignore_index
        ignore_features = model_config.ignore_features
        effective_processor_kwargs.update(model_config.processor_kwargs)

    if processor_kwargs is not None and len(processor_kwargs) > 0:
        # Override model-specific params with user-defined ones.
        effective_processor_kwargs.update(processor_kwargs)

    create_processor_fn = functools.partial(
        transformers.AutoProcessor.from_pretrained,
        processor_name,
        trust_remote_code=trust_remote_code,
    )
    if len(effective_processor_kwargs) > 0:
        worker_processor = create_processor_fn(**effective_processor_kwargs)
    else:
        worker_processor = create_processor_fn()

    return DefaultProcessor(
        processor_name,
        worker_processor,
        tokenizer,
        label_ignore_index=label_ignore_index,
        ignore_features=ignore_features,
    )
