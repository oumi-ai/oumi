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

from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


def load_dataset_from_config(config: AnalyzeConfig) -> BaseMapDataset:
    """Load dataset based on configuration.

    This function loads datasets directly from the registry for analysis purposes.
    If a tokenizer is provided in the config, it will be passed to the dataset
    constructor.
    """
    # Delayed import to avoid circular dependency with registry and dataset modules
    from oumi.core.registry import REGISTRY

    dataset_name = config.dataset_name
    split = config.split
    subset = config.subset
    tokenizer = config.tokenizer

    if not dataset_name:
        raise ValueError("Dataset name is required")

    try:
        # Load dataset from the REGISTRY
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=subset)

        if dataset_class is not None:
            # Prepare dataset constructor arguments
            dataset_kwargs = {
                "dataset_name": dataset_name,
                "dataset_path": None,
                "split": split,
                "subset": subset,
                "trust_remote_code": config.trust_remote_code,
            }

            # Add tokenizer if provided
            if tokenizer is not None:
                dataset_kwargs["tokenizer"] = tokenizer

            # Add processor parameters for vision-language datasets
            if config.processor_name:
                dataset_kwargs["processor_name"] = config.processor_name
                dataset_kwargs["processor_kwargs"] = config.processor_kwargs
                dataset_kwargs["trust_remote_code"] = config.trust_remote_code

            # Load registered dataset with parameters
            dataset = dataset_class(**dataset_kwargs)

            # Ensure we return a BaseMapDataset
            if isinstance(dataset, BaseMapDataset):
                return dataset
            else:
                raise NotImplementedError(
                    f"Dataset type {type(dataset)} is not supported for analysis. "
                    "Please use a dataset that inherits from BaseMapDataset."
                )
        else:
            # TODO: Implement HuggingFace Hub loading
            raise NotImplementedError(
                f"Dataset '{dataset_name}' is not registered in the REGISTRY. "
                "Loading from HuggingFace Hub is not yet implemented."
            )

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
