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
import os
from pathlib import Path
from typing import Optional, Union

import transformers

from oumi.core.configs.internal.supported_models import (
    is_custom_model,
)
from oumi.utils.logging import logger


def is_cached_to_disk_hf_dataset(dataset_folder: Union[str, Path]) -> bool:
    """Detects whether a dataset was saved using `dataset.save_to_disk()`.

    Such datasets should be loaded using `datasets.Dataset.load_from_disk()`

    Returns:
        Whether the dataset was saved using `dataset.save_to_disk()` method.
    """
    if not dataset_folder:
        return False

    dataset_path: Path = Path(dataset_folder)

    if dataset_path.exists() and dataset_path.is_dir():
        for file_name in ("dataset_info.json", "state.json"):
            file_path: Path = dataset_path / file_name
            if not (file_path.exists() and file_path.is_file()):
                logger.warning(
                    f"The dataset {str(dataset_path)} is missing "
                    f"a required file: {file_name}."
                )
                return False
        return True

    return False


def find_hf_token() -> Optional[str]:
    """Attempts to find HuggingFace access token.

    Returns:
        A valid HF access token, or `None` if not found.
    """
    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        _DEFAULT_HF_HOME_PATH = "~/.cache/huggingface"

        file_must_exist = False
        token_path = os.environ.get("HF_TOKEN_PATH", None)
        if token_path:
            token_path = Path(token_path)
            file_must_exist = True
        else:
            hf_home_dir = Path(
                os.environ.get("HF_HOME", _DEFAULT_HF_HOME_PATH)
                or _DEFAULT_HF_HOME_PATH
            )
            token_path = hf_home_dir / "token"

        if token_path.exists():
            if token_path.is_file():
                hf_token = token_path.read_text().strip()
        elif file_must_exist:
            raise FileNotFoundError(f"Missing HF token file: '{token_path}'")

    return hf_token if hf_token else None


@functools.cache
def get_hf_chat_template(
    tokenizer_name: str, *, trust_remote_code: bool = False
) -> Optional[str]:
    """Returns chat template provided by HF for `tokenizer_name`."""
    if not tokenizer_name or is_custom_model(tokenizer_name):
        return None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.chat_template:
        if not isinstance(tokenizer.chat_template, str):
            raise RuntimeError(
                f"Chat template for tokenizer_name: {tokenizer_name} "
                "is not a string! "
                f"Actual type: {type(tokenizer.chat_template)}"
            )
        return tokenizer.chat_template
    return None
