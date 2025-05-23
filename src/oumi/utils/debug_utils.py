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

from typing import Any

from oumi.utils.logging import logger


def log_example_for_debugging(
    raw_example: Any,
    formatted_example: str,
    tokenized_example: list[tuple[int, str]],
    model_input: dict[str, Any],
) -> None:
    """Logs an example of the data in each step for debugging purposes.

    Args:
        raw_example: The raw example from the dataset.
        formatted_example: The formatted example after processing.
        tokenized_example: The tokenized example after tokenization.
        model_input: The final model input after collating.
    """
    # Log to debug file
    logger.debug("Raw example: %s", raw_example)
    logger.debug("Formatted example: %s", formatted_example)
    logger.debug("Tokenized example: %s", tokenized_example)
    logger.debug("Model input: %s", model_input)
