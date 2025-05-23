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


def try_prepare_grpo_example(
    example: dict,
) -> dict:
    """Prepares an example for GRPO_TRL processing.

    This function checks if the input example is one of known special cases
    e.g., SFT example, and transforms it into a GRPO compatible format.
    Otherwise, it returns the original example.

    Args:
        example (dict): The input example.

    Returns:
        GRPO compatible example, or an original example.
    """
    if not isinstance(example, dict):
        return example

    if "conversation_json" in example:
        # This is a special case for the GRPO_TRL dataset.
        # The example is already in the correct format.
        return example

    return example
