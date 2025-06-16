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

"""Registry for built-in SimpleJudge configurations."""

from typing import Optional

from oumi.core.configs.judge_config_v2 import JudgeConfig
from oumi.judges_v2.builtin_simple_judges.qa.relevance import RELEVANCE_JUDGE_CONFIG

# Central registry mapping judge names to their configs
JUDGES_DICT: dict[str, JudgeConfig] = {
    "qa/relevance": RELEVANCE_JUDGE_CONFIG,
}


class BuiltinJudgeRegistry:
    """Registry for built-in judge configurations."""

    @classmethod
    def get_config(cls, judge_name: str) -> Optional[JudgeConfig]:
        """Get a built-in judge configuration by name.

        Args:
            judge_name: Name of the judge (e.g., "qa/relevance")

        Returns:
            JudgeConfig if found, None otherwise
        """
        return JUDGES_DICT.get(judge_name)

    @classmethod
    def list_available_judges(cls) -> dict[str, list[str]]:
        """List all available built-in judge names grouped by category.

        Returns:
            Dict mapping category names to lists of judge names.
            Categories are folder prefixes (e.g., "qa") plus "shortcuts" for aliases.
        """
        grouped = {}

        for judge_name in JUDGES_DICT.keys():
            if "/" in judge_name:
                category, name = judge_name.split("/", 1)
                if category not in grouped:
                    grouped[category] = []
                grouped[category].append(name)
            else:
                if "ungrouped" not in grouped:
                    grouped["ungrouped"] = []
                grouped["ungrouped"].append(judge_name)

        return grouped
