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

from typing_extensions import override

from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.registry import REGISTRY, RegistryType
from oumi.judges.base_judge import BaseJudge, JudgeOutput, JudgeOutputField

# Keys for output fields
JUDGMENT_KEY = "judgment"


class RuleBasedJudge(BaseJudge):
    """A Rule Based Judge for evaluating outputs based on a configuration."""

    def __init__(self, judge_config: JudgeConfig | str):
        """Initialize the RuleBasedJudge.

        Args:
            judge_config: JudgeConfig object or path to a judge configuration file.
                Must contain rule_judge_params for rule-based evaluation.

        Raises:
            ValueError: If rule_judge_params is not provided in the config.
        """
        if isinstance(judge_config, str):
            judge_config = JudgeConfig.from_path(judge_config)

        if judge_config.rule_judge_params is None:
            raise ValueError(
                "rule_judge_params must be provided for RuleBasedJudge. "
                "Please add rule_judge_params to your JudgeConfig."
            )

        self._judge_params = judge_config.judge_params
        self._rule_params = judge_config.rule_judge_params

        output_fields = self._create_output_fields()

        super().__init__(
            prompt_template=self._judge_params.prompt_template,
            prompt_template_placeholders=set(self._rule_params.input_fields),
            system_instruction=None,
            example_field_values=[],
            response_format=self._rule_params.response_format,
            output_fields=output_fields,
            inference_engine=None,
        )

    def _create_output_fields(self) -> list[JudgeOutputField]:
        """Create output fields based on rule parameters."""
        return [
            JudgeOutputField(
                field_key=JUDGMENT_KEY,
                field_type=self._rule_params.judgment_type,
                field_scores=self._rule_params.judgment_scores,
            )
        ]

    @override
    def judge(self, inputs: list[dict[str, str]]) -> list[JudgeOutput]:
        self.validate_dataset(inputs)
        results = []
        for input_data in inputs:
            judgment, score = self._apply_rule(input_data)

            results.append(
                JudgeOutput(
                    raw_output=f"{JUDGMENT_KEY}: {judgment}",
                    field_values={JUDGMENT_KEY: judgment},
                    field_scores={JUDGMENT_KEY: score},
                    response_format=self._rule_params.response_format,
                    output_fields=self.output_fields,
                )
            )

        return results

    def _apply_rule(self, input_data: dict[str, str]) -> tuple[bool, float]:
        """Pulls specified rule from registry and applies it on the input data."""
        rule_type = self._rule_params.rule_type
        rule_class = REGISTRY.get(rule_type, RegistryType.RULE)

        if rule_class is None:
            raise ValueError(f"Rule type {rule_type} not found in registry")

        rule = rule_class()
        return rule.apply(input_data, self._rule_params.rule_config)
