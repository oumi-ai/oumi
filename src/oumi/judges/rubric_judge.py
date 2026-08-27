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

import logging

from typing_extensions import override

from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.configs.params.rubric_judge_params import (
    JudgeAggregation,
    JudgeCriterion,
    RubricJudgeParams,
)
from oumi.judges.base_judge import (
    BaseJudge,
    JudgeOutput,
    JudgeOutputField,
)
from oumi.judges.judge_utils import (
    build_judgment_field_schema,
    describe_judgment_options,
)

logger = logging.getLogger(__name__)

# Prompt scaffolding: introducing the rubric to the judge.
RUBRIC_HEADER = (
    "\n\nEvaluate the input against each of the following criteria, independently:\n"
)

# Prompt suffix: describing to the judge how to format its response (XML or JSON).
XML_SUFFIX = (
    "\n\nProvide your response in XML format only. Include exactly the following "
    "tags, in this order:\n{tag_list}\nDo not include any text outside the XML. "
    "Ensure that all tags are properly closed and that the XML is well-formed."
)
JSON_SUFFIX = (
    "\n\nProvide your response in JSON format only, as a flat object with exactly "
    "the following keys, in this order: {key_list}. Do not nest objects and do not "
    "include any text outside the JSON. Ensure the JSON is properly formatted and "
    "valid."
)


class RubricJudge(BaseJudge):
    """Judge that scores several criteria of the same input in one inference call."""

    def __init__(
        self,
        judge_config: JudgeConfig | str,
    ):
        """Initialize the RubricJudge.

        Args:
            judge_config: JudgeConfig object or a path to a judge configuration file.
                Must contain rubric_judge_params, together with judge_params (for the
                shared prompt scaffolding) and inference_config.

        Raises:
            ValueError: If rubric_judge_params or inference_config are missing, if the
                response format is RAW, or if judge_params sets single-judgment fields
                that a rubric judge does not use.
        """
        if isinstance(judge_config, str):
            judge_config = JudgeConfig.from_path(judge_config)

        if judge_config.rubric_judge_params is None:
            raise ValueError(
                "rubric_judge_params must be provided for RubricJudge. "
                "Please add rubric_judge_params to your JudgeConfig."
            )

        self._judge_params = judge_config.judge_params
        self._judge_params.replace_template_variables()
        self._rubric_params: RubricJudgeParams = judge_config.rubric_judge_params
        self._inference_config = judge_config.inference_config

        self._validate_judge_params(self._judge_params)

        # RAW cannot delimit multiple fields, so it cannot be parsed back into
        # per-criterion judgments.
        if self._judge_params.response_format == JudgeResponseFormat.RAW:
            raise ValueError(
                "RubricJudge does not support the RAW response format, which cannot "
                "delimit one judgment per criterion. Please use XML or JSON."
            )

        if self._inference_config is None:
            raise ValueError(
                "inference_config must be provided in JudgeConfig for RubricJudge. "
                "Please ensure your JudgeConfig includes a valid inference_config."
            )
        json_format = self._judge_params.response_format == JudgeResponseFormat.JSON
        inference_engine = self._create_inference_engine(
            inference_config=self._inference_config,
            response_schema=self._build_response_schema() if json_format else None,
        )

        output_fields = self._create_output_fields()
        self._rubric_suffix = self._build_rubric_suffix(output_fields)
        self._warn_about_aggregation_gaps()

        # Append the rubric and format suffix to the system instruction if it exists
        system_instruction = self._judge_params.system_instruction
        if system_instruction:
            system_instruction = f"{system_instruction}{self._rubric_suffix}"

        # Get set of prompt template placeholders
        prompt_template_placeholders_set = (
            set(self._judge_params.prompt_template_placeholders)
            if self._judge_params.prompt_template_placeholders
            else self._judge_params.get_placeholders()
        )

        super().__init__(
            prompt_template=self._judge_params.prompt_template,
            prompt_template_placeholders=prompt_template_placeholders_set,
            system_instruction=system_instruction,
            example_field_values=self._judge_params.examples,
            response_format=self._judge_params.response_format,
            output_fields=output_fields,
            inference_engine=inference_engine,
        )

    @property
    def criteria(self) -> list[JudgeCriterion]:
        """The criteria this judge evaluates, in prompt order."""
        return self._rubric_params.criteria

    @override
    def _build_judgment_prompt(self, judge_input: dict[str, str]) -> str:
        """Generate judge prompts using the template."""
        prompt_content = super()._build_judgment_prompt(judge_input)

        # Only append the rubric suffix to the judgment prompt if no system
        # instruction exists (otherwise it was already appended in __init__)
        if not self._judge_params.system_instruction:
            prompt_content += self._rubric_suffix

        return prompt_content

    @override
    def _transform_judge_output(self, raw_output: str) -> JudgeOutput:
        """Parse raw model output into a structured rubric judge output.

        Args:
            raw_output: The raw string output from the judge model.

        Returns:
            Structured rubric output with per-criterion values and an aggregate score.
        """
        judge_output = JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=self.response_format,
            output_fields=self.output_fields,
        )

        if not any(
            criterion.id in judge_output.parsed_output for criterion in self.criteria
        ):
            logger.warning(
                "No criteria could be parsed from the judge's response; reporting "
                f"every criterion as None. Expected "
                f"{self.response_format.value.upper()} output containing "
                f"{sorted(c.id for c in self.criteria)}. This usually means the "
                "response was truncated (consider raising `max_new_tokens`) or was "
                f"not well-formed. Raw output: {raw_output[:500]!r}"
            )

        judge_output.aggregate_score = self._aggregate_scores(
            field_values=judge_output.field_values,
            field_scores=judge_output.field_scores,
        )
        return judge_output

    def _aggregate_scores(
        self,
        field_values: dict[str, float | int | str | bool | None],
        field_scores: dict[str, float | None],
    ) -> float | None:
        """Combine the per-criterion scores into a single overall score.

        A criterion is excluded from the aggregation unless it produced both a usable
        value and a numeric score. The value check matters for BOOL criteria: the base
        parser scores an unparseable boolean as 0.0, which would otherwise let a
        garbled response masquerade as a genuine failing judgment.

        Args:
            field_values: Typed value per output field key.
            field_scores: Numeric score per output field key.

        Returns:
            The aggregate score, or None if aggregation is disabled or no criterion
            contributed a score.
        """
        aggregation = self._rubric_params.aggregation
        if aggregation == JudgeAggregation.NONE:
            return None

        scored = [
            (criterion, score)
            for criterion in self.criteria
            if field_values.get(criterion.id) is not None
            and (score := field_scores.get(criterion.id)) is not None
        ]
        if not scored:
            return None

        if aggregation == JudgeAggregation.WEIGHTED_MEAN:
            total_weight = sum(criterion.weight for criterion, _ in scored)
            if total_weight <= 0:
                return None
            weighted_sum = sum(criterion.weight * score for criterion, score in scored)
            return weighted_sum / total_weight
        elif aggregation == JudgeAggregation.MIN:
            return min(score for _, score in scored)
        elif aggregation == JudgeAggregation.ALL:
            return 1.0 if all(score == 1.0 for _, score in scored) else 0.0

        raise ValueError(f"Unsupported aggregation: {aggregation}")

    def _create_output_fields(self) -> list[JudgeOutputField]:
        """Create the output fields, one (or two) per criterion, in judging order.

        A criterion's explanation field precedes its judgment field, so that the
        judge reasons before committing to a judgment.
        """
        output_fields: list[JudgeOutputField] = []
        for criterion in self.criteria:
            if criterion.include_explanation:
                output_fields.append(
                    JudgeOutputField(
                        field_key=criterion.explanation_id,
                        field_type=JudgeOutputType.TEXT,
                        field_scores=None,
                    )
                )
            output_fields.append(
                JudgeOutputField(
                    field_key=criterion.id,
                    field_type=criterion.judgment_type,
                    field_scores=criterion.judgment_scores,
                )
            )
        return output_fields

    def _build_rubric_suffix(self, output_fields: list[JudgeOutputField]) -> str:
        """Build the rubric block and the response format instructions."""
        rubric_block = self._build_rubric_block()
        format_suffix = self._build_format_suffix(output_fields)
        return f"{rubric_block}{format_suffix}"

    def _build_rubric_block(self) -> str:
        """Enumerate the criteria, their descriptions, and their allowed values."""
        lines = [RUBRIC_HEADER]
        for index, criterion in enumerate(self.criteria, start=1):
            options = describe_judgment_options(
                judgment_type=criterion.judgment_type,
                judgment_scores=criterion.judgment_scores,
            ).strip()

            line = f"{index}. {criterion.id}: {criterion.description.strip()}"
            if options:
                line += f"\n   {options}"
            if criterion.include_explanation:
                explanation_id = criterion.explanation_id
                line += f"\n   Justify this judgment first, in '{explanation_id}'."
            lines.append(line)
        return "\n".join(lines)

    def _build_format_suffix(self, output_fields: list[JudgeOutputField]) -> str:
        """Describe the expected response format, listing every field in order."""
        field_keys = [output_field.field_key for output_field in output_fields]
        if self._judge_params.response_format == JudgeResponseFormat.XML:
            tag_list = "\n".join(f"<{key}></{key}>" for key in field_keys)
            return XML_SUFFIX.format(tag_list=tag_list)
        else:  # JudgeResponseFormat.JSON (RAW is rejected in __init__)
            key_list = ", ".join(f'"{key}"' for key in field_keys)
            return JSON_SUFFIX.format(key_list=key_list)

    def _build_response_schema(self) -> dict:
        """JSON schema describing the expected judge response."""
        properties: dict[str, dict] = {}

        for criterion in self.criteria:
            if criterion.include_explanation:
                properties[criterion.explanation_id] = {"type": "string"}
            properties[criterion.id] = build_judgment_field_schema(
                judgment_type=criterion.judgment_type,
                judgment_scores=criterion.judgment_scores,
            )

        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        }

    def _validate_judge_params(self, judge_params: JudgeParams) -> None:
        """Reject single-judgment settings that a rubric judge does not read.

        A rubric judge takes its judgment types, score mappings, and explanations
        from each criterion, so these `judge_params` fields would be silently ignored.

        Raises:
            ValueError: If any single-judgment field is set to a non-default value.
        """
        ignored_fields = []
        if judge_params.judgment_type != JudgeOutputType.BOOL:
            ignored_fields.append("judgment_type")
        if judge_params.judgment_scores is not None:
            ignored_fields.append("judgment_scores")
        if judge_params.include_explanation:
            ignored_fields.append("include_explanation")

        if ignored_fields:
            raise ValueError(
                f"judge_params.{', judge_params.'.join(ignored_fields)} "
                f"{'is' if len(ignored_fields) == 1 else 'are'} not used by "
                "RubricJudge, which reads these settings from each criterion in "
                "`rubric_judge_params.criteria`. Please move them there and remove "
                "them from `judge_params`."
            )

    def _warn_about_aggregation_gaps(self) -> None:
        """Warn about config that silently has no effect on the aggregate score."""
        self._warn_about_ignored_weights()
        self._warn_about_unscored_criteria()

    def _warn_about_ignored_weights(self) -> None:
        """Warn when weights are set under an aggregation that does not read them.

        Only WEIGHTED_MEAN consults `weight`; MIN and ALL are order statistics and
        NONE aggregates nothing, so a weight set under those is a silent no-op.
        """
        aggregation = self._rubric_params.aggregation
        if aggregation == JudgeAggregation.WEIGHTED_MEAN:
            return

        weighted = [c.id for c in self.criteria if c.weight != 1.0]
        if weighted:
            logger.warning(
                f"Criteria {sorted(weighted)} set a non-default `weight`, but "
                f"'{aggregation.value}' aggregation ignores weights. The weights "
                "have no effect. Use 'weighted_mean' aggregation, or remove them."
            )

    def _warn_about_unscored_criteria(self) -> None:
        """Warn about criteria that cannot contribute to the aggregate score."""
        if self._rubric_params.aggregation == JudgeAggregation.NONE:
            return

        # Only BOOL criteria (scored 1.0/0.0) and criteria with an explicit
        # `judgment_scores` mapping produce a numeric score; the rest are skipped
        # when aggregating.
        unscored = [
            criterion.id
            for criterion in self.criteria
            if not criterion.judgment_scores
            and criterion.judgment_type != JudgeOutputType.BOOL
        ]
        if unscored:
            logger.warning(
                f"Criteria {sorted(unscored)} produce no numeric score and will be "
                f"excluded from the '{self._rubric_params.aggregation.value}' "
                "aggregate score. Add `judgment_scores` to include them."
            )
