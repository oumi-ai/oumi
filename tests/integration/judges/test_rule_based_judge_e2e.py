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

"""End-to-end integration tests for rule-based judges.

These tests verify that rule-based judges work correctly via the CLI interface.
Unlike LLM-based judges, rule-based judges don't require API keys and can run
deterministically without external dependencies.
"""

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from oumi.cli.main import get_app
from oumi.utils.io_utils import save_jsonlines

runner = CliRunner()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# CLI Integration Tests


def test_regex_match_phone_local_config():
    """Test the regex-match-phone judge config via CLI using local file."""
    test_data = [
        {"response": "Call me at 555-1234"},  # Should match
        {"response": "No phone number here"},  # Should not match
        {"response": "My numbers are 123-4567 and 987-6543"},  # Should match
    ]

    config_path = (
        PROJECT_ROOT / "configs/projects/judges/rule_based/regex_match_phone.yaml"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file_path = str(Path(temp_dir) / "input.jsonl")
        output_file_path = str(Path(temp_dir) / "output.jsonl")

        save_jsonlines(input_file_path, test_data)

        result = runner.invoke(
            get_app(),
            [
                "judge",
                "dataset",
                "--config",
                str(config_path),
                "--input",
                input_file_path,
                "--output",
                output_file_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.exception}"
        assert Path(output_file_path).exists()

        output_content = Path(output_file_path).read_text()
        output_rows = output_content.strip().split("\n")
        judge_outputs = [json.loads(row) for row in output_rows]

        assert len(judge_outputs) == 3
        assert judge_outputs[0]["field_values"]["judgment"] is True
        assert judge_outputs[0]["field_scores"]["judgment"] == 1.0
        assert judge_outputs[1]["field_values"]["judgment"] is False
        assert judge_outputs[1]["field_scores"]["judgment"] == 0.0
        assert judge_outputs[2]["field_values"]["judgment"] is True
        assert judge_outputs[2]["field_scores"]["judgment"] == 1.0


def test_regex_no_error_keywords_local_config():
    """Test the regex-no-error-keywords judge config via CLI using local file."""
    test_data = [
        {"response": "Success! Operation completed."},  # No errors - should pass
        {"response": "An error occurred"},  # Has error - should fail
        {"response": "Task failed to execute"},  # Has fail - should fail
        {"response": "Everything is working fine"},  # No errors - should pass
    ]

    config_path = (
        PROJECT_ROOT / "configs/projects/judges/rule_based/regex_no_error_keywords.yaml"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file_path = str(Path(temp_dir) / "input.jsonl")
        output_file_path = str(Path(temp_dir) / "output.jsonl")

        save_jsonlines(input_file_path, test_data)

        result = runner.invoke(
            get_app(),
            [
                "judge",
                "dataset",
                "--config",
                str(config_path),
                "--input",
                input_file_path,
                "--output",
                output_file_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.exception}"
        assert Path(output_file_path).exists()

        output_content = Path(output_file_path).read_text()
        output_rows = output_content.strip().split("\n")
        judge_outputs = [json.loads(row) for row in output_rows]

        assert len(judge_outputs) == 4
        # No errors - passes
        assert judge_outputs[0]["field_values"]["judgment"] is True
        assert judge_outputs[0]["field_scores"]["judgment"] == 1.0
        # Has "error" - fails
        assert judge_outputs[1]["field_values"]["judgment"] is False
        assert judge_outputs[1]["field_scores"]["judgment"] == 0.0
        # Has "fail" - fails
        assert judge_outputs[2]["field_values"]["judgment"] is False
        assert judge_outputs[2]["field_scores"]["judgment"] == 0.0
        # No errors - passes
        assert judge_outputs[3]["field_values"]["judgment"] is True
        assert judge_outputs[3]["field_scores"]["judgment"] == 1.0


def test_custom_regex_rule_config():
    """Test a custom rule-based judge config via CLI."""
    test_data = [
        {"text": "user@example.com"},  # Valid email
        {"text": "invalid-email"},  # Invalid email
        {"text": "Contact: test@domain.org"},  # Valid email in text
    ]

    yaml_config = """
judge_params:
    prompt_template: "{text}"

rule_judge_params:
    rule_type: "regex"
    input_fields:
        - "text"
    rule_config:
        pattern: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}"
        input_field: "text"
        match_mode: "search"
    response_format: XML
    judgment_type: BOOL
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = str(Path(temp_dir) / "email_judge.yaml")
        input_file_path = str(Path(temp_dir) / "input.jsonl")
        output_file_path = str(Path(temp_dir) / "output.jsonl")

        Path(config_path).write_text(yaml_config)
        save_jsonlines(input_file_path, test_data)

        result = runner.invoke(
            get_app(),
            [
                "judge",
                "dataset",
                "--config",
                config_path,
                "--input",
                input_file_path,
                "--output",
                output_file_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.exception}"
        assert Path(output_file_path).exists()

        output_content = Path(output_file_path).read_text()
        output_rows = output_content.strip().split("\n")
        judge_outputs = [json.loads(row) for row in output_rows]

        assert len(judge_outputs) == 3
        assert judge_outputs[0]["field_values"]["judgment"] is True
        assert judge_outputs[1]["field_values"]["judgment"] is False
        assert judge_outputs[2]["field_values"]["judgment"] is True


def test_rule_based_judge_json_output_format():
    """Test rule-based judge with JSON output format."""
    test_data = [{"response": "555-1234"}]

    yaml_config = """
judge_params:
    prompt_template: "{response}"

rule_judge_params:
    rule_type: "regex"
    input_fields:
        - "response"
    rule_config:
        pattern: "\\\\d{3}-\\\\d{4}"
        input_field: "response"
    response_format: JSON
    judgment_type: BOOL
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = str(Path(temp_dir) / "judge_config.yaml")
        input_file_path = str(Path(temp_dir) / "input.jsonl")
        output_file_path = str(Path(temp_dir) / "output.jsonl")

        Path(config_path).write_text(yaml_config)
        save_jsonlines(input_file_path, test_data)

        result = runner.invoke(
            get_app(),
            [
                "judge",
                "dataset",
                "--config",
                config_path,
                "--input",
                input_file_path,
                "--output",
                output_file_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.exception}"

        output_content = Path(output_file_path).read_text()
        judge_output = json.loads(output_content.strip())

        assert judge_output["response_format"] == "json"
        assert judge_output["field_values"]["judgment"] is True


def test_rule_based_judge_case_insensitive():
    """Test rule-based judge with case-insensitive regex."""
    test_data = [
        {"text": "ERROR: something went wrong"},  # uppercase - should match
        {"text": "error: something went wrong"},  # lowercase - should match
        {"text": "No problems here"},  # no "error" - should NOT match
    ]

    # re.IGNORECASE = 2
    yaml_config = """
judge_params:
    prompt_template: "{text}"

rule_judge_params:
    rule_type: "regex"
    input_fields:
        - "text"
    rule_config:
        pattern: "error"
        input_field: "text"
        flags: 2
    response_format: XML
    judgment_type: BOOL
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = str(Path(temp_dir) / "judge_config.yaml")
        input_file_path = str(Path(temp_dir) / "input.jsonl")
        output_file_path = str(Path(temp_dir) / "output.jsonl")

        Path(config_path).write_text(yaml_config)
        save_jsonlines(input_file_path, test_data)

        result = runner.invoke(
            get_app(),
            [
                "judge",
                "dataset",
                "--config",
                config_path,
                "--input",
                input_file_path,
                "--output",
                output_file_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.exception}"

        output_content = Path(output_file_path).read_text()
        output_rows = output_content.strip().split("\n")
        judge_outputs = [json.loads(row) for row in output_rows]

        assert len(judge_outputs) == 3
        # Both "ERROR" and "error" should match
        assert judge_outputs[0]["field_values"]["judgment"] is True
        assert judge_outputs[1]["field_values"]["judgment"] is True
        # "No problems here" should NOT match (no "error" substring)
        assert judge_outputs[2]["field_values"]["judgment"] is False


# Python API Integration Tests


def test_rule_based_judge_direct_usage():
    """Test using RuleBasedJudge directly via Python API."""
    from oumi.core.configs.judge_config import JudgeConfig
    from oumi.core.configs.params.judge_params import JudgeParams
    from oumi.core.configs.params.rule_judge_params import RuleJudgeParams
    from oumi.judges.rule_based_judge import RuleBasedJudge

    judge_config = JudgeConfig(
        judge_params=JudgeParams(prompt_template="{text}"),
        rule_judge_params=RuleJudgeParams(
            rule_type="regex",
            input_fields=["text"],
            rule_config={
                "pattern": r"\d+",
                "input_field": "text",
            },
        ),
    )

    judge = RuleBasedJudge(judge_config)

    inputs = [
        {"text": "The answer is 42"},
        {"text": "No numbers here"},
        {"text": "123 and 456"},
    ]

    results = judge.judge(inputs)

    assert len(results) == 3
    assert results[0].field_values["judgment"] is True
    assert results[1].field_values["judgment"] is False
    assert results[2].field_values["judgment"] is True


def test_rule_based_judge_with_config_file():
    """Test loading RuleBasedJudge from a config file."""
    from oumi.judges.rule_based_judge import RuleBasedJudge

    yaml_config = """
judge_params:
    prompt_template: "{output}"

rule_judge_params:
    rule_type: "regex"
    input_fields:
        - "output"
    rule_config:
        pattern: "success"
        input_field: "output"
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(yaml_config)
        temp_file_path = temp_file.name

    try:
        judge = RuleBasedJudge(temp_file_path)

        inputs = [
            {"output": "Operation completed with success"},
            {"output": "Operation failed"},
        ]

        results = judge.judge(inputs)

        assert len(results) == 2
        assert results[0].field_values["judgment"] is True
        assert results[1].field_values["judgment"] is False
    finally:
        Path(temp_file_path).unlink()


def test_rule_based_judge_output_structure():
    """Test that RuleBasedJudge output has correct structure."""
    from oumi.core.configs.judge_config import JudgeConfig
    from oumi.core.configs.params.judge_params import (
        JudgeOutputType,
        JudgeParams,
        JudgeResponseFormat,
    )
    from oumi.core.configs.params.rule_judge_params import RuleJudgeParams
    from oumi.judges.rule_based_judge import RuleBasedJudge

    judge_config = JudgeConfig(
        judge_params=JudgeParams(prompt_template="{text}"),
        rule_judge_params=RuleJudgeParams(
            rule_type="regex",
            input_fields=["text"],
            rule_config={"pattern": r"test", "input_field": "text"},
            response_format=JudgeResponseFormat.JSON,
            judgment_type=JudgeOutputType.BOOL,
        ),
    )

    judge = RuleBasedJudge(judge_config)
    results = judge.judge([{"text": "test"}])

    result = results[0]
    assert "judgment" in result.field_values
    assert "judgment" in result.field_scores
    assert result.raw_output is not None
    assert result.response_format == JudgeResponseFormat.JSON
    assert result.output_fields is not None
    assert len(result.output_fields) == 1
    assert result.output_fields[0].field_key == "judgment"
    assert result.output_fields[0].field_type == JudgeOutputType.BOOL
