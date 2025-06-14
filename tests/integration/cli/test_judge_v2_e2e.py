import json
import os
import tempfile
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.judge_v2 import judge_file
from oumi.utils.io_utils import save_jsonlines

skip_if_no_openai_key = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def app():
    judge_app = typer.Typer()
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(judge_file)
    yield judge_app


YAML_JUDGE_CONFIG = """
prompt_template: "Is the following Q&A correct? Question: {question} Answer: {answer}"
response_format: JSON
judgment_type: BOOL
include_explanation: False
"""

YAML_INFERENCE_CONFIG = """
model:
    model_name: gpt-4o-mini

engine: OPENAI

generation:
    max_new_tokens: 1024
    temperature: 0.0
"""

runner = CliRunner()


@skip_if_no_openai_key
def test_judge_with_output_file(app):
    """Test that judge saves the correct results into the output file."""

    test_data = [
        {
            "question": "Which is the capital of France?",
            "answer": "Paris",  # Correct answer
        },
        {
            "question": "What is 2+2?",
            "answer": "The answer is 5",  # Incorrect answer
        },
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        judge_config_path = str(Path(temp_dir) / "judge_config.yaml")
        inference_config_path = str(Path(temp_dir) / "inference_config.yaml")
        input_file_path = str(Path(temp_dir) / "input.jsonl")
        output_file_path = str(Path(temp_dir) / "output.jsonl")

        Path(judge_config_path).write_text(YAML_JUDGE_CONFIG)
        Path(inference_config_path).write_text(YAML_INFERENCE_CONFIG)
        save_jsonlines(input_file_path, test_data)

        result = runner.invoke(
            app,
            [
                "--judge-config",
                judge_config_path,
                "--inference-config",
                inference_config_path,
                "--input-file",
                input_file_path,
                "--output-file",
                output_file_path,
            ],
        )

        assert result.exit_code == 0, f"CLI command failed with: {result.exception}"
        assert Path(output_file_path).exists()

        # Verify output file content
        output_file_content = Path(output_file_path).read_text()
        output_file_rows = output_file_content.strip().split("\n")
        judge_outputs = [json.loads(row) for row in output_file_rows]

        assert judge_outputs[0]["field_values"]["judgment"] is True
        assert judge_outputs[0]["field_scores"]["judgment"] == 1.0
        assert judge_outputs[0]["parsed_output"]["judgment"] == "Yes"

        assert judge_outputs[1]["field_values"]["judgment"] is False
        assert judge_outputs[1]["field_scores"]["judgment"] == 0.0
        assert judge_outputs[1]["parsed_output"]["judgment"] == "No"
