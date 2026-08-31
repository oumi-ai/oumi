import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.judge import judge_dataset_file
from oumi.judges.base_judge import JudgeOutput

runner = CliRunner()


@pytest.fixture
def mock_parse_extra_cli_args():
    with patch("oumi.cli.cli_utils.parse_extra_cli_args") as m_parse:
        m_parse.return_value = {}
        yield m_parse


@pytest.fixture
def app():
    import typer

    judge_app = typer.Typer()
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(judge_dataset_file)
    yield judge_app


@pytest.fixture
def mock_judge_file():
    with patch("oumi.judge.judge_dataset_file") as m_jf:
        yield m_jf


@pytest.fixture
def mock_judge_config_from_path():
    with patch("oumi.core.configs.judge_config.JudgeConfig.from_path") as m_rjc:
        yield m_rjc


@pytest.fixture
def sample_judge_output():
    return JudgeOutput(
        raw_output="Test judgment",
        parsed_output={"quality": "good"},
        field_values={"quality": "good"},
        field_scores={"quality": 0.5},
    )


def test_judge_file(
    app,
    mock_parse_extra_cli_args,
    mock_judge_file,
    mock_judge_config_from_path,
    sample_judge_output,
):
    """Test that judge_file command runs successfully with all required parameters."""
    judge_config = "judge_config.yaml"
    input_file = "input.jsonl"

    mock_judge_file.return_value = [sample_judge_output]

    with patch("oumi.cli.judge.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        result = runner.invoke(
            app,
            [
                "dataset",
                "--config",
                judge_config,
                "--input",
                input_file,
            ],
        )

        assert result.exit_code == 0
        mock_parse_extra_cli_args.assert_called_once()
        mock_judge_config_from_path.assert_called_once_with(
            path=judge_config, extra_args={}
        )

        mock_judge_file.assert_called_once_with(
            judge_config=mock_judge_config_from_path.return_value,
            input_file=input_file,
            output_file=None,
        )


def test_judge_file_with_output_file(
    app,
    mock_parse_extra_cli_args,
    mock_judge_file,
    mock_judge_config_from_path,
    sample_judge_output,
):
    """Test that judge_file saves results to output file when specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        judge_config = "judge_config.yaml"
        input_file = "input.jsonl"
        output_file = str(Path(temp_dir) / "output.jsonl")

        mock_judge_file.return_value = [sample_judge_output]

        with patch("oumi.cli.judge.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            result = runner.invoke(
                app,
                [
                    "dataset",
                    "--config",
                    judge_config,
                    "--input",
                    input_file,
                    "--output",
                    output_file,
                ],
            )

            assert result.exit_code == 0
            mock_parse_extra_cli_args.assert_called_once()
            mock_judge_config_from_path.assert_called_once_with(
                path=judge_config, extra_args={}
            )

            mock_judge_file.assert_called_once_with(
                judge_config=mock_judge_config_from_path.return_value,
                input_file=input_file,
                output_file=output_file,
            )


#
# Rendering tests: these drive the real CLI and assert on what the user sees.
#


def _field(key, field_type=None):
    from oumi.core.configs.params.judge_params import JudgeOutputType
    from oumi.judges.base_judge import JudgeOutputField

    return JudgeOutputField(
        field_key=key,
        field_type=field_type or JudgeOutputType.BOOL,
        field_scores=None,
    )


def _output(values, scores, aggregate=None, raw="RAWTEXT"):
    return JudgeOutput(
        raw_output=raw,
        parsed_output={},
        output_fields=[_field(k) for k in values],
        field_values=values,
        field_scores=scores,
        aggregate_score=aggregate,
    )


def _simple(judgment, score, explanation="why"):
    return _output(
        {"explanation": explanation, "judgment": judgment},
        {"explanation": None, "judgment": score},
    )


def _rubric(values, scores, aggregate):
    return _output(values, scores, aggregate=aggregate)


def _run(app, outputs, extra_args=None):
    """Invoke the real CLI with a stubbed judge and return its stdout.

    Rich wraps output to the terminal width, so whitespace is collapsed before
    the caller asserts on it; otherwise "Judge Results" arrives as "Judge\nResults".
    """
    with (
        patch("oumi.judge.judge_dataset_file") as m_jf,
        patch("oumi.core.configs.judge_config.JudgeConfig.from_path"),
        patch("oumi.cli.cli_utils.parse_extra_cli_args", return_value={}),
        patch("oumi.cli.judge.Path") as mock_path,
    ):
        mock_path.return_value.exists.return_value = True
        m_jf.return_value = outputs
        result = runner.invoke(
            app,
            ["dataset", "--config", "c.yaml", "--input", "in.jsonl"]
            + (extra_args or []),
        )
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_render_single_judgment_shape(app):
    """A SimpleJudge-shaped result keeps its explanation column."""
    out = _run(app, [_simple(True, 1.0), _simple(False, 0.0)])

    assert "explanation" in out
    assert "judgment" in out
    assert "Score" in out
    assert "Overall Score: 50.00%" in out
    assert "Omitting" not in out


def test_render_omits_overall_score_when_a_row_lacks_one(app):
    out = _run(app, [_simple(True, 1.0), _simple(None, None)])

    assert "Overall Score" not in out
    assert "N/A" in out


def test_render_rule_based_shape(app):
    """A single 'judgment' field and no explanation still renders."""
    out = _run(app, [_output({"judgment": True}, {"judgment": 1.0})])

    assert "judgment" in out
    assert "Overall Score: 100.00%" in out


def test_render_rubric_keeps_a_lone_explanation(app):
    out = _run(
        app,
        [
            _rubric(
                {"a_explanation": "r", "a": True},
                {"a_explanation": None, "a": 1.0},
                1.0,
            )
        ],
    )

    assert "a_explanation" in out
    assert "Omitting" not in out


def test_render_rubric_omits_several_explanations(app):
    """Many explanation columns make the table unreadable, so they are dropped."""
    values = {"a_explanation": "r1", "a": True, "b_explanation": "r2", "b": False}
    scores = {"a_explanation": None, "a": 1.0, "b_explanation": None, "b": 0.0}
    out = _run(app, [_rubric(values, scores, 0.5)])

    assert "Omitting 2 explanation columns" in out
    assert "a_explanation" not in out
    assert "b_explanation" not in out
    # ...but the criteria themselves are still shown.
    assert "True" in out and "False" in out


def test_render_uses_aggregate_score_when_present(app):
    """The Score column and overall score come from aggregate_score for rubrics."""
    out = _run(app, [_rubric({"a": True}, {"a": 1.0}, 0.25)])

    assert "0.25" in out
    assert "Overall Score: 25.00%" in out


def test_render_aggregation_none_shows_no_score(app):
    out = _run(app, [_rubric({"a": True, "b": False}, {"a": 1.0, "b": 0.0}, None)])

    assert "Overall Score" not in out
    assert "N/A" in out


def test_render_unparsed_row_alongside_good_rows(app):
    good = _rubric({"a": True, "b": False}, {"a": 1.0, "b": 0.0}, 0.5)
    unparsed = _rubric({"a": None, "b": None}, {"a": None, "b": None}, None)
    out = _run(app, [good, unparsed])

    assert "N/A" in out
    assert "Overall Score" not in out


def test_render_raw_flag_adds_a_column(app):
    out = _run(app, [_simple(True, 1.0)], extra_args=["--raw"])

    assert "Raw Output" in out
    assert "RAWTEXT" in out


def test_render_empty_results(app):
    out = _run(app, [])

    assert "No judge outputs were produced" in out
    assert "Judge Results" not in out


def test_render_empty_results_with_output_file(app):
    """An empty run must still tell the user where the (empty) file went."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = str(Path(temp_dir) / "out.jsonl")
        out = _run(app, [], extra_args=["--output", output_file])

    assert "No judge outputs were produced" in out
    assert f"Results saved to {output_file}" in out


def test_render_missing_output_fields_does_not_crash(app):
    """Defensive: JudgeOutput.output_fields is Optional."""
    bare = JudgeOutput(raw_output="x", field_values={}, field_scores={})
    out = _run(app, [bare])

    assert "Judge Results" in out
    assert "N/A" in out


def test_render_no_table_when_output_file_given(app):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = str(Path(temp_dir) / "out.jsonl")
        out = _run(app, [_simple(True, 1.0)], extra_args=["--output", output_file])

    assert "Judge Results" not in out
    assert f"Results saved to {output_file}" in out
    # The overall score is still reported even when writing to a file.
    assert "Overall Score: 100.00%" in out
