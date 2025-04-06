import logging
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
import oumi.cli.alias
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.evaluate import evaluate
from oumi.core.configs import (
    EvaluationConfig,
    EvaluationTaskParams,
    ModelParams,
)
from oumi.utils.logging import logger

runner = CliRunner()


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        yield m_fetch


@pytest.fixture
def mock_alias():
    with patch("oumi.cli.evaluate.try_get_config_name_for_alias") as try_alias:
        yield try_alias


def _create_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        output_dir="output/dir",
        tasks=[
            EvaluationTaskParams(
                evaluation_backend="lm_harness",
                task_name="mmlu",
                num_samples=4,
            )
        ],
        model=ModelParams(
            model_name="MlpEncoder",
            trust_remote_code=True,
            tokenizer_name="gpt2",
        ),
    )


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(evaluate)
    yield fake_app


@pytest.fixture
def mock_evaluate():
    with patch.object(oumi, "evaluate", autospec=True) as m_evaluate:
        yield m_evaluate


def test_evaluate_runs(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path])
        mock_evaluate.assert_has_calls([call(config)])


def test_evaluate_calls_alias(app, mock_evaluate, mock_alias):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        mock_alias.return_value = yaml_path
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", "an_alias"])
        mock_alias.assert_has_calls(
            [
                call("an_alias", oumi.cli.alias.AliasType.EVAL),
            ]
        )
        mock_evaluate.assert_has_calls([call(config)])


def test_evaluate_unparsable_metrics(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        mock_evaluate.return_value = [
            {
                "results": {
                    "mmlu_college_computer_science": {
                        "alias": "college_computer_science",
                        "acc,none": 0.24,
                        "acc_stderr,none": 0.042923469599092816,
                    }
                }
            },
            {
                "results": {
                    "leaderboard_bbh": {" ": " ", "alias": "leaderboard_bbh"},
                    "leaderboard_bbh_boolean_expressions": {
                        "alias": " - leaderboard_bbh_boolean_expressions",
                        "acc_norm,none": 0.436,
                        "acc_norm_stderr,none": 0.03142556706028128,
                    },
                    "leaderboard_bbh_causal_judgement": {
                        "alias": " - leaderboard_bbh_causal_judgement",
                        "acc_norm,none": 0.5187165775401069,
                        "acc_norm_stderr,none": 0.03663608375537842,
                    },
                    "leaderboard_bbh_date_understanding": {
                        "alias": " - leaderboard_bbh_date_understanding",
                        "acc_norm,none": 0.192,
                        "acc_norm_stderr,none": 0.024960691989171994,
                    },
                    "leaderboard_bbh_disambiguation_qa": {
                        "alias": " - leaderboard_bbh_disambiguation_qa",
                        "acc_norm,none": 0.312,
                        "acc_norm_stderr,none": 0.029361067575219817,
                    },
                    "leaderboard_bbh_formal_fallacies": {
                        "alias": " - leaderboard_bbh_formal_fallacies",
                        "acc_norm,none": 0.468,
                        "acc_norm_stderr,none": 0.031621252575725504,
                    },
                    "leaderboard_bbh_geometric_shapes": {
                        "alias": " - leaderboard_bbh_geometric_shapes",
                        "acc_norm,none": 0.0,
                        "acc_norm_stderr,none": 0.0,
                    },
                    "leaderboard_bbh_hyperbaton": {
                        "alias": " - leaderboard_bbh_hyperbaton",
                        "acc_norm,none": 0.528,
                        "acc_norm_stderr,none": 0.0316364895315444,
                    },
                    "leaderboard_bbh_logical_deduction_five_objects": {
                        "alias": " - leaderboard_bbh_logical_deduction_five_objects",
                        "acc_norm,none": 0.188,
                        "acc_norm_stderr,none": 0.024760377727750513,
                    },
                    "leaderboard_bbh_logical_deduction_seven_objects": {
                        "alias": " - leaderboard_bbh_logical_deduction_seven_objects",
                        "acc_norm,none": 0.14,
                        "acc_norm_stderr,none": 0.02198940964524027,
                    },
                    "leaderboard_bbh_logical_deduction_three_objects": {
                        "alias": " - leaderboard_bbh_logical_deduction_three_objects",
                        "acc_norm,none": 0.34,
                        "acc_norm_stderr,none": 0.030020073605457904,
                    },
                    "leaderboard_bbh_movie_recommendation": {
                        "alias": " - leaderboard_bbh_movie_recommendation",
                        "acc_norm,none": 0.22,
                        "acc_norm_stderr,none": 0.02625179282460584,
                    },
                    "leaderboard_bbh_navigate": {
                        "alias": " - leaderboard_bbh_navigate",
                        "acc_norm,none": 0.42,
                        "acc_norm_stderr,none": 0.03127799950463662,
                    },
                    "leaderboard_bbh_object_counting": {
                        "alias": " - leaderboard_bbh_object_counting",
                        "acc_norm,none": 0.048,
                        "acc_norm_stderr,none": 0.013546884228085693,
                    },
                    "leaderboard_bbh_penguins_in_a_table": {
                        "alias": " - leaderboard_bbh_penguins_in_a_table",
                        "acc_norm,none": 0.2465753424657534,
                        "acc_norm_stderr,none": 0.03579404139909799,
                    },
                    "leaderboard_bbh_reasoning_about_colored_objects": {
                        "alias": " - leaderboard_bbh_reasoning_about_colored_objects",
                        "acc_norm,none": 0.156,
                        "acc_norm_stderr,none": 0.022995023034068734,
                    },
                    "leaderboard_bbh_ruin_names": {
                        "alias": " - leaderboard_bbh_ruin_names",
                        "acc_norm,none": 0.292,
                        "acc_norm_stderr,none": 0.028814320402205638,
                    },
                    "leaderboard_bbh_salient_translation_error_detection": {
                        "alias": " - leaderboard_bbh_salient_translation_error_detection",  # noqa E501
                        "acc_norm,none": 0.14,
                        "acc_norm_stderr,none": 0.02198940964524027,
                    },
                    "leaderboard_bbh_snarks": {
                        "alias": " - leaderboard_bbh_snarks",
                        "acc_norm,none": 0.46629213483146065,
                        "acc_norm_stderr,none": 0.037496800603689866,
                    },
                    "leaderboard_bbh_sports_understanding": {
                        "alias": " - leaderboard_bbh_sports_understanding",
                        "acc_norm,none": 0.46,
                        "acc_norm_stderr,none": 0.031584653891499,
                    },
                    "leaderboard_bbh_temporal_sequences": {
                        "alias": " - leaderboard_bbh_temporal_sequences",
                        "acc_norm,none": 0.284,
                        "acc_norm_stderr,none": 0.02857695873043742,
                    },
                    "leaderboard_bbh_tracking_shuffled_objects_five_objects": {
                        "alias": " - leaderboard_bbh_tracking_shuffled_objects_five_objects",  # noqa E501
                        "acc_norm,none": 0.2,
                        "acc_norm_stderr,none": 0.025348970020979085,
                    },
                    "leaderboard_bbh_tracking_shuffled_objects_seven_objects": {
                        "alias": " - leaderboard_bbh_tracking_shuffled_objects_seven_objects",  # noqa E501
                        "acc_norm,none": 0.12,
                        "acc_norm_stderr,none": 0.020593600596839946,
                    },
                    "leaderboard_bbh_tracking_shuffled_objects_three_objects": {
                        "alias": " - leaderboard_bbh_tracking_shuffled_objects_three_objects",  # noqa E501
                        "acc_norm,none": 0.304,
                        "acc_norm_stderr,none": 0.029150213374159677,
                    },
                    "leaderboard_bbh_web_of_lies": {
                        "alias": " - leaderboard_bbh_web_of_lies",
                        "acc_norm,none": 0.488,
                        "acc_norm_stderr,none": 0.03167708558254708,
                    },
                },
                "start_time": "20250316_150150",
                "duration_sec": 394,
            },
            {"results": {"mmlu": {"parsable_metric": 1.0}}},
            {
                "results": {
                    "q": {
                        "name": "our name",
                        "second_metric": 0.5,
                        "parsable_metric": 0.77,
                    },
                    "unparsable": "real name",
                }
            },
            {"results": []},
        ]
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        result = runner.invoke(app, ["--config", yaml_path])
        mock_evaluate.assert_has_calls([call(config)])
        assert result.exit_code == 0
        tables = [
            """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Benchmark                ┃ Metric ┃ Score  ┃ Std Error ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ college_computer_science │ Acc    │ 24.00% │ ±4.29%    │
└──────────────────────────┴────────┴────────┴───────────┘""",
            """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Benchmark                                    ┃ Metric   ┃ Score  ┃ Std Error ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│  - leaderboard_bbh_boolean_expressions       │ Acc Norm │ 43.60% │ ±3.14%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_causal_judgement          │ Acc Norm │ 51.87% │ ±3.66%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_date_understanding        │ Acc Norm │ 19.20% │ ±2.50%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_disambiguation_qa         │ Acc Norm │ 31.20% │ ±2.94%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_formal_fallacies          │ Acc Norm │ 46.80% │ ±3.16%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_geometric_shapes          │ Acc Norm │ 0.00%  │ ±0.00%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_hyperbaton                │ Acc Norm │ 52.80% │ ±3.16%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 18.80% │ ±2.48%    │
│ leaderboard_bbh_logical_deduction_five_obje… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 14.00% │ ±2.20%    │
│ leaderboard_bbh_logical_deduction_seven_obj… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 34.00% │ ±3.00%    │
│ leaderboard_bbh_logical_deduction_three_obj… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_movie_recommendation      │ Acc Norm │ 22.00% │ ±2.63%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_navigate                  │ Acc Norm │ 42.00% │ ±3.13%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_object_counting           │ Acc Norm │ 4.80%  │ ±1.35%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_penguins_in_a_table       │ Acc Norm │ 24.66% │ ±3.58%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 15.60% │ ±2.30%    │
│ leaderboard_bbh_reasoning_about_colored_obj… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_ruin_names                │ Acc Norm │ 29.20% │ ±2.88%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 14.00% │ ±2.20%    │
│ leaderboard_bbh_salient_translation_error_d… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_snarks                    │ Acc Norm │ 46.63% │ ±3.75%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_sports_understanding      │ Acc Norm │ 46.00% │ ±3.16%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_temporal_sequences        │ Acc Norm │ 28.40% │ ±2.86%    │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 20.00% │ ±2.53%    │
│ leaderboard_bbh_tracking_shuffled_objects_f… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 12.00% │ ±2.06%    │
│ leaderboard_bbh_tracking_shuffled_objects_s… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  -                                           │ Acc Norm │ 30.40% │ ±2.92%    │
│ leaderboard_bbh_tracking_shuffled_objects_t… │          │        │           │
├──────────────────────────────────────────────┼──────────┼────────┼───────────┤
│  - leaderboard_bbh_web_of_lies               │ Acc Norm │ 48.80% │ ±3.17%    │
└──────────────────────────────────────────────┴──────────┴────────┴───────────┘""",
            """
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃ Benchmark ┃ Metric          ┃ Score   ┃ Std Error ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩
│ mmlu      │ Parsable Metric │ 100.00% │ -         │
└───────────┴─────────────────┴─────────┴───────────┘""",
            """
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Benchmark  ┃ Metric          ┃ Score     ┃ Std Error ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ q          │ Second Metric   │ 50.00%    │ -         │
├────────────┼─────────────────┼───────────┼───────────┤
│ q          │ Parsable Metric │ 77.00%    │ -         │
├────────────┼─────────────────┼───────────┼───────────┤
│ unparsable │ <unknown>       │ <unknown> │ -         │
└────────────┴─────────────────┴───────────┴───────────┘""",
        ]
        for table in tables:
            assert table in result.stdout


def test_evaluate_with_overrides(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--model.tokenizer_name",
                "new_name",
                "--tasks",
                "[{evaluation_backend: lm_harness, num_samples: 5, task_name: mmlu}]",
            ],
        )
        expected_config = _create_eval_config()
        expected_config.model.tokenizer_name = "new_name"
        if expected_config.tasks:
            if expected_config.tasks[0]:
                expected_config.tasks[0].num_samples = 5
        mock_evaluate.assert_has_calls([call(expected_config)])


def test_evaluate_logging_levels(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path, "--log-level", "DEBUG"])
        assert logger.level == logging.DEBUG
        _ = runner.invoke(app, ["--config", yaml_path, "-log", "WARNING"])
        assert logger.level == logging.WARNING


def test_evaluate_with_oumi_prefix(app, mock_evaluate, mock_fetch):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_dir = Path(output_temp_dir)
        yaml_path = "oumi://configs/recipes/smollm/evaluation/135m/eval.yaml"
        expected_path = output_dir / "configs/recipes/smollm/evaluation/135m/eval.yaml"

        config: EvaluationConfig = _create_eval_config()
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(expected_path)
        mock_fetch.return_value = expected_path

        with patch.dict("os.environ", {"OUMI_DIR": str(output_dir)}):
            result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(yaml_path)
        mock_evaluate.assert_has_calls([call(config)])
