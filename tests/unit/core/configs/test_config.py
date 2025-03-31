import os
import tempfile
from pathlib import Path

import omegaconf
import pytest
from omegaconf import OmegaConf

from oumi.core.configs import (
    DatasetParams,
    EvaluationConfig,
    TrainingConfig,
)
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams


def test_config_serialization():
    with tempfile.TemporaryDirectory() as folder:
        original_config = TrainingConfig()
        dataset_params = DatasetParams(dataset_name="my_test_dataset")
        original_config.data.train.datasets = [dataset_params]
        original_config.model.model_name = "my_test_model"
        filename = os.path.join(folder, "test_config.yaml")
        original_config.to_yaml(filename)

        assert os.path.exists(filename)

        loaded_config = TrainingConfig.from_yaml(filename)
        assert loaded_config.model.model_name == "my_test_model"
        assert len(loaded_config.data.train.datasets) == 1
        assert loaded_config.data.train.datasets[0].dataset_name == "my_test_dataset"
        assert original_config == loaded_config

        loaded_config = TrainingConfig.from_yaml(Path(filename))
        assert original_config == loaded_config


def test_config_loading_from_str():
    yaml_str = """
            model:
                model_name: "my_test_model"

            data:
                train:
                    datasets:
                    - dataset_name: "my_test_dataset"
        """
    loaded_config = TrainingConfig.from_str(yaml_str)
    assert loaded_config.model.model_name == "my_test_model"
    assert len(loaded_config.data.train.datasets) == 1
    assert loaded_config.data.train.datasets[0].dataset_name == "my_test_dataset"


def test_config_equality():
    config_a = TrainingConfig()
    config_b = TrainingConfig()
    assert config_a == config_b

    config_a.model.model_name = "test_model"
    assert config_a != config_b


def test_config_override():
    low_priority_config = TrainingConfig()
    low_priority_config.model.model_name = "model_low_priority"

    high_priority_config = TrainingConfig()
    high_priority_config.model.model_name = "model_high_priority"

    # Override with CLI arguments if provided
    merged_config = OmegaConf.merge(low_priority_config, high_priority_config)
    assert merged_config.model.model_name == "model_high_priority"
    assert merged_config == high_priority_config
    assert merged_config != low_priority_config


def test_config_from_yaml_and_arg_list_override_values(tmp_path):
    task1 = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="mmlu",
        num_samples=5,
        eval_kwargs={"num_fewshot": 5},
    )
    task2 = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="hellaswag",
        num_samples=5,
        eval_kwargs={"num_fewshot": 5},
    )
    config = EvaluationConfig(tasks=[task1, task2])
    config.model.model_name = "foo"
    config_path = tmp_path / "eval.yaml"
    config.to_yaml(config_path)

    new_config = EvaluationConfig.from_yaml_and_arg_list(
        config_path,
        [
            "tasks[1].num_samples=1",  # override field
            "tasks[1].eval_kwargs.num_fewshot=1",  # override nested dict field
            "tasks[1].eval_kwargs.foo=bar",  # add new field
        ],
    )
    assert new_config.tasks == [
        EvaluationTaskParams(
            evaluation_backend="lm_harness",
            task_name="mmlu",
            num_samples=5,
            eval_kwargs={"num_fewshot": 5},
        ),
        EvaluationTaskParams(
            evaluation_backend="lm_harness",
            task_name="hellaswag",
            num_samples=1,
            eval_kwargs={"num_fewshot": 1, "foo": "bar"},
        ),
    ]


def test_config_from_yaml_and_arg_list_merge_dict(tmp_path):
    task = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="mmlu",
        num_samples=5,
        eval_kwargs={"num_fewshot": 5},
    )
    config = EvaluationConfig(tasks=[task])
    config.model.model_name = "foo"
    config_path = tmp_path / "eval.yaml"
    config.to_yaml(config_path)

    # By default, Omegaconf merges dicts together
    new_config = EvaluationConfig.from_yaml_and_arg_list(
        config_path,
        [
            "tasks.0.eval_kwargs={'foo': 'bar'}",
        ],
    )
    assert new_config.tasks == [
        EvaluationTaskParams(
            evaluation_backend="lm_harness",
            task_name="mmlu",
            num_samples=5,
            eval_kwargs={"num_fewshot": 5, "foo": "bar"},
        )
    ]


def test_config_from_yaml_and_arg_list_override_list(tmp_path):
    task = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="mmlu",
        num_samples=5,
        eval_kwargs={"num_fewshot": 5},
    )
    config = EvaluationConfig(tasks=[task])
    config.model.model_name = "foo"
    config_path = tmp_path / "eval.yaml"
    config.to_yaml(config_path)

    # By default, Omegaconf replaces lists
    new_config = EvaluationConfig.from_yaml_and_arg_list(
        config_path,
        [
            "tasks=[{'evaluation_backend': 'lm_harness', 'task_name': 'mmlu', "
            "'num_samples': 1, 'eval_kwargs': {'foo': 'bar'}}]",
        ],
    )
    assert new_config.tasks == [
        EvaluationTaskParams(
            evaluation_backend="lm_harness",
            task_name="mmlu",
            num_samples=1,
            eval_kwargs={"foo": "bar"},
        )
    ]


def test_config_from_yaml_and_arg_list_failure_nonexistent_index(tmp_path):
    task = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="mmlu",
        num_samples=5,
        eval_kwargs={"num_fewshot": 5},
    )
    config = EvaluationConfig(tasks=[task])
    config.model.model_name = "foo"
    config_path = tmp_path / "eval.yaml"
    config.to_yaml(config_path)

    with pytest.raises(omegaconf.errors.ValidationError):
        EvaluationConfig.from_yaml_and_arg_list(
            config_path,
            [
                "tasks[2].num_samples=1",  # index doesn't exist
            ],
        )


def test_config_from_yaml_and_arg_list_failure_empty_list(tmp_path):
    config = EvaluationConfig(tasks=[])
    config.model.model_name = "foo"
    config_path = tmp_path / "eval.yaml"
    config.to_yaml(config_path)

    with pytest.raises(omegaconf.errors.ValidationError):
        EvaluationConfig.from_yaml_and_arg_list(
            config_path,
            [
                "tasks[0].num_samples=1",  # index doesn't exist
            ],
        )
