from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import optuna  # type: ignore[reportMissingImports]
except ImportError:
    pytest.skip("Optuna not installed", allow_module_level=True)

from oumi.core.configs import TuningConfig
from oumi.core.configs.params.tuning_params import TuningParams
from oumi.core.tuners.optuna_tuner import OptunaTuner


#
# Fixtures
#
@pytest.fixture
def mock_tuning_params():
    """Create mock TuningParams for testing."""
    params = MagicMock(spec=TuningParams)
    params.evaluation_direction = ["minimize", "maximize"]
    params.evaluation_metrics = ["eval_loss", "eval_accuracy"]
    params.tuner_sampler = "TPESampler"
    params.storage = None
    params.tuning_study_name = "test_study"
    params.load_if_exists = False
    params.tunable_training_params = {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2,
        },
        "optimizer": {
            "type": "categorical",
            "choices": ["adamw_torch", "sgd"],
        },
        "warmup_ratio": {
            "type": "uniform",
            "low": 0.0,
            "high": 0.3,
        },
        "gradient_accumulation_steps": {
            "type": "int",
            "low": 1,
            "high": 8,
        },
    }
    params.tunable_peft_params = {
        "lora_r": {
            "type": "int",
            "low": 8,
            "high": 64,
        }
    }
    params.output_dir = "/tmp/test_tuning"
    return params


@pytest.fixture
def tuner(mock_tuning_params: TuningParams):
    """Create OptunaTuner instance for testing."""
    return OptunaTuner(mock_tuning_params)


@pytest.fixture
def mock_tuning_config():
    """Create a mock TuningConfig."""
    config = MagicMock(spec=TuningConfig)
    config.tuning = MagicMock(spec=TuningParams)
    config.tuning.output_dir = "/tmp/test_tuning"
    return config


@pytest.fixture
def mock_objective_fn():
    """Create a mock objective function."""

    def objective(_train_params, _test_params, trial_number):
        return {
            "eval_loss": 0.5 - trial_number * 0.01,
            "eval_accuracy": 0.7 + trial_number * 0.01,
        }

    return objective


#
# Tests
#
def test_tuner_initialization(tuner, mock_tuning_params):
    """Test that OptunaTuner initializes correctly."""
    assert tuner.tuning_params == mock_tuning_params
    assert not hasattr(tuner, "_study") or tuner._study is None


def test_create_study_with_tpe_sampler(tuner):
    """Test study creation with TPE sampler."""
    tuner.create_study()

    assert tuner._study is not None
    assert isinstance(tuner._sampler, optuna.samplers.TPESampler)
    assert len(tuner._study.directions) == 2
    assert tuner._study.directions[0] == optuna.study.StudyDirection.MINIMIZE
    assert tuner._study.directions[1] == optuna.study.StudyDirection.MAXIMIZE


def test_create_study_with_random_sampler(tuner, mock_tuning_params):
    """Test study creation with Random sampler."""
    mock_tuning_params.tuner_sampler = "RandomSampler"
    tuner = OptunaTuner(mock_tuning_params)
    tuner.create_study()

    assert isinstance(tuner._sampler, optuna.samplers.RandomSampler)


def test_create_study_with_invalid_sampler(tuner, mock_tuning_params):
    """Test that invalid sampler raises ValueError."""
    mock_tuning_params.tuner_sampler = "InvalidSampler"
    tuner = OptunaTuner(mock_tuning_params)

    with pytest.raises(ValueError, match="Unsupported sampler"):
        tuner.create_study()


def test_create_study_with_invalid_direction(tuner, mock_tuning_params):
    """Test that invalid direction raises ValueError."""
    mock_tuning_params.evaluation_direction = ["invalid_direction"]
    tuner = OptunaTuner(mock_tuning_params)

    with pytest.raises(ValueError, match="Unsupported optimization direction"):
        tuner.create_study()


def test_suggest_parameters_categorical(tuner):
    """Test parameter suggestion for categorical parameters."""
    tuner.create_study()
    trial = tuner._study.ask()

    train_params, _ = tuner.suggest_parameters(trial)

    assert "optimizer" in train_params
    assert train_params["optimizer"] in ["adamw_torch", "sgd"]


def test_suggest_parameters_loguniform(tuner):
    """Test parameter suggestion for loguniform parameters."""
    tuner.create_study()
    trial = tuner._study.ask()

    train_params, _ = tuner.suggest_parameters(trial)

    assert "learning_rate" in train_params
    assert 1e-5 <= train_params["learning_rate"] <= 1e-2


def test_suggest_parameters_uniform(tuner):
    """Test parameter suggestion for uniform parameters."""
    tuner.create_study()
    trial = tuner._study.ask()

    train_params, _ = tuner.suggest_parameters(trial)

    assert "warmup_ratio" in train_params
    assert 0.0 <= train_params["warmup_ratio"] <= 0.3


def test_suggest_parameters_int(tuner):
    """Test parameter suggestion for integer parameters."""
    tuner.create_study()
    trial = tuner._study.ask()

    train_params, _ = tuner.suggest_parameters(trial)

    assert "gradient_accumulation_steps" in train_params
    assert 1 <= train_params["gradient_accumulation_steps"] <= 8
    assert isinstance(train_params["gradient_accumulation_steps"], int)


def test_suggest_parameters_peft(tuner):
    """Test parameter suggestion for PEFT parameters."""
    tuner.create_study()
    trial = tuner._study.ask()

    _, peft_params = tuner.suggest_parameters(trial)

    assert "lora_r" in peft_params
    assert 8 <= peft_params["lora_r"] <= 64


def test_optimize_single_objective(tuner, mock_objective_fn):
    """Test optimization with single objective."""
    tuner.tuning_params.evaluation_metrics = ["eval_loss"]
    tuner.tuning_params.evaluation_direction = ["minimize"]

    tuner.optimize(mock_objective_fn, n_trials=3)

    assert tuner._study is not None
    assert len(tuner._study.trials) == 3


def test_optimize_multi_objective(tuner, mock_objective_fn):
    """Test optimization with multiple objectives."""
    tuner.optimize(mock_objective_fn, n_trials=3)

    assert tuner._study is not None
    assert len(tuner._study.trials) == 3
    # Check that each trial has multiple values
    for trial in tuner._study.trials:
        assert len(trial.values) == 2


def test_get_best_trial_single_objective(tuner, mock_objective_fn):
    """Test getting best trial for single objective."""
    tuner.tuning_params.evaluation_metrics = ["eval_loss"]
    tuner.tuning_params.evaluation_direction = ["minimize"]

    tuner.optimize(mock_objective_fn, n_trials=3)
    best_trial = tuner.get_best_trial()

    assert "trial_number" in best_trial
    assert "params" in best_trial
    assert "values" in best_trial
    assert isinstance(best_trial["params"], dict)


def test_get_best_trials_multi_objective(tuner, mock_objective_fn):
    """Test getting best trials for multi-objective optimization."""
    tuner.optimize(mock_objective_fn, n_trials=5)
    best_trials = tuner.get_best_trials()

    assert isinstance(best_trials, list)
    assert len(best_trials) > 0
    for trial in best_trials:
        assert "trial_number" in trial
        assert "params" in trial
        assert "values" in trial


def test_get_best_trial_without_study(tuner):
    """Test that getting best trial without study raises error."""
    with pytest.raises(RuntimeError, match="Study not created"):
        tuner.get_best_trial()

    with pytest.raises(RuntimeError, match="Study not created"):
        tuner.get_best_trials()


def test_save_study(tuner, mock_objective_fn, mock_tuning_config, tmp_path):
    """Test saving study results to CSV."""
    output_dir = Path(mock_tuning_config.tuning.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tuner.optimize(mock_objective_fn, n_trials=3)

    tuner.save_study(mock_tuning_config)

    # Checking the if the csv file creation was sucessfull
    csv_path = output_dir / "trials_results.csv"
    assert csv_path.exists()
    assert csv_path.is_file()


def test_suggest_invalid_parameters(tuner, mock_tuning_params):
    """Test that invalid parameter raises ValueError."""
    mock_tuning_params.tunable_training_params = {
        "invalid_param": 123  # Invalid spec (not list or dict)
    }
    tuner = OptunaTuner(mock_tuning_params)
    tuner.create_study()
    trial = tuner._study.ask()

    with pytest.raises(ValueError, match="Parameter specification .* is invalid"):
        tuner.suggest_parameters(trial)

    mock_tuning_params.tunable_training_params = {
        "invalid_param": {"type": "invalid_type", "low": 0, "high": 1}
    }
    tuner = OptunaTuner(mock_tuning_params)
    tuner.create_study()
    trial = tuner._study.ask()

    with pytest.raises(ValueError, match=" is not a valid ParamType"):
        tuner.suggest_parameters(trial)
