import os
import tempfile
from pathlib import Path

from oumi.core.configs.params.data_params import DatasetParams
from oumi.core.configs.tuning_config import TuningConfig


def test_tuning_config_serialization():
    """Test that TuningConfig can be serialized to/from YAML."""
    with tempfile.TemporaryDirectory() as folder:
        original_config = TuningConfig()
        dataset_params = DatasetParams(dataset_name="my_test_dataset")
        original_config.data.train.datasets = [dataset_params]
        original_config.model.model_name = "my_test_model"
        original_config.tuning.tuning_study_name = "test_study"

        filename = os.path.join(folder, "test_tuning_config.yaml")
        original_config.to_yaml(filename)

        assert os.path.exists(filename)

        loaded_config = TuningConfig.from_yaml(filename)
        assert loaded_config.model.model_name == "my_test_model"
        assert len(loaded_config.data.train.datasets) == 1
        assert loaded_config.data.train.datasets[0].dataset_name == "my_test_dataset"
        assert loaded_config.tuning.tuning_study_name == "test_study"
        assert original_config == loaded_config

        # Test with Path object
        loaded_config = TuningConfig.from_yaml(Path(filename))
        assert original_config == loaded_config
