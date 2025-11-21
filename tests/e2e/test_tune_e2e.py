import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

import pytest

from oumi.core.configs import TuningConfig
from oumi.utils.torch_utils import device_cleanup
from tests import get_configs_dir
from tests.e2e import get_e2e_test_output_dir
from tests.markers import requires_gpus


class TuneTestConfig(NamedTuple):
    test_name: str
    config_path: Path
    n_trials: int
    max_steps: int
    skip: bool = False
    interactive_logs: bool = True

    model_max_length: Optional[int] = None
    batch_size: Optional[int] = None
    enable_wandb: Optional[bool] = False


def get_tune_test_id_fn(val):
    assert isinstance(val, TuneTestConfig), f"{type(val)}: {val}"
    return val.test_name


def _test_tune_impl(
    test_config: TuneTestConfig,
    tmp_path: Path,
    *,
    cleanup_output_dir_on_success: bool = True,
):
    device_cleanup()
    if test_config.skip:
        pytest.skip(f"Skipped the test '{test_config.test_name}'!")
        return

    interactive_logs = test_config.interactive_logs
    test_tag = f"[{test_config.test_name}]"

    _START_TIME = time.perf_counter()
    output_dir = get_e2e_test_output_dir(test_config.test_name, tmp_path=tmp_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Verify config is loadable
        assert test_config.config_path.exists(), (
            f"{test_tag} Path doesn't exist: {test_config.config_path}"
        )
        assert test_config.config_path.is_file(), (
            f"{test_tag} Path is not a file: {test_config.config_path}"
        )

        try:
            TuningConfig.from_yaml(test_config.config_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tuning config from: {test_config.config_path}"
            ) from e

        assert test_config.n_trials > 0, f"n_trials: {test_config.n_trials}"
        assert test_config.max_steps > 0, f"max_steps: {test_config.max_steps}"

        # Build command
        cmd = [
            "oumi",
            "tune",
            "-c",
            str(test_config.config_path),
            "--tuning.n_trials",
            str(test_config.n_trials),
            "--tuning.fixed_training_params.max_steps",
            str(test_config.max_steps),
            "--tuning.fixed_training_params.run_name",
            test_config.test_name,
            "--tuning.output_dir",
            str(output_dir / "tune"),
        ]

        # Add optional parameters
        if test_config.model_max_length is not None:
            cmd.append(f"--model.model_max_length={test_config.model_max_length}")

        if test_config.batch_size is not None:
            cmd.append(
                f"--tuning.fixed_training_params.per_device_train_batch_size={test_config.batch_size}"
            )

        if test_config.enable_wandb is not None:
            cmd.append(
                f"--tuning.fixed_training_params.enable_wandb={test_config.enable_wandb}"
            )

        env_vars = dict(os.environ)
        if "TOKENIZERS_PARALLELISM" not in env_vars:
            env_vars["TOKENIZERS_PARALLELISM"] = "false"

        shell_command = " ".join(cmd)
        print(f"{test_tag} Running the command:\n{shell_command}\n")
        device_cleanup()

        result = subprocess.run(
            shell_command,
            shell=True,
            text=True,
            capture_output=(not interactive_logs),
            stdout=(sys.stdout if interactive_logs else None),
            stderr=(sys.stderr if interactive_logs else None),
            env=env_vars,
        )

        duration_sec = time.perf_counter() - _START_TIME
        if result.returncode == 0:
            print(
                f"{test_tag} Tuning job successfully finished in {duration_sec:.2f}s!"
            )
        else:
            print(
                f"{test_tag} Tuning job failed with error code: {result.returncode} in "
                f"{duration_sec:.2f}s!"
            )
            if not interactive_logs:
                print(f"{test_tag} STDOUT:\n\n{result.stdout}\n\n")
                print(f"{test_tag} STDERR:\n\n{result.stderr}\n\n")
            assert result.returncode == 0, (
                f"{test_tag} Tuning failed with error code: {result.returncode}"
                + ("" if interactive_logs else f"\nSTDERR:\n\n{result.stderr}\n")
            )

        # Validate output directory
        tune_output_dir = output_dir / "tune"
        assert tune_output_dir.exists(), f"{test_tag} Output directory was not created"
        assert tune_output_dir.is_dir(), (
            f"{test_tag} Output directory is not a directory"
        )

        # Check for tuning results (best hyperparameters, trial history, etc.)
        # This will depend on your tuner implementation
        # Example checks:
        # - best_params.json or similar
        # - trial_history.json or similar
        # - final trained model from best trial

    except Exception as e:
        duration_sec = time.perf_counter() - _START_TIME
        print(f"{test_tag} Test failed: {str(e)}")
        print(f"{test_tag} Test duration: {duration_sec:.2f}s")
        print(f"{test_tag} Test artifacts can be found in: {output_dir}")
        raise

    if cleanup_output_dir_on_success:
        print(f"{test_tag} Cleaning up output dir on success: '{output_dir}'...")
        shutil.rmtree(output_dir)


@requires_gpus(count=1, min_gb=6.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TuneTestConfig(
            test_name="tune_smollm_135m_basic",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "smollm"
                / "tuning"
                / "135m"
                / "tune.yaml"  # Your tuning config
            ),
            n_trials=2,
            max_steps=10,
            model_max_length=128,
        ),
    ],
    ids=get_tune_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.single_gpu
def test_tune_1gpu_6gb(
    test_config: TuneTestConfig,
    tmp_path: Path,
):
    _test_tune_impl(test_config=test_config, tmp_path=tmp_path)
