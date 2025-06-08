import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

import pytest

from oumi.core.configs import EvaluationConfig
from oumi.utils.torch_utils import device_cleanup
from tests import get_configs_dir
from tests.e2e import get_e2e_test_output_dir
from tests.markers import requires_gpus


class EvalTestConfig(NamedTuple):
    test_name: str
    config_path: Path
    skip: bool = False
    interactive_logs: bool = True

    use_simple_oumi_evaluate_command: bool = False
    """
    If True, the test will use the simple `oumi evaluate` command instead of the
    distributed version. This sometimes leads to lower GPU RAM usage.
    """

    model_max_length: Optional[int] = None
    batch_size: Optional[int] = None
    num_samples: Optional[int] = 20  # Limit the number of samples by default
    num_fewshot: Optional[int] = None
    enable_wandb: Optional[bool] = False  # Disable `wandb`` by default


def get_eval_test_id_fn(val):
    assert isinstance(val, EvalTestConfig), f"{type(val)}: {val}"
    return val.test_name


def _test_eval_impl(
    test_config: EvalTestConfig,
    tmp_path: Path,
    *,
    cleanup_output_dir_on_success: bool = True,
    single_gpu: Optional[bool] = None,
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
        # Copy config file to output directory
        assert test_config.config_path.exists(), (
            f"{test_tag} Path doesn't exist: {test_config.config_path}"
        )
        assert test_config.config_path.is_file(), (
            f"{test_tag} Path is not a file: {test_config.config_path}"
        )

        # Verify the config is loadable
        try:
            eval_config = EvaluationConfig.from_yaml(test_config.config_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load eval config from: {test_config.config_path}"
            ) from e

        cmd: list[str] = []
        if test_config.use_simple_oumi_evaluate_command:
            cmd.append("oumi evaluate")
        else:
            cmd.append("oumi distributed accelerate launch -m oumi evaluate")

        config_path = test_config.config_path
        # Overriding nested fields using OmegaConf's dot-list syntax is complicated,
        # or impossible. Let's just create a modified config copy instead.
        if test_config.num_samples is not None or test_config.num_fewshot is not None:
            for task in eval_config.tasks:
                if test_config.num_samples is not None:
                    task.num_samples = test_config.num_samples
                if test_config.num_fewshot is not None:
                    task.eval_kwargs["num_fewshot"] = test_config.num_fewshot

            config_path = (
                output_dir / f"MODIFIED_{test_config.config_path.name}"
            ).resolve()
            eval_config.to_yaml(config_path)

        cmd.extend(
            [
                "-c",
                str(config_path),
                "--run_name",
                test_config.test_name,
            ]
        )

        for param_name, param_value in [
            ("enable_wandb", test_config.enable_wandb),
        ]:
            if param_value is not None:
                cmd.append(f"--{param_name}={str(param_value)}")

        for param_name, param_value in [
            ("model_max_length", test_config.model_max_length),
            ("shard_for_eval", False if single_gpu else None),
        ]:
            if param_value is not None:
                cmd.append(f"--model.{param_name}={str(param_value)}")

        for param_name, param_value in [
            ("batch_size", test_config.batch_size),
        ]:
            if param_value is not None:
                cmd.append(f"--generation.{param_name}={str(param_value)}")

        env_vars = dict(os.environ)
        if "TOKENIZERS_PARALLELISM" not in env_vars:
            # Resolves the warning: "Avoid using `tokenizers` before the fork ..."
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
            print(f"{test_tag} Eval job successfully finished in {duration_sec:.2f}s!")
        else:
            print(
                f"{test_tag} Eval job failed with error code: {result.returncode} "
                f"in {duration_sec:.2f}s!"
            )
            if not interactive_logs:
                print(f"{test_tag} STDOUT:\n\n{result.stdout}\n\n")
                print(f"{test_tag} STDERR:\n\n{result.stderr}\n\n")
            assert result.returncode == 0, (
                f"{test_tag} Training failed with error code: {result.returncode}"
                + ("" if interactive_logs else f"\nSTDERR:\n\n{result.stderr}\n")
            )

    except Exception as e:
        duration_sec = time.perf_counter() - _START_TIME
        print(f"{test_tag} Test failed: {str(e)}")
        print(f"{test_tag} Test duration: {duration_sec:.2f}s")
        print(f"{test_tag} Test artifacts can be found in: {output_dir}")
        raise

    if cleanup_output_dir_on_success:
        # Clean-up temp data to stay under disk quota.
        print(f"{test_tag} Cleaning up output dir on success: '{output_dir}'...")
        shutil.rmtree(output_dir)


@requires_gpus(count=1, min_gb=24.0)
@pytest.mark.parametrize(
    "test_config",
    [
        EvalTestConfig(
            test_name="eval_text_smollm_135m_single_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "smollm"
                / "evaluation"
                / "135m"
                / "eval.yaml"
            ),
            num_samples=20,
        ),
        EvalTestConfig(
            test_name="eval_text_llama32_1b_single_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "llama3_2"
                / "evaluation"
                / "1b_eval.yaml"
            ),
            num_samples=20,
        ),
        EvalTestConfig(
            test_name="eval_text_phi3_single_gpu",
            config_path=(
                get_configs_dir() / "recipes" / "phi3" / "evaluation" / "eval.yaml"
            ),
            num_samples=10,
            use_simple_oumi_evaluate_command=True,
        ),
        EvalTestConfig(
            test_name="eval_text_llama32_3b_single_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "llama3_2"
                / "evaluation"
                / "3b_eval.yaml"
            ),
            num_samples=20,
        ),
        EvalTestConfig(
            test_name="eval_text_llama31_8b_single_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "llama3_1"
                / "evaluation"
                / "8b_eval.yaml"
            ),
            num_samples=20,
        ),
    ],
    ids=get_eval_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.single_gpu
def test_eval_text_1gpu_24gb(test_config: EvalTestConfig, tmp_path: Path):
    _test_eval_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        single_gpu=True,
    )


@requires_gpus(count=1, min_gb=24.0)
@pytest.mark.parametrize(
    "test_config",
    [
        EvalTestConfig(
            test_name="eval_mm_llama32v_11b_single_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "llama3_2_vision"
                / "evaluation"
                / "11b_eval.yaml"
            ),
            num_samples=2,  # The actual number is ~30X (30 sub-tasks)
            num_fewshot=3,
        ),
    ],
    ids=get_eval_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.single_gpu
def test_eval_multimodal_1gpu_24gb(test_config: EvalTestConfig, tmp_path: Path):
    _test_eval_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        single_gpu=True,
    )


@requires_gpus(count=4, min_gb=39.0)
@pytest.mark.parametrize(
    "test_config",
    [
        EvalTestConfig(
            test_name="eval_text_llama31_70b_multi_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "llama3_1"
                / "evaluation"
                / "70b_eval.yaml"
            ),
            num_samples=20,
            use_simple_oumi_evaluate_command=True,
        ),
        EvalTestConfig(
            test_name="eval_text_deepseek_r1_distill_llama8b_multi_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "deepseek_r1"
                / "evaluation"
                / "distill_llama_8b"
                / "eval.yaml"
            ),
            num_samples=20,
            use_simple_oumi_evaluate_command=True,
        ),
        EvalTestConfig(
            test_name="eval_text_deepseek_r1_distill_llama70b_multi_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "deepseek_r1"
                / "evaluation"
                / "distill_llama_70b"
                / "eval.yaml"
            ),
            num_samples=20,
            use_simple_oumi_evaluate_command=True,
        ),
    ],
    ids=get_eval_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.multi_gpu
def test_eval_text_4gpu_40gb(test_config: EvalTestConfig, tmp_path: Path):
    _test_eval_impl(
        test_config=test_config,
        tmp_path=tmp_path,
    )


@requires_gpus(count=4, min_gb=24.0)
@pytest.mark.parametrize(
    "test_config",
    [
        EvalTestConfig(
            test_name="eval_mm_llama32v_11b_multi_gpu",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "llama3_2_vision"
                / "evaluation"
                / "11b_eval.yaml"
            ),
            num_samples=8,  # The actual number is ~30X (30 sub-tasks)
        ),
    ],
    ids=get_eval_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.multi_gpu
def test_eval_multimodal_4gpu_24gb(test_config: EvalTestConfig, tmp_path: Path):
    _test_eval_impl(
        test_config=test_config,
        tmp_path=tmp_path,
    )
