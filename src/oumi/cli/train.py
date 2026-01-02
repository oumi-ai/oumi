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

from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_train_config
from oumi.utils.logging import logger

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.TRAIN, "Available Training Configs", "train"
)


def train(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name (e.g. llama3.1-8b-sft).",
            rich_help_panel="Options",
            autocompletion=complete_train_config,
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available training configs.",
            callback=_list_configs_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    level: Annotated[
        cli_utils.LogLevel | None,
        typer.Option(
            "--log-level",
            "-log",
            help="Logging level.",
            show_default=False,
            show_choices=True,
            case_sensitive=False,
            callback=cli_utils.set_log_level,
            rich_help_panel="Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
            rich_help_panel="Options",
        ),
    ] = False,
    # Model overrides
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model.model_name",
            help="Model name or HuggingFace path.",
            rich_help_panel="Model",
        ),
    ] = None,
    model_max_length: Annotated[
        int | None,
        typer.Option(
            "--model.model_max_length",
            help="Maximum sequence length.",
            rich_help_panel="Model",
        ),
    ] = None,
    torch_dtype: Annotated[
        str | None,
        typer.Option(
            "--model.torch_dtype_str",
            help="Torch dtype (float16, bfloat16, float32).",
            rich_help_panel="Model",
        ),
    ] = None,
    trust_remote_code: Annotated[
        bool | None,
        typer.Option(
            "--model.trust_remote_code/--no-model.trust_remote_code",
            help="Trust remote code from HuggingFace.",
            rich_help_panel="Model",
        ),
    ] = None,
    # Training overrides
    learning_rate: Annotated[
        float | None,
        typer.Option(
            "--training.learning_rate",
            help="Learning rate.",
            rich_help_panel="Training",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--training.per_device_train_batch_size",
            help="Batch size per device.",
            rich_help_panel="Training",
        ),
    ] = None,
    num_epochs: Annotated[
        int | None,
        typer.Option(
            "--training.num_train_epochs",
            help="Number of training epochs.",
            rich_help_panel="Training",
        ),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--training.output_dir",
            help="Output directory for checkpoints.",
            rich_help_panel="Training",
        ),
    ] = None,
    save_steps: Annotated[
        int | None,
        typer.Option(
            "--training.save_steps",
            help="Save checkpoint every N steps.",
            rich_help_panel="Training",
        ),
    ] = None,
    # Data overrides
    dataset_name: Annotated[
        str | None,
        typer.Option(
            "--data.train.datasets[0].dataset_name",
            help="Training dataset name.",
            rich_help_panel="Data",
        ),
    ] = None,
):
    """Train a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        list_configs: List all available training configs.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        model_name: Model name or HuggingFace path.
        model_max_length: Maximum sequence length.
        torch_dtype: Torch dtype string.
        trust_remote_code: Trust remote code from HuggingFace.
        learning_rate: Learning rate.
        batch_size: Batch size per device.
        num_epochs: Number of training epochs.
        output_dir: Output directory for checkpoints.
        save_steps: Save checkpoint every N steps.
        dataset_name: Training dataset name.
    """
    # Auto-collect overrides from dot-notation options (e.g., --model.model_name)
    option_overrides = cli_utils.collect_config_overrides(ctx)
    # Parse any additional extra args from command line
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    # Combine: explicit options take precedence (added last)
    all_overrides = extra_args + option_overrides

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.TRAIN),
        )
    )
    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import train as oumi_train
        from oumi.core.configs import TrainingConfig
        from oumi.core.distributed import set_random_seeds
        from oumi.utils.torch_utils import (
            device_cleanup,
            limit_per_process_memory,
        )
        # End imports

    cli_utils.configure_common_env_vars()

    parsed_config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        config, all_overrides, logger=logger
    )
    parsed_config.finalize_and_validate()

    from oumi.telemetry import TelemetryManager

    TelemetryManager.get_instance().tags(
        model_name=parsed_config.model.model_name,
        trainer_type=parsed_config.training.trainer_type.value,
        use_peft=parsed_config.training.use_peft,
        q_lora=parsed_config.peft.q_lora,
        fsdp=parsed_config.fsdp.enable_fsdp,
        deepspeed=parsed_config.deepspeed.enable_deepspeed,
    )

    limit_per_process_memory()
    device_cleanup()
    set_random_seeds(
        parsed_config.training.seed, parsed_config.training.use_deterministic
    )

    # Run training
    oumi_train(parsed_config, verbose=verbose)

    device_cleanup()
