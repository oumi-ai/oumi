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

import os
import signal
import sys
import traceback

import typer

from oumi.cli.cache import card as cache_card
from oumi.cli.cache import get as cache_get
from oumi.cli.cache import ls as cache_ls
from oumi.cli.cache import rm as cache_rm
from oumi.cli.cli_utils import (
    CONSOLE,
    CONTEXT_ALLOW_EXTRA_ARGS,
    create_github_issue_url,
)
from oumi.cli.distributed_run import accelerate, torchrun
from oumi.cli.env import env
from oumi.cli.evaluate import evaluate
from oumi.cli.fetch import fetch
from oumi.cli.infer import infer
from oumi.cli.judge import judge_conversations_file, judge_dataset_file
from oumi.cli.launch import cancel, down, logs, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.cli.quantize import quantize
from oumi.cli.synth import synth
from oumi.cli.train import train
from oumi.cli.tune import tune
from oumi.utils.logging import should_use_rich_logging

_ASCII_LOGO = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|
"""


def experimental_features_enabled():
    """Check if experimental features are enabled."""
    is_enabled = os.environ.get("OUMI_ENABLE_EXPERIMENTAL_FEATURES", "False")
    return is_enabled.lower() in ("1", "true", "yes", "on")


def _oumi_welcome(ctx: typer.Context):
    if ctx.invoked_subcommand == "distributed":
        return
    # Skip logo for rank>0 for multi-GPU jobs to reduce noise in logs.
    if int(os.environ.get("RANK", 0)) > 0:
        return
    CONSOLE.print(_ASCII_LOGO, style="green", highlight=False)


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer(pretty_exceptions_enable=False)
    app.callback(context_settings={"help_option_names": ["-h", "--help"]})(
        _oumi_welcome
    )
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Evaluate a model.",
    )(evaluate)
    app.command()(env)
    app.command(  # Alias for evaluate
        name="eval",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Evaluate a model.",
    )(evaluate)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Run inference on a model.",
    )(infer)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Synthesize a dataset.",
    )(synth)
    app.command(  # Alias for synth
        name="synthesize",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="🚧 [Experimental] Synthesize a dataset.",
    )(synth)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Train a model.",
    )(train)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Tune the parameters for a model.",
    )(tune)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Quantize a model.",
    )(quantize)
    judge_app = typer.Typer(pretty_exceptions_enable=False)
    judge_app.command(name="dataset", context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        judge_dataset_file
    )
    judge_app.command(name="conversations", context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        judge_conversations_file
    )
    app.add_typer(judge_app, name="judge", help="Judge datasets or conversations.")

    launch_app = typer.Typer(pretty_exceptions_enable=False)
    launch_app.command(help="Cancels a job.")(cancel)
    launch_app.command(help="Turns down a cluster.")(down)
    launch_app.command(
        name="run", context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Runs a job."
    )(launcher_run)
    launch_app.command(help="Prints the status of jobs launched from Oumi.")(status)
    launch_app.command(help="Stops a cluster.")(stop)
    launch_app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Launches a job."
    )(up)
    launch_app.command(help="Prints the available clouds.")(which)
    launch_app.command(help="Gets the logs of a job.")(logs)
    app.add_typer(launch_app, name="launch", help="Launch jobs remotely.")

    distributed_app = typer.Typer(pretty_exceptions_enable=False)
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(accelerate)
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(torchrun)
    app.add_typer(
        distributed_app,
        name="distributed",
        help=(
            "A wrapper for torchrun/accelerate "
            "with reasonable default values for distributed training."
        ),
    )

    app.command(
        help="Fetch configuration files from the oumi GitHub repository.",
    )(fetch)

    cache_app = typer.Typer(pretty_exceptions_enable=False)
    cache_app.command(name="ls", help="List locally cached items.")(cache_ls)
    cache_app.command(name="get", help="Download a repository from Hugging Face.")(
        cache_get
    )
    cache_app.command(name="card", help="Show information for a repository.")(
        cache_card
    )
    cache_app.command(name="rm", help="Remove a repository from the local cache.")(
        cache_rm
    )
    app.add_typer(cache_app, name="cache", help="Manage local Hugging Face cache.")

    return app


# Store original signal handlers to restore if needed
# signal.signal() returns Callable | int | None (int for SIG_DFL=0, SIG_IGN=1)
_original_sigterm_handler = None
_original_sigint_handler = None


def _create_signal_handler(sig_name: str):
    """Create a signal handler that logs helpful OOM-related messages.

    Note: SIGKILL (signal 9) cannot be caught - this is a fundamental OS limitation.
    When the Linux OOM killer terminates a process, it sends SIGKILL which immediately
    kills the process without any chance to run cleanup code.

    However, when running distributed training with torchrun:
    - If one worker is killed by the OOM killer (SIGKILL), torchrun detects this
    - torchrun then sends SIGTERM to the remaining workers to shut them down
    - We can catch SIGTERM and provide helpful guidance to the user
    """

    def _signal_handler(signum: int, frame) -> None:
        """Handle termination signals with helpful OOM-related messages."""
        # Use print instead of logger since logging may not be safe in signal handlers
        print(f"\n{'=' * 70}")
        print(f"OUMI: Received {sig_name} (signal {signum})")
        print("=" * 70)
        print(
            "\nIf you see 'Signal 9 (SIGKILL)' in the error output, your process was "
            "likely terminated by the Linux OOM (Out-Of-Memory) killer."
        )
        print("\nCommon causes and solutions:")
        print("  1. GPU memory exhaustion:")
        print("     - Reduce batch size (training.per_device_train_batch_size)")
        print("     - Enable gradient checkpointing")
        print("       (model.enable_gradient_checkpointing)")
        print("     - Use a smaller model or enable model sharding (FSDP/DeepSpeed)")
        print("     - Reduce sequence length (data.max_length)")
        print("  2. CPU/System memory exhaustion:")
        print("     - Reduce dataloader workers")
        print("       (training.dataloader_num_workers)")
        print("     - Enable CPU offloading for FSDP (fsdp.cpu_offload)")
        print("     - Use memory-mapped datasets")
        print("  3. Check memory usage: nvidia-smi (GPU) or htop (CPU)")
        print("=" * 70 + "\n")
        sys.stdout.flush()

        # Re-raise the signal to allow normal termination
        # First restore the original handler to avoid infinite loop
        if sig_name == "SIGTERM" and _original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, _original_sigterm_handler)
        elif sig_name == "SIGINT" and _original_sigint_handler is not None:
            signal.signal(signal.SIGINT, _original_sigint_handler)

        # Re-send the signal to trigger default behavior
        os.kill(os.getpid(), signum)

    return _signal_handler


def _setup_signal_handlers() -> None:
    """Set up signal handlers to provide helpful messages on termination.

    This catches SIGTERM and SIGINT to provide guidance about potential OOM issues.
    SIGKILL cannot be caught (OS limitation), but we explain this in the message.
    """
    global _original_sigterm_handler, _original_sigint_handler

    # Only set up handlers for the main process in distributed training
    # to avoid duplicate messages from all workers
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Only set up signal handlers for rank 0 or non-distributed runs
    if rank == 0 or local_rank == 0:
        try:
            _original_sigterm_handler = signal.signal(
                signal.SIGTERM, _create_signal_handler("SIGTERM")
            )
            _original_sigint_handler = signal.signal(
                signal.SIGINT, _create_signal_handler("SIGINT")
            )
        except (ValueError, OSError):
            # signal.signal can fail if not called from the main thread
            # or on some platforms - silently ignore
            pass


def run():
    """The entrypoint for the CLI."""
    _setup_signal_handlers()
    app = get_app()
    try:
        return app()
    except Exception as e:
        tb_str = traceback.format_exc()
        CONSOLE.print(tb_str)
        issue_url = create_github_issue_url(e, tb_str)
        CONSOLE.print(
            "\n[red]If you believe this is a bug, please file an issue:[/red]"
        )
        if should_use_rich_logging():
            CONSOLE.print(
                f"📝 [yellow]Templated issue:[/yellow] "
                f"[link={issue_url}]Click here to report[/link]"
            )
        else:
            CONSOLE.print(
                "https://github.com/oumi-ai/oumi/issues/new?template=bug-report.yaml"
            )

        sys.exit(1)


if "sphinx" in sys.modules:
    # Create the CLI app when building the docs to auto-generate the CLI reference.
    app = get_app()
