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
from datetime import datetime
from pprint import pformat
from typing import Any

try:
    import alpaca_eval  # pyright: ignore[reportMissingImports]
    import alpaca_eval.evaluate  # pyright: ignore[reportMissingImports]
except ImportError:
    alpaca_eval = None

import pandas as pd

from oumi.core.configs import (
    AlpacaEvalTaskParams,
    EvaluationConfig,
)
from oumi.core.distributed import is_world_process_zero
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.inference import BaseInferenceEngine
from oumi.datasets.evaluation import AlpacaEvalDataset, utils
from oumi.utils.logging import logger

ALPACA_EVAL_TASK_NAME = "alpaca_eval"


def evaluate(
    task_params: AlpacaEvalTaskParams,
    config: EvaluationConfig,
    inference_engine: BaseInferenceEngine,
) -> EvaluationResult:
    """Evaluates a model using the Alpaca Eval framework.

    For detailed documentation on the AlpacaEval framework, we refer you to the
    following readme: https://github.com/tatsu-lab/alpaca_eval.

    Args:
        task_params: The AlpacaEval parameters to use for evaluation.
        config: The desired configuration for evaluation.
        inference_engine: The inference engine to use for generating responses.

    Returns:
        The evaluation result (including metrics and their values).
    """
    # Prerequisites
    if not alpaca_eval:
        raise RuntimeError(
            "The `alpaca_eval` package is NOT installed. Please either install all "
            "evaluation dependencies with `pip install oumi[evaluation]` or directly "
            "install the missing package with `pip install alpaca_eval`."
        )

    open_ai_key = os.environ.get("OPENAI_API_KEY")
    if not open_ai_key:
        logger.warning(
            "`OPENAI_API_KEY` environment variable is NOT set. If you are using an "
            "OpenAI model as an annotator (judge), the execution will fail."
        )

    # Set the annotators config and metric function based on the version.
    if task_params.version == 1.0:
        os.environ["IS_ALPACA_EVAL_2"] = str(False)
        annotators_config = "alpaca_eval_gpt4"
        fn_metric = "get_winrate"
        sort_by_metric = "win_rate"
    elif task_params.version == 2.0:
        os.environ["IS_ALPACA_EVAL_2"] = str(True)
        annotators_config = "weighted_alpaca_eval_gpt4_turbo"
        fn_metric = "get_length_controlled_winrate"
        sort_by_metric = "length_controlled_winrate"
    else:
        raise ValueError(
            "The `version` field in `AlpacaEvalTaskParams` must be either 1.0 or 2.0."
        )

    # Get a timestamp for the current run.
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the evaluation dataset.
    logger.info("Loading the `tatsu-lab/alpaca_eval` dataset.")
    alpaca_dataset = AlpacaEvalDataset(
        dataset_name="tatsu-lab/alpaca_eval"
    ).conversations()

    if task_params.num_samples:
        alpaca_dataset = alpaca_dataset[: task_params.num_samples]

    # Run inference for the alpaca_dataset.
    logger.info(
        "\tAlpacaEval inference `model_params`:\n"
        f"{pformat(inference_engine._model_params)}\n"
        "\tAlpacaEval inference `generation_params`:\n"
        f"{pformat(inference_engine._generation_params)}"
    )
    responses = inference_engine.infer(input=alpaca_dataset)

    # Convert the model responses from Oumi format to Alpaca format.
    generator_display_name = config.run_name or start_time_str  # No run name? use time.
    responses_json = utils.conversations_to_alpaca_format(responses)
    responses_df = pd.DataFrame(responses_json)
    responses_df["generator"] = generator_display_name

    # Run AlpacaEval evaluation, i.e. annotate the model's responses.
    logger.info("Running AlpacaEval annotation.")
    logger.info(f"\tAlpacaEval `task_params`:\n{pformat(task_params)}")
    df_leaderboard, _ = alpaca_eval.evaluate(
        model_outputs=responses_df,
        annotators_config=annotators_config,
        fn_metric=fn_metric,
        is_return_instead_of_print=True,
        is_overwrite_leaderboard=True,
        max_instances=task_params.num_samples,
        sort_by=sort_by_metric,
        **task_params.eval_kwargs,
    )

    # Metrics are only available on the main process, and `None` on others.
    if not is_world_process_zero():
        return EvaluationResult()

    metric_dict = {}
    if df_leaderboard is not None:
        if generator_display_name in df_leaderboard.index:
            metrics = df_leaderboard.loc[generator_display_name]
            metric_dict: dict[str, Any] = {
                str(metric): value for metric, value in metrics.items()
            }
            logger.info(f"AlpacaEval's metric dict is {pformat(metric_dict)}.")
        else:
            logger.error("AlpacaEval results not found in leaderboard.")
    else:
        logger.error("The `alpaca_eval` API did not return a leaderboard.")

    backend_task_config = {
        "IS_ALPACA_EVAL_2": os.environ.get("IS_ALPACA_EVAL_2", "None"),
        "annotators_config": annotators_config,
        "fn_metric": fn_metric,
        "max_instances": task_params.num_samples,
        "other_params": task_params.eval_kwargs,
    }

    if task_params.log_samples:
        backend_task_config["model_outputs"] = responses_json

    return EvaluationResult(
        task_name=task_params.task_name,
        task_result={"results": {ALPACA_EVAL_TASK_NAME: metric_dict}},
        backend_config=backend_task_config,
    )
