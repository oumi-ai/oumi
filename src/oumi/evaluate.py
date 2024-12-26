from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import (
    AlpacaEvalTaskParams,
    EvaluationPlatform,
    LMHarnessTaskParams,
)
from oumi.evaluation.alpaca_eval import evaluate as evaluate_alpaca_eval
from oumi.evaluation.lm_harness import evaluate as evaluate_lm_harness


def evaluate(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    for task in config.tasks:
        if task.get_evaluation_platform() == EvaluationPlatform.LM_HARNESS:
            lm_harness_task_params = task.get_evaluation_platform_task_params()
            assert isinstance(lm_harness_task_params, LMHarnessTaskParams)
            evaluate_lm_harness(
                lm_harness_task_params=lm_harness_task_params,
                model_params=config.model,
                generation_params=config.generation,
                output_dir=config.output_dir,
                enable_wandb=config.enable_wandb,
                run_name=config.run_name,
            )
        elif task.get_evaluation_platform() == EvaluationPlatform.ALPACA_EVAL:
            alpaca_eval_task_params = task.get_evaluation_platform_task_params()
            assert isinstance(alpaca_eval_task_params, AlpacaEvalTaskParams)
            evaluate_alpaca_eval(
                alpaca_eval_task_params=alpaca_eval_task_params,
                model_params=config.model,
                generation_params=config.generation,
                inference_engine_type=config.inference_engine,  # type: ignore
                inference_remote_params=config.inference_remote_params,  # type: ignore
                output_dir=config.output_dir,
                run_name=config.run_name,
            )
        else:
            raise ValueError("Unknown evaluation platform")
