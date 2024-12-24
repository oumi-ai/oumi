from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import (
    AlpacaEvalTaskParams,
    EvaluationPlatform,
    LMHarnessTaskParams,
    evaluation_task_params_factory,
)
from oumi.evaluation.lm_harness import evaluate_lm_harness


def evaluate(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    for task in config.tasks:
        task_params = evaluation_task_params_factory(task)
        if task_params.evaluation_platform == EvaluationPlatform.LM_HARNESS.value:
            assert isinstance(task_params, LMHarnessTaskParams)
            evaluate_lm_harness(
                model_params=config.model,
                lm_harness_params=task_params,
                generation_params=config.generation,
                output_dir=config.output_dir,
                enable_wandb=config.enable_wandb,
                run_name=config.run_name,
            )
        elif task_params.evaluation_platform == EvaluationPlatform.ALPACA_EVAL.value:
            assert isinstance(task_params, AlpacaEvalTaskParams)
            raise NotImplementedError("Alpaca Eval is not yet supported.")
        else:
            raise ValueError(
                f"Unknown evaluation platform: {task_params.evaluation_platform}"
            )
