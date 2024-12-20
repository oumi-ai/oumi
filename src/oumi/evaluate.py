from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.evaluation.lm_harness import evaluate_lm_harness


def evaluate(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    for task in config.tasks:
        if task.evaluation_platform() == EvaluationPlatform.LM_HARNESS:
            assert task.lm_harness_task_params
            evaluate_lm_harness(
                model_params=config.model,
                lm_harness_params=task.lm_harness_task_params,
                generation_params=config.generation,
                output_dir=config.output_dir,
                enable_wandb=config.enable_wandb,
                run_name=config.run_name,
            )
        elif task.evaluation_platform == EvaluationPlatform.ALPACA_EVAL:
            raise NotImplementedError("Alpaca Eval is not yet supported.")
        else:
            raise ValueError(f"Unknown evaluation platform: {task.evaluation_platform}")
