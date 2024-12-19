from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import (
    EvaluationPlatform,
    LMHarnessParams,
)
from oumi.evaluation.lm_harness import evaluate_lm_harness
from oumi.evaluation.lm_harness_task_groups import TASK_GROUPS


def evaluate(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    if not config.tasks:
        return
    for task in config.tasks:
        if task.evaluation_platform == EvaluationPlatform.LM_HARNESS:
            if not isinstance(task, LMHarnessParams):
                raise ValueError(f"Expected LMHarnessParams, but got: {type(task)}")
            if task in TASK_GROUPS:
                # If this task is an Oumi-defined list of tasks (such as a leaderboard),
                # then execute each sub-task of the group (serially).
                task_group = TASK_GROUPS[task]
                for sub_task in task_group:
                    evaluate_lm_harness(
                        model_params=config.model,
                        lm_harness_params=sub_task,
                        generation_params=config.generation,
                        output_dir=config.output_dir,
                        enable_wandb=config.enable_wandb,
                        run_name=config.run_name,
                    )
            else:
                evaluate_lm_harness(
                    model_params=config.model,
                    lm_harness_params=task,
                    generation_params=config.generation,
                    output_dir=config.output_dir,
                    enable_wandb=config.enable_wandb,
                    run_name=config.run_name,
                )
        elif task.evaluation_platform == EvaluationPlatform.ALPACA_EVAL:
            raise NotImplementedError("Alpaca Eval is not yet supported.")
        else:
            raise ValueError(f"Unknown evaluation platform: {task.evaluation_platform}")
