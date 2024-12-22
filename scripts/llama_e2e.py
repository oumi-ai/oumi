from oumi.core.configs import EvaluationConfig, InferenceConfig, TrainingConfig
from oumi.evaluate import evaluate
from oumi.infer import infer
from oumi.train import train
from oumi.utils.torch_utils import device_cleanup


def main() -> None:
    """Run Llama 1B train/eval/infer."""
    model_output_dir = "output/llama1b_e2e"
    device_cleanup()
    train_config: TrainingConfig = TrainingConfig.from_yaml(
        "configs/recipes/llama3_2/sft/1b_full/train.yaml"
    )
    train_config.training.enable_wandb = False
    train_config.training.max_steps = 100
    train_config.training.output_dir = model_output_dir
    train_config.finalize_and_validate()
    train(train_config)

    device_cleanup()
    eval_config: EvaluationConfig = EvaluationConfig.from_yaml(
        "configs/recipes/llama3_2/evaluation/1b_eval.yaml"
    )
    eval_config.model.model_name = model_output_dir
    eval_config.finalize_and_validate()
    evaluate(eval_config)

    device_cleanup()
    infer_config: InferenceConfig = InferenceConfig.from_yaml(
        "configs/recipes/llama3_2/inference/1b_infer.yaml"
    )
    infer_config.model.model_name = model_output_dir
    infer_config.finalize_and_validate()
    model_responses = infer(
        config=infer_config,
        inputs=[
            "Foo",
            "Bar",
        ],
    )
    print(model_responses)

    device_cleanup()


if __name__ == "__main__":
    main()
