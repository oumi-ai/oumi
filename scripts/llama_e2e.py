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
        "configs/oumi/llama1b.sft.yaml"
    )
    train_config.training.enable_wandb = False
    train_config.training.max_steps = 100
    train_config.training.output_dir = model_output_dir
    train_config.validate()
    train(train_config)

    device_cleanup()
    eval_config: EvaluationConfig = EvaluationConfig.from_yaml(
        "configs/oumi/llama1b.eval.yaml"
    )
    eval_config.model.model_name = model_output_dir
    eval_config.validate()
    evaluate(eval_config)

    device_cleanup()
    infer_config: InferenceConfig = InferenceConfig.from_yaml(
        "configs/oumi/llama1b.infer.yaml"
    )
    infer_config.model.model_name = model_output_dir
    infer_config.validate()
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
