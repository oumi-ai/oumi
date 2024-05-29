from lema.core.types import DataParams
from lema.core.types import EvaluationConfig
from lema.core.types import ModelParams


def main():
    """Main entry point for evaluating LeMa."""

    # TODO: Implement config/CLI-arguments parsing.
    config: EvaluationConfig = EvaluationConfig(data=DataParams(), model=ModelParams())
    evaluate(config)


def evaluate(config: EvaluationConfig) -> None:
    """Evaluate a model using the provided configuration."""
    raise NotImplementedError("Model evaluation is is not implemented yet")


if __name__ == "__main__":
    main()
