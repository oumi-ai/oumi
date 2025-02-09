import os

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types import Conversation, Message, Role
from oumi.inference import SambanovaInferenceEngine


def load_config(config_path):
    """Load the inference configuration from a YAML file."""
    return InferenceConfig.from_yaml(config_path)


def initialize_engine(api_key, model_name):
    """Initialize the SambaNova inference engine."""
    return SambanovaInferenceEngine(
        model_params=ModelParams(model_name=model_name),
        generation_params=GenerationParams(
            batch_size=1,
        ),
        remote_params=RemoteParams(
            api_key=api_key,
        ),
    )


def create_conversation():
    """Create a conversation for inference."""
    return [
        Conversation(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content="Answer the question in a couple sentences.",
                ),
                Message(
                    role=Role.USER, content="What is the strength of SambaNova Systems?"
                ),
            ]
        ),
    ]


def perform_inference(engine, conversations, config):
    """Perform inference using the SambaNova engine."""
    try:
        generations = engine.infer(
            input=conversations,
            inference_config=config,
        )
        return generations
    except Exception as e:
        print("An error occurred during inference:", str(e))
        return None


def main():
    # Set the path to the configuration file
    config_path = os.path.join(
        os.path.dirname(__file__), "sambanova_infer_tutorial.yaml"
    )

    # Load the configuration
    config = load_config(config_path)

    # Initialize the engine
    api_key = os.getenv("SAMBANOVA_API_KEY")
    model_name = "Meta-Llama-3.1-405B-Instruct"
    engine = initialize_engine(api_key, model_name)

    # Create the conversation
    conversations = create_conversation()

    # Perform inference
    generations = perform_inference(engine, conversations, config)

    # Print the results
    if generations:
        print(generations)


if __name__ == "__main__":
    main()
