import json

from oumi.core.registry import REGISTRY


######################## Creating a conversation JSON ########################
def save_conversations_for_dataset(
    output_path: str, dataset_name="yahma/alpaca-cleaned", num_samples: int = 10
):
    """Save the conversations to a file.

    Args:
        output_path (str): The path to save the conversations.
        dataset_name (str): The name of the dataset to use.
        num_samples (int): The number of samples to save.
    """
    with open(output_path, "w") as f:
        dataset_cls = REGISTRY.get_dataset(dataset_name)
        if not dataset_cls:
            raise ValueError("Dataset not found.")
        dataset = dataset_cls()
        for idx in range(num_samples):
            conversation = dataset.conversation(idx)
            # Let's remove the last message from the conversation
            conversation.messages.pop()
            f.write(conversation.to_json() + "\n")


######################## Comparing Results ########################
def compare_predictions(prediction_files: list[str], idx: int):
    """Compares the predictions from different files at a given index.

    Args:
        prediction_files (list[str]): The paths to the prediction files.
        idx (int): The index of the prediction to compare.
    """
    for target_file in prediction_files:
        input(f"Press Enter to load results from {target_file}")
        convo = json.loads(_read_line(target_file, idx))
        print("Model input:")
        for msg in convo["messages"][:-1]:
            print(f"(New Message. ROLE: {msg['role']})\n(CONTENT:) {msg['content']}")
        print("\n\n")
        print("Model response:")
        print(convo["messages"][-1]["content"])
        print("---\n\n")


def _read_line(file_path: str, idx: int) -> str:
    """Reads a single line from a file as a raw string."""
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return line
    raise RuntimeError(f"Index out of range: {idx}")


######################## Batch Prediction ########################
import time  # noqa: E402

from oumi.core.configs import InferenceConfig, ModelParams, RemoteParams  # noqa: E402
from oumi.core.types.conversation import Conversation  # noqa: E402
from oumi.inference import OpenAIInferenceEngine  # noqa: E402


def _read_convos(file_path: str) -> list[Conversation]:
    with open(file_path) as f:
        return [Conversation.from_json(convo.strip()) for convo in f]


def run_batch(file_path: str) -> str:
    """Runs batch prediction on the conversations in `file_path`.

    Args:
        file_path (str): the path to a jsonl file of conversations.

    Returns:
        str: the id of the batch job.
    """
    config = InferenceConfig(
        model=ModelParams(model_name="gpt-4o-mini"),
        remote_params=RemoteParams(
            batch_completion_window="24h"  # Time window for processing
        ),
    )

    engine = OpenAIInferenceEngine(
        model_params=config.model, remote_params=config.remote_params
    )

    # Prepare batch of conversations
    conversations = _read_convos(file_path)
    # The only difference with online inference is using `infer_batch` instead of
    # `infer`
    return engine.infer_batch(conversations, config)


def poll_status(id: str, file_path: str):
    """Polls the batch status every 5 seconds and prints the result.

    Args:
        id: The ID for the batch job.
        file_path: The original file submitted for batch processing.
    """
    config = InferenceConfig(
        model=ModelParams(model_name="gpt-4o-mini"),
    )
    engine = OpenAIInferenceEngine(
        model_params=config.model, remote_params=config.remote_params
    )
    status = engine.get_batch_status(id)
    while not status.is_terminal:
        print(f"Completion percentage: {status.completion_percentage}%")
        print(f"Has errors: {status.has_errors}")
        time.sleep(5)
        status = engine.get_batch_status(id)
    print("Job complete!")
    print(f"Status: {status.status}")
    conversations = _read_convos(file_path)
    results = engine.get_batch_results(id, conversations)
    for conv in results:
        print(f"Question: {conv.messages[-2].content}")  # User message
        print(f"Answer: {conv.messages[-1].content}")  # Assistant response
        print()
