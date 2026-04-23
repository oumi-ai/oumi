"""Run inference on the filtered Hermes tool call dataset.

Loads the filtered dataset (from filter_dataset.py), runs inference using an
Oumi inference engine, and saves predictions as JSONL. The output can then be
evaluated with eval_tool_calls.py.

Usage:
    # With a YAML inference config:
    python run_inference.py \
        --input_file data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
        --output_file output/llama3.1_8b_preds.jsonl \
        --inference_config configs/llama3.1_8b_instruct.yaml

    # With overrides:
    python run_inference.py \
        --input_file data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
        --output_file output/preds.jsonl \
        --inference_config configs/llama3.1_8b_instruct.yaml \
        --num_samples 100

    # Then evaluate:
    python eval_tool_calls.py \
        --dataset_path data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
        --predictions_path output/llama3.1_8b_preds.jsonl
"""

import argparse
import json
from pathlib import Path

import dotenv

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.types.conversation import Conversation, Message, Role

dotenv.load_dotenv()


def load_conversations(path: str, num_samples: int | None = None) -> list[Conversation]:
    """Load filtered dataset as Oumi Conversations."""
    conversations = []
    with open(path) as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            record = json.loads(line)
            messages = [
                Message(role=Role(m["role"]), content=m["content"])
                for m in record["messages"]
            ]
            conversations.append(
                Conversation(
                    messages=messages,
                    metadata=record.get("metadata", {}),
                    conversation_id=record.get("conversation_id"),
                )
            )
    return conversations


def save_predictions(results: list[Conversation], output_path: str):
    """Save inference results as JSONL.

    Each line has a 'content' field (the model's response) for eval_tool_calls.py,
    plus the full conversation for debugging.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for conv in results:
            last_msg = conv.last_message()
            content = last_msg.content if last_msg else ""
            record = {
                "content": content,
                "conversation_id": conv.conversation_id,
                "metadata": conv.metadata,
                "messages": [
                    {"role": m.role.value, "content": m.content} for m in conv.messages
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference for tool call eval")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--inference_config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    print(args)

    # Load data
    conversations = load_conversations(args.input_file, args.num_samples)
    print(f"Loaded {len(conversations)} conversations")

    # Build inference engine
    config = InferenceConfig.from_yaml(args.inference_config)
    engine = build_inference_engine(
        engine_type=config.engine or InferenceEngineType.VLLM,
        model_params=config.model,
        remote_params=config.remote_params,
        generation_params=config.generation,
    )

    # Run inference
    results = engine.infer(input=conversations, inference_config=config)
    print(f"Got {len(results)} results")

    # Save
    save_predictions(results, args.output_file)
    print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()
