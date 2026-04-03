import argparse
import json

import dotenv

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.types.conversation import Conversation, Message, Role

dotenv.load_dotenv()


def main(args):
    """Run inference on input data and save results to output file."""
    if "jsonl" in args.input_file:
        data = []
        with open(args.input_file) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = json.load(open(args.input_file))

    processed_data = []
    for d in data[: args.num_samples if args.num_samples else len(data)]:
        output = Conversation(
            messages=[
                Message(
                    role=Role(m["role"]),
                    content=m["content"],
                )
                for m in d["messages"]
            ],
            metadata={k: v for k, v in d.items() if k != "messages"},
        )
        processed_data.append(output)

    config = InferenceConfig.from_yaml(str(args.inference_config))

    engine = build_inference_engine(
        engine_type=config.engine or InferenceEngineType.VLLM,
        model_params=config.model,
        remote_params=config.remote_params,
        generation_params=config.generation,
    )

    results = engine.infer(input=processed_data, inference_config=config)
    # make each a dict
    results = [result.to_dict() for result in results]

    # save as json
    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--inference_config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    print(args)
    main(args)
