import json
import argparse
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.configs import InferenceConfig
from oumi.inference import VLLMInferenceEngine

def main(args):

    if "jsonl" in args.input_file:
        data = []
        with open(args.input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = json.load(open(args.input_file))

    processed_data = []
    for d in data[:args.num_samples if args.num_samples else len(data)]:
        output = Conversation(
                messages=[
                    Message(
                        role=Role.USER,
                        content=d["content"]["request"]
                    ),
                ]
            )
        processed_data.append(output)

    config = InferenceConfig.from_yaml(str(args.inference_config))

    engine = VLLMInferenceEngine(
                model_params=config.model,
                generation_params=config.generation,
                gpu_memory_utilization=0.95,
                max_num_seqs=10,
            )

    results = engine.infer(input=processed_data, inference_config=config)
    # make each a dict
    results = [result.to_dict() for result in results]

    # save as json
    results = json.dump(results, open(args.output_file, "w"), indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--inference_config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    print(args)
    main(args)
