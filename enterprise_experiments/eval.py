import argparse
import json

from oumi.judges.simple_judge import SimpleJudge


def main(args):
    if "jsonl" in args.input_file:
        data = []
        with open(args.input_file) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = json.load(open(args.input_file))

    processed_data = []
    if args.chat_format:
        for d in data[: args.num_samples if args.num_samples else len(data)]:
            output = {
                "request": d["messages"][0]["content"],
                "response": d["messages"][1]["content"],
            }
            processed_data.append(output)
    else:
        processed_data = data[: args.num_samples if args.num_samples else len(data)]

    judge = SimpleJudge(args.judge_config)
    outputs = judge.judge(processed_data)

    save_outputs = []
    for input, output in zip(processed_data, outputs):
        request = input["request"]
        response = input["response"]
        judgment = output.field_values["judgment"]
        explanation = output.field_values["explanation"]
        save_outputs.append(
            {
                "request": request,
                "response": response,
                "judgment": judgment,
                "explanation": explanation,
            }
        )

    # save as json
    json.dump(save_outputs, open(args.output_file, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--judge_config", type=str, required=True)
    parser.add_argument("--chat_format", action="store_false")
    args = parser.parse_args()

    print(args)
    main(args)
