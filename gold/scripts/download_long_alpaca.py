"""Download Yukang/LongAlpaca-16k-length and convert to oumi chat JSONL format."""

import json

from datasets import load_dataset


def main():
    ds = load_dataset("Yukang/LongAlpaca-16k-length", split="train")
    print(f"Total examples: {len(ds)}")

    output_path = "/data/shanghong/oumi/gold/data/long_alpaca_16k.jsonl"
    with open(output_path, "w") as f:
        for ex in ds:
            messages = [
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["output"]},
            ]
            f.write(json.dumps({"messages": messages}) + "\n")

    print(f"Written {len(ds)} examples to {output_path}")


if __name__ == "__main__":
    main()
