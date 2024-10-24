import csv
from typing import Any

import pandas as pd
from datasets import load_dataset
from llama_cpp import Llama, llama_chat_format, llama_types
from tqdm import tqdm

# Note: REQUIRES INSTALLING llama-cpp-python:
# https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation
# Also note that you may need a different command to install for Nvidia GPUs:

# CMAKE_ARGS="-DLLAMA_CUDA=on"
# pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir


@llama_chat_format.register_chat_format("llama-3-modified")
def _format_llama3(
    messages: list[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> llama_chat_format.ChatFormatterResponse:
    _roles = dict(
        system="<|start_header_id|>system<|end_header_id|>\n\n",
        user="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    # Underlying code already adds this token, so this would only add a 2nd one.
    # _begin_token = "<|begin_of_text|>"
    _sep = "<|eot_id|>"
    _messages = llama_chat_format._map_roles(messages, _roles)
    _messages.append((_roles["assistant"], None))
    _prompt = llama_chat_format._format_no_colon_single("", _messages, _sep)
    return llama_chat_format.ChatFormatterResponse(prompt=_prompt, stop=_sep)


def _get_response(llm, messages, system_instruction=""):
    new_messages = messages
    if system_instruction:
        new_messages = [{"role": "system", "content": system_instruction}] + messages
    output = llm.create_chat_completion(new_messages)
    response = output["choices"][0]["message"]["content"]
    return response


def main() -> None:
    """Runs inference on specified model using llama-cpp-python library."""
    REPO_ID = "lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF"
    FILENAME = "*Q1_M.gguf"
    CONTEXT_WINDOW_SIZE = 8192
    llm = Llama.from_pretrained(
        repo_id=REPO_ID,
        filename=FILENAME,
        n_gpu_layers=-1,
        seed=1337,
        n_ctx=CONTEXT_WINDOW_SIZE,
        verbose=False,
    )

    # Overwrite chat formatter to ensure it gets reset.
    llm.chat_format = "llama-3-modified"
    llm.chat_handler = None

    hf_dataset = load_dataset("yahma/alpaca-cleaned", split="train[:10]")

    records = []
    for row in tqdm(hf_dataset):
        hf_prompt = row["instruction"]

        # Formatting specific to alpaca-cleaned dataset
        if row["input"]:
            hf_prompt += "\n\n" + row["input"]
        messages = [{"role": "user", "content": hf_prompt}]

        base_response = _get_response(llm, messages)
        record = {
            "instruction": row["instruction"],
            "input": row["input"],
            "prompt": hf_prompt,
            "original response": row["output"],
            "base response": base_response,
        }
        records.append(record)

    frame = pd.DataFrame(records)
    frame.to_csv("out.tsv", sep="\t", quoting=csv.QUOTE_ALL, index=False)


if __name__ == "__main__":
    main()
