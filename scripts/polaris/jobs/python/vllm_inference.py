import os
import threading
import time
from pathlib import Path
from queue import Queue

import jsonlines
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def _get_model_name(model_id: str) -> str:
    segments = model_id.split("/")

    # Get first 5 characters of the snapshot id to keep the name small.
    snapshot = segments[-1][:5]
    model_segment_key = "models--"
    for s in segments:
        if model_segment_key in s:
            model_name = s[len(model_segment_key) :]
            return f"{model_name}_{snapshot}"
    # If the model_id is not in the HF format (ex. it's a path to a custom model),
    # return up to the last 20 characters.
    # We replace slashes in the directory path with underscores since the model
    # name is used as part of a filename.
    return model_id.replace("/", "_")[-20:]


def main() -> None:
    """Run inference against vLLM model hosted as an OpenAI API."""
    openai_api_key = "EMPTY"
    IP = os.environ["THIS_IP_ADDRESS"]
    openai_api_base = f"http://{IP}:8000/v1"
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    MODEL = models.data[0].id
    MODEL_NAME = _get_model_name(MODEL)
    JOB_NUMBER = os.environ["JOB_NUMBER"]
    INPUT_FILEPATH = Path(os.environ["OUMI_VLLM_INPUT_FILEPATH"])
    OUTPUT_DIR = Path(os.environ["OUMI_VLLM_OUTPUT_DIR"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILENAME = f"{JOB_NUMBER}_vllm_output_{TIMESTR}_{MODEL_NAME}.jsonl"
    OUTPUT_FILEPATH = OUTPUT_DIR / OUTPUT_FILENAME
    print(f"Input filepath is {INPUT_FILEPATH}")
    print(f"Files will be output to {OUTPUT_FILEPATH}")

    if not INPUT_FILEPATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILEPATH}")
    json_objects = pd.read_json(INPUT_FILEPATH, lines=True)
    all_messages = json_objects["messages"].to_list()
    write_queue = Queue()

    def _thread_write_to_file():
        while True:
            messages = write_queue.get()
            if messages is None:
                write_queue.task_done()
                break

            with jsonlines.open(OUTPUT_FILEPATH, mode="a") as writer:
                json_obj = {"messages": messages}
                writer.write(json_obj)
                write_queue.task_done()

    threading.Thread(target=_thread_write_to_file, daemon=True).start()
    for messages in tqdm(all_messages):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL,
        )

        messages.append(
            {"role": "assistant", "content": chat_completion.choices[0].message.content}
        )
        write_queue.put(messages)

    write_queue.put(None)
    write_queue.join()
    print("Inference complete")


if __name__ == "__main__":
    main()
