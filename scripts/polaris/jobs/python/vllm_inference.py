import os
import threading
import time
from pathlib import Path
from queue import Queue

import jsonlines
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def _get_model_name(model_id):
    segments = model_id.split("/")
    snapshot = segments[-1][:5]
    model_segment_key = "models--"
    for s in segments:
        if model_segment_key in s:
            model_name = s[len(model_segment_key) :]
            return f"{model_name}_{snapshot}"


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
    INPUT_FILE = os.environ["LEMA_VLLM_INPUT_PATH"]
    OUTPUT_PATH = os.environ["LEMA_VLLM_OUTPUT_PATH"]
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE_NAME = f"{JOB_NUMBER}_vllm_output_{TIMESTR}_{MODEL_NAME}.jsonl"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_PATH, OUTPUT_FILE_NAME)
    print(f"Input file is {INPUT_FILE}")
    print(f"Files will be output to {OUTPUT_FILE_PATH}")

    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    json_objects = pd.read_json(INPUT_FILE, lines=True)
    all_messages = json_objects["messages"].to_list()
    write_queue = Queue()

    def _thread_write_to_file():
        while True:
            messages = write_queue.get()
            if messages is None:
                write_queue.task_done()
                break

            with jsonlines.open(OUTPUT_FILE_PATH, mode="a") as writer:
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
