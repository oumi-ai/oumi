import os
import threading
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Queue

import jsonlines
import pandas as pd
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm


class MultithreadedRatelimitedClient:
    def __init__(self, openai_client, num_calls=15, period_seconds=60, num_threads=15):
        """Client makes one call per thread and limits the total calls in seconds."""
        self._client = openai_client
        self._num_threads = num_threads

        @sleep_and_retry
        @limits(calls=num_calls, period=period_seconds)
        def _single_call(messages):
            completion = self._client.chat.completions.create(
                messages=messages, model=self._model
            )
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content}
            )
            self._queue.put(messages)
            return messages

        self._call = _single_call

    def get_responses(self, samples, output_file_path, model):
        """Get responses for samples. Responses sent to output file and returned."""
        self._queue = Queue()
        self._model = model

        def _thread_write_to_file():
            while True:
                messages = self._queue.get()
                if messages is None:
                    self._queue.task_done()
                    break

                with jsonlines.open(output_file_path, mode="a") as writer:
                    json_obj = {"messages": messages}
                    writer.write(json_obj)
                    self._queue.task_done()

        threading.Thread(target=_thread_write_to_file, daemon=True).start()

        with ThreadPool(self._num_threads) as pool:
            responses = list(tqdm(pool.imap(self._call, samples), total=len(samples)))

        self._queue.put(None)
        self._queue.join()
        return responses


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
    print(f"Files will be output to {OUTPUT_FILE_PATH}")

    # Limit to 50 QPS (likely doesn't actually hit this limit)
    multhreaded_client = MultithreadedRatelimitedClient(
        openai_client=client,
        num_calls=50,
        period_seconds=1,
        num_threads=50,
    )

    json_objects = pd.read_json(INPUT_FILE, lines=True)
    all_messages = json_objects["messages"].to_list()

    _ = multhreaded_client.get_responses(all_messages, OUTPUT_FILE_PATH, model=MODEL)

    print("Inference complete")


if __name__ == "__main__":
    main()
