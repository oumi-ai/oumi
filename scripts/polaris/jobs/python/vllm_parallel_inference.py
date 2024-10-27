import os
import random
import time
from pathlib import Path
from queue import Queue

import gevent
import jsonlines
import locust
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

RANDOM_SEED = 42


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


def _get_metrics(
    total_input_tokens: int,
    total_output_tokens: int,
    delta_time: float,
    requests_completed: int,
):
    input_tokens_per_second = total_input_tokens / delta_time
    output_tokens_per_second = total_output_tokens / delta_time
    total_tokens_per_second = (total_input_tokens + total_output_tokens) / delta_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(delta_time))
    metrics = {
        "Total tok/s": round(total_tokens_per_second, 2),
        "Total tokens": total_input_tokens + total_output_tokens,
        "Input tok/s": round(input_tokens_per_second, 2),
        "Input tokens": total_input_tokens,
        "Output tok/s": round(output_tokens_per_second, 2),
        "Output tokens": total_output_tokens,
        "Requests/s": round(requests_completed / delta_time, 2),
        "Request count": requests_completed,
        "Elapsed Time": elapsed_time,
        "Elapsed Time (seconds)": round(delta_time, 2),
    }
    return metrics


def main():
    """Run parallelized inference against a vLLM server."""
    locust.log.setup_logging("INFO")
    random.seed(RANDOM_SEED)

    IP = os.environ["THIS_IP_ADDRESS"]
    openai_api_base = f"http://{IP}:8000/v1"
    client = OpenAI(
        base_url=openai_api_base,
    )
    models = client.models.list()
    MODEL = models.data[0].id
    MODEL_NAME = _get_model_name(MODEL)

    NUM_WORKERS = int(os.environ["OUMI_VLLM_NUM_WORKERS"])
    SPAWN_RATE = int(os.environ["OUMI_VLLM_WORKERS_SPAWNED_PER_SECOND"])
    print(f"Num workers: {NUM_WORKERS}")
    JOB_NUMBER = os.environ["JOB_NUMBER"]
    INPUT_FILEPATH = Path(os.environ["OUMI_VLLM_INPUT_FILEPATH"])
    OUTPUT_DIR = Path(os.environ["OUMI_VLLM_OUTPUT_DIR"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILENAME = f"{JOB_NUMBER}_vllm_output_{TIMESTR}_{MODEL_NAME}.jsonl"
    METRIC_FILENAME = f"{JOB_NUMBER}_vllm_metrics_{TIMESTR}_{MODEL_NAME}.jsonl"
    OUTPUT_FILEPATH = OUTPUT_DIR / OUTPUT_FILENAME
    METRIC_FILEPATH = OUTPUT_DIR / METRIC_FILENAME
    REQUEST_RETRIES = 3
    print(f"Input file is {INPUT_FILEPATH}")
    print(f"Files will be output to {OUTPUT_DIR}")

    if not INPUT_FILEPATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILEPATH}")
    json_objects = pd.read_json(INPUT_FILEPATH, lines=True)
    ALL_MESSAGES = json_objects["messages"].to_list()

    global output_queue
    output_queue = Queue()
    global input_queue
    input_queue = Queue()

    for i in range(len(ALL_MESSAGES)):
        input_queue.put(i)

    global REQUEST_TIMES
    REQUEST_TIMES = [(0.0, 0.0) for _ in range(len(ALL_MESSAGES))]
    global failed_request_counts
    failed_request_counts = {}

    class VllmWorker(locust.FastHttpUser):
        host = openai_api_base
        network_timeout = 300
        connection_timeout = 300
        max_retries = 3

        @locust.task
        def run_inference(self):
            """Runs inference by pulling indices from queue shared by all workers."""
            global output_queue
            global input_queue
            global REQUEST_TIMES
            global failed_request_counts

            while True:
                index = input_queue.get()
                input_queue.task_done()
                messages = ALL_MESSAGES[index]
                payload = {"model": MODEL, "messages": messages, "seed": RANDOM_SEED}
                try:
                    request_sent_time = time.perf_counter()
                    response = self.client.post(url="/chat/completions", json=payload)
                    request_complete_time = time.perf_counter()
                    response_dict = response.json()
                    response_message = response_dict["choices"][0]["message"]["content"]
                    num_input_tokens = response_dict["usage"]["prompt_tokens"]
                    num_output_tokens = response_dict["usage"]["completion_tokens"]
                    output_queue.put(
                        (index, response_message, num_input_tokens, num_output_tokens)
                    )
                    REQUEST_TIMES[index] = (request_sent_time, request_complete_time)
                except Exception as e:
                    print(e)
                    if index not in failed_request_counts:
                        failed_request_counts[index] = 0

                    failed_request_counts[index] += 1
                    if failed_request_counts[index] >= REQUEST_RETRIES:
                        print(f"Failed for request at index {index}, skipping...")
                        continue
                    input_queue.put(index)

    # setup Environment and Runner
    env = locust.env.Environment(user_classes=[VllmWorker], events=locust.events)
    runner = env.create_local_runner()

    # execute init event handlers (only really needed if you have registered any)
    env.events.init.fire(environment=env, runner=runner)

    # start a greenlet that periodically outputs the current stats
    gevent.spawn(locust.stats.stats_printer(env.stats))

    # start a greenlet that save current stats to history
    gevent.spawn(locust.stats.stats_history, env.runner)

    # start the test
    runner.start(NUM_WORKERS, spawn_rate=SPAWN_RATE)

    print("Waiting for inference to finish.")
    start = time.perf_counter()
    pbar = tqdm(total=len(ALL_MESSAGES))
    total_input_tokens = 0
    total_output_tokens = 0
    requests_completed = 0
    while requests_completed < len(ALL_MESSAGES):
        queue_item = output_queue.get()
        index, response_message, num_input_tokens, num_output_tokens = queue_item
        output_queue.task_done()
        messages = ALL_MESSAGES[index]
        total_input_tokens += num_input_tokens
        total_output_tokens += num_output_tokens

        messages.append({"role": "assistant", "content": response_message})
        requests_completed += 1
        pbar.update()

        with jsonlines.open(OUTPUT_FILEPATH, mode="a") as writer:
            request_sent_time, request_complete_time = REQUEST_TIMES[index]
            elapsed_time = request_complete_time - request_sent_time
            json_obj = {
                "messages": messages,
                "original_index": index,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "elapsed_time": elapsed_time,
                "request_sent_time": request_sent_time,
                "request_completed_time": request_complete_time,
            }
            writer.write(json_obj)

        if requests_completed % 50 == 0:
            metrics = _get_metrics(
                total_input_tokens,
                total_output_tokens,
                time.perf_counter() - start,
                requests_completed,
            )

            with jsonlines.open(METRIC_FILEPATH, "a") as metric_writer:
                metric_writer.write(metrics)

    average_request_completion_time = sum([x[1] - x[0] for x in REQUEST_TIMES]) / len(
        REQUEST_TIMES
    )
    metrics = _get_metrics(
        total_input_tokens,
        total_output_tokens,
        time.perf_counter() - start,
        requests_completed,
    )
    metrics["Avg Request Completion Time (seconds)"] = round(
        average_request_completion_time, 2
    )

    with jsonlines.open(METRIC_FILEPATH, "a") as metric_writer:
        metric_writer.write(metrics)

    print("All responses written to file.")
    runner.quit()

    print("Joining runners...")
    # wait for the greenlets
    runner.greenlet.join()
    print("OUMI INFERENCE JOB DONE")


if __name__ == "__main__":
    main()
