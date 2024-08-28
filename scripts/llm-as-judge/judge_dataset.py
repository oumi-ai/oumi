import argparse
import json
import os
import re
import time

import numpy as np
import openai
import pandas as pd
from google.auth import default
from google.auth.transport import requests
from tqdm import tqdm

#########################################################
###################### Clients ##########################
#########################################################


def get_llama_client(judge_config):
    """Create client for Llama 3.1 405B model."""
    project_id = judge_config.project_id
    model_location = judge_config.model_location

    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = requests.Request()
    credentials.refresh(auth_request)

    client = openai.OpenAI(
        base_url=f"https://{model_location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{model_location}/endpoints/openapi/chat/completions?",
        api_key=credentials.token,
    )
    return client


def get_gpt4_client(opena_ai_api_key):
    """Create client for GPT-4 model."""
    client = openai.OpenAI(api_key=opena_ai_api_key)
    return client


#########################################################
################### Judge Inference #####################
#########################################################


def get_response(prompt, judge_config):
    """Get response from judge."""
    # Get client.
    if not judge_config.client:
        if judge_config.model_name == "GPT-4":
            client = get_gpt4_client(judge_config.opena_ai_api_key)
        elif judge_config.model_name == "Llama3.1 405B":
            client = get_llama_client(judge_config)
        else:
            raise ValueError("Invalid model name.")
    else:
        client = judge_config.client

    # Return response.
    return client.chat.completions.create(
        messages=prompt,
        model=judge_config.model_id,
        temperature=judge_config.temperature,
        max_tokens=judge_config.max_tokens,
    )


def get_response_with_retry(prompt, judge_config):
    """Get response from judge with retries."""
    last_exception = None
    sleep_time = judge_config.sleep_time

    for _ in range(judge_config.retries):
        # Sleep.
        if sleep_time:
            time.sleep(sleep_time)

        # Inference.
        try:
            response = get_response(prompt, judge_config)
            return response, last_exception
        except Exception as e:
            last_exception = type(e).__name__
            sleep_time = sleep_time * 2

    return None, last_exception


#########################################################
#################### Judge Configs ######################
#########################################################


class JudgeConfigLlama:
    def __init__(self):
        """Config for Llama 3.1 405B model."""
        self.model_name = "Llama3.1 405B"
        self.model_id = "meta/llama3-405b-instruct-maas"
        self.project_id = "lema-dev"
        self.model_location = "us-central1"
        self.temperature = 0.2  # FIXME: TBD for Llama
        self.max_tokens = 1000  # FIXME: TBD for Llama
        self.client = None
        # To be adjusted:
        self.sleep_time = 5
        self.retries = 4


class JudgeConfigGPT:
    def __init__(self, opena_ai_api_key):
        """Config for GPT-4 model."""
        if not opena_ai_api_key:
            raise ValueError("OpenAI API key is required.")
        self.model_name = "GPT-4"
        self.model_id = "gpt-4"
        self.opena_ai_api_key = opena_ai_api_key
        self.temperature = 0.2
        self.max_tokens = 1000
        self.client = get_gpt4_client(opena_ai_api_key)
        # To be adjusted:
        self.sleep_time = 0
        self.retries = 4


#########################################################
################# Extract Judge Answer ##################
#########################################################


def extract_bool_answer(full_answer):
    """Extract boolean answer from judge's reponse."""
    MATCH_PATTERN = "<answer>.*</answer>"

    if not full_answer:
        print("(!) Full Answer ERROR:", full_answer)
        return "Error"

    answer = re.search(MATCH_PATTERN, full_answer)
    if answer:
        answer = answer.group(0)
        answer = answer.replace("<answer>", "")
        answer = answer.replace("</answer>", "")
    else:
        print("(!) Answer ERROR:", answer)
        return "Error"

    if (len(answer) >= 3) and (answer[:3].lower() == "yes"):
        return True
    elif (len(answer) >= 2) and (answer[:2].lower() == "no"):
        return False
    else:
        print("(!) Extraction ERROR:", full_answer)
        return "Error"


#########################################################
#################### Main Function ######################
#########################################################

DEFAULT_JUDGE = "llama"
DEFAULT_DATASET_IN_PATH = (
    "/Users/kostas/Desktop/Judge/kostas-v7-sft-magpie/data/"
    "magpie_dataset_with_prompts.csv"
)
DEFAULT_DATASET_OUT_PATH = "./dataset_judged.csv"
DEFAULT_SUBSET_SIZE = 5

# Attribute names.
attribute_names = [
    "helpful",
    "honest",
    "safe",
]


# Column names.
def get_prompt_col_name(attribute_name):
    """Get the column name for the prompt."""
    return "prompt_" + attribute_name


def get_judge_answer_col_name(attribute_name):
    """Get the column name for the judge's answer."""
    return "judge_answer_" + attribute_name


def get_judge_answer_bool_col_name(attribute_name):
    """Get the column name for the judge's extracted bool answer."""
    return "judge_answer_tf_" + attribute_name


def get_judge_exception_col_name(attribute_name):
    """Get the column name for the judge's exception."""
    return "judge_exception_" + attribute_name


def main(args):
    """Run inference against LLM Judge."""
    # Load dataset.
    print(f"Loading dataset from {args.dataset_in_path}...")
    df_dataset = pd.read_csv(args.dataset_in_path)

    # Sample a subset of the dataset.
    if args.subset_size:
        df_dataset = df_dataset[: args.subset_size]

    # Define the judge config.
    if args.judge == "llama":
        print("Using Llama 3.1 405B as Judge.")
        judge_config = JudgeConfigLlama()
    elif args.judge == "gpt":
        print("Using GPT-4 as Judge.")
        judge_config = JudgeConfigGPT(opena_ai_api_key=args.opena_ai_api_key)
    else:
        raise ValueError("Invalid judge name.")

    # Create output columns.
    column_fns = [
        get_judge_answer_col_name,
        get_judge_answer_bool_col_name,
        get_judge_exception_col_name,
    ]
    for attribute_name in attribute_names:
        for column_fn in column_fns:
            column_name = column_fn(attribute_name)
            if column_name not in df_dataset.columns:
                df_dataset[column_name] = ""

    # Iterate over each attribute and judge the dataset entries.
    for attribute_name in attribute_names:
        for dataset_index in tqdm(df_dataset.index, desc=attribute_name):
            # Jump to first empty record.
            if any(
                df_dataset.loc[dataset_index, column_fn(attribute_name)]
                for column_fn in column_fns
            ):
                continue

            # Retrieve prompt (stored in JSON format).
            prompt_col_name = get_prompt_col_name(attribute_name)
            prompt = json.loads(df_dataset[prompt_col_name][dataset_index])

            # Query judge
            response, exception = get_response_with_retry(prompt, judge_config)
            response_bool = ""
            if response:
                if response.choices[0].finish_reason != "stop":
                    print("(!) Judge Response ERROR", response.choices[0].finish_reason)
                    response = None
                    exception = "judge_response_error"
                else:
                    response = response.choices[0].message.content
                    response_bool = extract_bool_answer(response)
            else:
                print("(!) Judge Expection", exception)

            # Save outcome.
            ans_col = get_judge_answer_col_name(attribute_name)
            ans_bool_col = get_judge_answer_bool_col_name(attribute_name)
            exc_col = get_judge_exception_col_name(attribute_name)
            df_dataset.loc[dataset_index, ans_col] = response
            df_dataset.loc[dataset_index, ans_bool_col] = response_bool
            df_dataset.loc[dataset_index, exc_col] = exception if exception else ""

    # Save dataset.
    df_dataset.to_csv(args.dataset_out_path, index=False)
    print(f"Saved dataset to {args.dataset_out_path}.")

    # Report aggregate judgements (i.e, True/False counts per attribute).
    for attribute_name in attribute_names:
        ans_bool_col = get_judge_answer_bool_col_name(attribute_name)
        tf_counts = np.unique(df_dataset[ans_bool_col], return_counts=True)
        print(attribute_name, tf_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge a dataset.")
    parser.add_argument(
        "-i",
        "--dataset_in_path",
        type=str,
        default=DEFAULT_DATASET_IN_PATH,
        help=(
            "CSV file containing the dataset to be judged. This dataset must have a "
            "column `prompt_{attribute}` for each attribute to be judged."
        ),
    )
    parser.add_argument(
        "-o",
        "--dataset_out_path",
        type=str,
        default=DEFAULT_DATASET_OUT_PATH,
        help="CSV file path to output the judgements.",
    )
    parser.add_argument(
        "-j",
        "--judge",
        type=str,
        choices=["llama", "gpt"],
        default=DEFAULT_JUDGE,
        help="Judge to be used for inference (`llama` or `gpt`).",
    )
    parser.add_argument(
        "-s",
        "--subset_size",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help="Number of samples to judge.",
    )
    parser.add_argument(
        "-k",
        "--opena_ai_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="Open AI api key.",
    )
    args = parser.parse_args()

    if (args.judge == "gpt") and not args.opena_ai_api_key:
        raise ValueError("--opena_ai_api_key is required for GPT judge.")
    main(args)
