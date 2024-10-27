import argparse
import copy
import json
import os

import gspread
import pandas as pd

#########################################################
################ Google Sheets Access ###################
#########################################################


def open_gsheet(sheet_url, tab_name, service_account_file):
    """Open a Google Sheet and return the requested tab."""
    gc = gspread.service_account(filename=service_account_file)
    sheet = gc.open_by_url(sheet_url)
    tab = sheet.worksheet(tab_name)
    return tab


#########################################################
########## Load from GSheets: Prompts, Data #############
#########################################################


def get_df_prompt_templates(sheet_url, sheet_tab, service_account_file):
    """Load the prompt templates from a Google Sheet."""
    worksheet_prompts = open_gsheet(
        sheet_url=sheet_url,
        tab_name=sheet_tab,
        service_account_file=service_account_file,
    )
    df_prompt_templates = pd.DataFrame(worksheet_prompts.get_all_records())
    df_prompt_templates = df_prompt_templates.set_index("index")
    print(f"Prompt templates loaded from `{worksheet_prompts.title}`.")
    return df_prompt_templates


def get_df_dataset(sheet_url, sheet_tab, service_account_file):
    """Load the dataset from a Google Sheet."""
    worksheet_dataset = open_gsheet(
        sheet_url=sheet_url,
        tab_name=sheet_tab,
        service_account_file=service_account_file,
    )
    print(f"Dataset loaded from `{worksheet_dataset.title}`.")
    return pd.DataFrame(worksheet_dataset.get_all_records())


#########################################################
############### Generate Judge Prompts ##################
#########################################################


def get_prompt_templates(attribute_names, sheet_url, sheet_tab, service_account_file):
    """Load the prompt templates, structure prompt, and return as a dictionary."""
    df_prompt_templates = get_df_prompt_templates(
        sheet_url, sheet_tab, service_account_file
    )

    prompt_templates = {}
    for attribute_name in attribute_names:
        prompt_name = get_prompt_col_name(attribute_name=attribute_name)
        prompt_templates[prompt_name] = [
            {
                "role": "system",
                "content": df_prompt_templates[attribute_name]["system-description"],
            },
            {
                "role": "user",
                "content": df_prompt_templates[attribute_name]["user-example1"],
            },
            {
                "role": "assistant",
                "content": df_prompt_templates[attribute_name]["assistant-answer1"],
            },
            {
                "role": "user",
                "content": df_prompt_templates[attribute_name]["user-example2"],
            },
            {
                "role": "assistant",
                "content": df_prompt_templates[attribute_name]["assistant-answer2"],
            },
            {
                "role": "user",
                "content": df_prompt_templates[attribute_name]["user-request"],
            },
        ]
    return prompt_templates


def generate_judge_prompt(
    row, prompt_template, request_col_name, context_col_name, response_col_name
):
    """Replace variables in prompt templates and return as json."""
    content = prompt_template[-1]["content"]
    content = content.replace("$user_input_request", row[request_col_name])
    content = content.replace("$ai_response", str(row[response_col_name]))
    if not context_col_name or row.isna()[context_col_name]:
        content = content.replace("\n\n$optional_context", "")
    else:
        content = content.replace("$optional_context", str(row[context_col_name]))
    prompt = copy.deepcopy(prompt_template)
    prompt[-1]["content"] = content
    return json.dumps(prompt)


#########################################################
#################### Main Function ######################
#########################################################


# Default column names (can be overwritten by args).
REQUEST_COLUMN_NAME = "request"
CONTEXT_COLUMN_NAME = "context"
RESPONSE_COLUMN_NAME = "response"

# Default file paths and urls (can be overwritten by args).
SERVICE_ACCOUNT_FILE = f"/Users/{os.getlogin()}/.config/gspread/service_account.json"
# Inputs.
PROMPTS_SHEET_URL = "https://docs.google.com/spreadsheets/d/1pOw3r7Q_7gKWfU44hDS0nlBVa9Syo_d5IxPWsMZmMXM"
PROMPTS_SHEET_TAB = "v_8-29"
DATASET_SHEET_URL = "https://docs.google.com/spreadsheets/d/1YO17DygGpFkEkLVK0dN5YvoVmuw8-0BYF0CQx3qYXRM"
DATASET_SHEET_TAB = "Dataset v_8-22"
# Output.
DATASET_OUT_FILE = "./dataset_with_prompts.csv"
DATASET_OUT_FILE_PREFIX_POLARIS = "./prompts_polaris_"

# Attribute names.
attribute_names = [
    "helpful",
    "honest",
    "safe",
    "valid",
]


# Column names.
def get_prompt_col_name(attribute_name):
    """Get the column name for the prompt."""
    return "prompt_" + attribute_name


def main(args):
    """Generate judge prompts for a dataset."""
    prompt_templates = get_prompt_templates(
        attribute_names=attribute_names,
        sheet_url=args.prompts_sheet_url,
        sheet_tab=args.prompts_sheet_tab,
        service_account_file=args.service_account_file,
    )
    df_dataset = get_df_dataset(
        sheet_url=args.dataset_sheet_url,
        sheet_tab=args.dataset_sheet_tab,
        service_account_file=args.service_account_file,
    )
    for attribute_name in attribute_names:
        prompt_col_name = get_prompt_col_name(attribute_name)
        prompt_template = prompt_templates[prompt_col_name]
        df_dataset[prompt_col_name] = df_dataset.apply(
            generate_judge_prompt,
            args=(
                prompt_template,
                args.request_col_name,
                args.context_col_name,
                args.response_col_name,
            ),
            axis=1,
        )

    # Save the dataset with prompts.
    df_dataset.to_csv(DATASET_OUT_FILE, index=False)
    print(f"Dataset with prompts saved at {DATASET_OUT_FILE}.")

    # Save the prompts (stand-alone) for Polaris.
    for attribute_name in attribute_names:
        prompt_col_name = get_prompt_col_name(attribute_name)
        file_path = f"{DATASET_OUT_FILE_PREFIX_POLARIS}{attribute_name}.jsonl"
        with open(file_path, "w") as file_handle:
            for dataset_index in df_dataset.index:
                prompt = json.loads(df_dataset[prompt_col_name][dataset_index])
                entry = {"messages": prompt}
                print(json.dumps(entry), file=file_handle)
        print(f"Prompts for `{attribute_name}` saved at {file_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge a dataset.")
    parser.add_argument(
        "--request_col_name",
        type=str,
        default=REQUEST_COLUMN_NAME,
        help="Name of column that includes the request.",
    )
    parser.add_argument(
        "--context_col_name",
        type=str,
        default=CONTEXT_COLUMN_NAME,
        help="Name of column that includes the request's context.",
    )
    parser.add_argument(
        "--response_col_name",
        type=str,
        default=RESPONSE_COLUMN_NAME,
        help="Name of column that includes the reponse.",
    )
    parser.add_argument(
        "--prompts_sheet_url",
        type=str,
        default=PROMPTS_SHEET_URL,
        help="Url of the gsheet that includes the judge's prompts.",
    )
    parser.add_argument(
        "--prompts_sheet_tab",
        type=str,
        default=PROMPTS_SHEET_TAB,
        help="Name of tab that includes the judge's prompts.",
    )
    parser.add_argument(
        "--dataset_sheet_url",
        type=str,
        default=DATASET_SHEET_URL,
        help="Url of the gsheet that includes the dataset to be judged.",
    )
    parser.add_argument(
        "--dataset_sheet_tab",
        type=str,
        default=DATASET_SHEET_TAB,
        help="Name of tab that includes the dataset to be judged.",
    )
    parser.add_argument(
        "--dataset_out_file",
        type=str,
        default=DATASET_OUT_FILE,
        help="Path of the output CSV file where the prompts will be saved.",
    )
    parser.add_argument(
        "--dataset_out_file_prefix_polaris",
        type=str,
        default=DATASET_OUT_FILE_PREFIX_POLARIS,
        help="Prefix of output JSONL files to save the prompts for Polaris inference.",
    )
    parser.add_argument(
        "--service_account_file",
        type=str,
        default=SERVICE_ACCOUNT_FILE,
        help="Path to the service account json file (for gsheet access).",
    )
    args = parser.parse_args()
    main(args)
