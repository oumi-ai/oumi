# Design Specification
This file contains a design spec for the scripts in this library.

## Pre-process Banking-77 Dataset
Create a Python script that does the following, getting user confirmation after each step has been implemented before proceeding to the next step:

### Step 1: Download and read dataset
Download the Hugging Face Hub dataset: https://huggingface.co/datasets/PolyAI/banking77 and load into two Pandas dataframes `df` and `df_test` corresponding to the two splits: `datasets.Split.TRAIN` and `datasets.Split.TEST`.

### Step 2: Transform into Oumi-compatible dataset
Write a function that translates each row of `df` or `df_test` so it is of the following format:

```python
sample = {"messages": [{"role": "system", "content": classifier_instruction}, {"role": "user", "content": query}, {"role": "assistant", "content": str(label_id)}], "metadata": {"label": str(label_id), "label_name": label_name"}}
```

`classifier_instruction` is a prompt to instruct the LLM how to classify user intent for the `banking77` dataset:

```python
classifier_instruction = f"""You are a banking intent classifier. Classify the user's query into one of  77 banking intents (output is a single integer ID).


IDS:
{id_list}


CRITICAL INSTRUCTIONS:
1. Choose exactly one integer ID (0-76).
2. Reply with ONLY that number. No words, no reasoning, no punctuation.
Examples: 0, 1, 42


{"EXAMPLES TO HELP DISTINGUISH SIMILAR INTENTS:\n\n" + '\n\n'.join(in_context_examples) if in_context_examples is not None and len(in_context_examples) > 0 else ""}
Remember: Respond with ONLY the numeric ID, nothing else."""
```

`id_list` is a string of classification labels constructed as follows:

```python
id_list = "0: activate_my_card\n1: age_limit\n2: apple_pay_or_google_pay" etc...
```

I have omitted the remainder of `id_list`. The text labels can be found, in ascending order starting from 0, one per line, in banking77_label.txt.

`in_context_examples is a list of *optional* in-context examples that help the model distinguish between two semantically similar labels, where a single element has the form, for example:

```python
"card_arrival (ID 11) vs card_delivery_estimate (ID 12):\n   card_arrival = asking about YOUR specific card that hasn't arrived yet (tracking, status)\n   card_delivery_estimate = asking about general delivery timeframes/how long it takes\n   - Query: \"Could you send me an update on the arrival of my card?\" → 11\n   - Query: \"My card was supposed to arrive, but hasn't?\" → 11\n   - Query: \"I have not received my card and it's been a week, what do I do?\" → 11\n   - Query: \"can you express my card to me?\" → 12\n   - Query: \"Can I choose the day it's delivered?\" → 12\n   - Query: \"i need to add express delivery if that's an option\" → 12"
```

The file banking77_in_context_examples.txt contains 8 such in-context examples in this format, one per line.

`query` contains the user's banking query and there is one per row of the dataset. An example is: "If I bought something I didn't like, can I get a refund?"

`label_id` contains the integer label for the ground truth classification label for `query` as given by the dataset. This value must be represented as a string, not an integer, in the transformed dataset.

`label_name` contains the corresponding human-readable label for `label_id` as specified in banking77_labels.txt.

### Step 3: Create transformed dataset
Using the function created in Step 2, create the following JSONL datasets:

* banking77-test.jsonl → use `df_test` and `in_context_examples = []`
* banking77-test-1-shot.jsonl → use `df_test` and `in_context_examples` contains 1 example from banking77_in_context_examples randomly chosen for *each* row
* banking77-test-3-shot.jsonl → use `df_test` and `in_context_examples` contains 3 example from banking77_in_context_examples randomly chosen for *each* row
* banking77-test-5-shot.jsonl → use `df_test` and `in_context_examples` contains 5 examples from banking77_in_context_examples randomly chosen for *each* row
* banking77-train.jsonl → use `df` and `in_context_examples = []`
