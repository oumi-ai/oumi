# CLI Usage

The Judge framework provides a command-line interface for evaluating datasets without writing Python code.
This is particularly useful for batch evaluation, pipeline integration, and quick testing.

The Judge CLI is accessed through the `oumi judge` command:

```bash
oumi judge dataset \
    --config CONFIG_FILE \
    --input INPUT_FILE \
    [--output OUTPUT_FILE \]
    [--display-raw-output]
```

Arguments
- `--config`: Path to the judge configuration YAML file. This can either be a local file or a file retrieved from Oumi's GitHub repository using `oumi:// prefix`
(e.g. `oumi://configs/projects/judges/generic/truthfulness.yaml`)
- `--input`: Path to the input dataset (JSONL format)
- `--output`: Path to save results (JSONL format). If not specified, results are displayed in a formatted table
- `--display-raw-output`: Include raw model output in the displayed table (when no output file is specified)

## Input Format

The input file must be in JSONL (JSON Lines) format.
Each line contains a JSON object with the fields referenced in the judge configuration's prompt template.

Example:

For a judge configuration with prompt template `"Rate the helpfulness: {question} | {answer}"`:
```json
{"question": "What is Python?", "answer": "Python is a programming language."}
{"question": "How to cook pasta?", "answer": "I don't know."}
```

## Output Format

If an `--output-file` was not specified, results are displayed in the terminal as a formatted table.
```
                     Judge Results
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Judgment  ┃ Judgment Score ┃ Explanation                  ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ True      │ 1.0            │ Clear and accurate response  │
│ False     │ 0.0            │ Did not address question     │
└───────────┴────────────────┴──────────────────────────────┘
```

When using `--output-file`, results are saved in JSONL format with detailed information:

```json
{"raw_output": "<judgment>True</judgment><explanation>Clear and accurate response</explanation>", "parsed_output": {"judgment": "True", "explanation": "Clear and accurate response"}, "field_values": {"judgment": true, "explanation": "Clear and accurate response"}, "field_scores": {"judgment": 1.0}}
{"raw_output": "<judgment>False</judgment><explanation>Did not address question</explanation>", "parsed_output": {"judgment": "False", "explanation": "Did not address question"}, "field_values": {"judgment": false, "explanation": "Did not address question"}, "field_scores": {"judgment": 0.0}}
```
