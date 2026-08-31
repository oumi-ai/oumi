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
The table shows one column per output field, followed by the row's score.

```
Overall Score: 50.00%
                          Judge Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ explanation                  ┃ judgment ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ Clear and accurate response  │ True     │ 1.00  │
├──────────────────────────────┼──────────┼───────┤
│ Did not address question     │ False    │ 0.00  │
└──────────────────────────────┴──────────┴───────┘
```

The overall score is the mean of the per-row scores, and is only shown when every
row produced one. A row's score comes from `aggregate_score` for a
{doc}`rubric judge </user_guides/judge/judge_config>`, and from the `judgment`
field's score otherwise; rows without a score show `N/A`.

Because the columns are the judge's output fields, a rubric judge shows one column
per criterion:

```
Overall Score: 62.50%
                  Judge Results
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┓
┃ relevance ┃ groundedness ┃ completeness ┃ Score ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━┩
│ True      │ True         │ complete     │ 1.00  │
├───────────┼──────────────┼──────────────┼───────┤
│ True      │ False        │ incomplete   │ 0.25  │
└───────────┴──────────────┴──────────────┴───────┘
```

Explanation columns are long, so once there is more than one of them they are left
out of the table and a note says so. Use `--output` to capture them, or `--raw` to
print the judge's full response.

When using `--output-file`, results are saved in JSONL format with detailed information. `aggregate_score` is set only by judges that score several criteria at once, and is `null` otherwise. Each line also carries `output_fields` and `response_format`, elided here for brevity:

```json
{"raw_output": "<judgment>True</judgment><explanation>Clear and accurate response</explanation>", "parsed_output": {"judgment": "True", "explanation": "Clear and accurate response"}, "field_values": {"judgment": true, "explanation": "Clear and accurate response"}, "field_scores": {"judgment": 1.0}, "aggregate_score": null}
{"raw_output": "<judgment>False</judgment><explanation>Did not address question</explanation>", "parsed_output": {"judgment": "False", "explanation": "Did not address question"}, "field_values": {"judgment": false, "explanation": "Did not address question"}, "field_scores": {"judgment": 0.0}, "aggregate_score": null}
```
