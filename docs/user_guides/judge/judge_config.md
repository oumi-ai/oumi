# Judge Config

Oumi allows users to define their judge configurations through a `YAML` file, providing a flexible, human-readable, and easily customizable format for setting up LLM-based evaluations. By using `YAML`, users can effortlessly configure judge prompts, response formats, output types, and the underlying inference models. This approach not only streamlines the process of configuring evaluations but also ensures that configurations are easily versioned, shared, and reproduced across different environments and teams.

## Configuration Options

The configuration `YAML` file is loaded into the {py:class}`~oumi.core.configs.judge_config.JudgeConfig` class, and consists of `judge_params` ({py:class}`~oumi.core.configs.params.judge_params.JudgeParams`) and `inference_config` ({py:class}`~oumi.core.configs.inference_config.InferenceConfig`). The judge parameters define the evaluation criteria, prompts, and output format, while the inference configuration specifies the underlying judge model and generation parameters used for the judge's reasoning.

### Judge Parameters

#### Prompt Template
`prompt_template` *(required)*: This is a text prompt that defines the judge's behavior.

To be clear and effective, it should should include the following:
- Role Declaration: Clearly state that the model is acting as a judge and explain what it is evaluating.
- Inputs: List and explain the inputs the judge will receive (e.g., request, response, ground_truth).
- Evaluation Criteria: Specify the exact dimensions to judge.
- Data: Insert placeholders (e.g. `{request}`, `{response}`) for all the inputs listed above, so that they can be replaced at runtime with each example's actual inputs.
- Expected Output: Describe the expected output type. This must be consistent with the `judgment_type` below.

#### System Instruction
`system_instruction` *(optional)*: System message to guide the judge's behavior and evaluation criteria. It is a common practice to break down the judge prompt into two messages: A system instruction message (role: `system`), and a user prompt message (role: `user`). If we use a `system` role, then the `prompt_template` (described above) should only include information related to the particular example (i.e., the "Data"), while the `system_instruction` should include all the remaining fields that describe the judge's behavior, inputs, and output. See example in the [next section](/user_guides/judge/judge_config.md#configuration-example).



#### Template Variables
`template_variables` *(optional)*: Dictionary of variables to replace in `prompt_template` and `system_instruction`, before processing input data. In addition to the placeholders that will be replaced by each example, the user can also leverage this dict to define additional placeholders that will be statically replaced when loading the YAML file. This is useful for defining versatile judges that can be re-used with slight variations. For example, our {gh}`format compliance <configs/projects/judges/generic/format_compliance.yaml>` judge can be used to validate JSON, XML, HTML, etc outputs, just by updating the `response_format` template variable.

#### Response Format
`response_format`: How the judge should structure its output.

Options:
- `XML`: Structured XML with tagged fields (e.g., `<judgment>Yes</judgment>`)
- `JSON`: JSON object with named fields (e.g., `{"judgment": "Yes"}`)
- `RAW`: Plain text response

#### Guided Decoding

`use_guided_decoding` *(default: true)*: With `response_format: JSON`, Oumi derives a
JSON schema from the judge's output fields and enables guided decoding, so the response
is structurally guaranteed to be valid and to contain every expected field. This applies
to every LLM-based judge, and only to `JSON` — the other formats have no schema to
constrain against.

Set it to `false` to let the model generate freely:

```yaml
judge_params:
  response_format: JSON
  use_guided_decoding: false
```

#### Judgment Types
`judgment_type`: The expected type of the judge's primary output.

Options:
- `BOOL`: Boolean judgments (True/False, Yes/No)
- `INT`: Integer scores (e.g., 1-5 ratings)
- `FLOAT`: Floating-point values (e.g., 0.0-1.0 scores)
- `TEXT`: Free-form textual judgments
- `ENUM`: Categorical selections from predefined options

#### Judgment Scores
`judgment_scores` *(optional)*: For `ENUM` types, these scores enable mapping from category names to numeric scores.
Example: `{"excellent": 1.0, "good": 0.7, "poor": 0.3}`

#### Explanation
`include_explanation` *(default: false)*: Whether to include explanations alongside judgments.

#### Few-Shot Examples
`examples` *(optional)*: List of example evaluations to guide the judge.

Each example should include a dictionary that contains (key, value) pairs for:
- All input placeholders defined in the `prompt_template`.
- A `judgment` consistent with the `judgment_type`.
- An `explanation` text (if `include_explanation` is set) that explains the rationale of the judgment above.

### Inference Configuration
The `inference_config` section ({py:class}`~oumi.core.configs.inference_config.InferenceConfig`) specifies the model and generation parameters.
For more details, visit the {doc}`inference configuration </user_guides/infer/configuration>` page.


## Configuration Example
Here's a comprehensive configuration example, showing many of the available parameters:

```yaml
judge_params:
  system_instruction: |
    You are a truthfulness judge. Your task is to determine whether the response is factually accurate, grounded in verifiable information, and free from hallucinations, fabrications, or speculative claims. The response should not include opinions, beliefs, or unverifiable assertions.

    Carefully assess the response against the user’s request. Evaluate truthfulness by systematically checking: specific factual claims, numerical data, names and references, causal relationships, and generalizations for accuracy and verifiability.

    {expected_response}

    Evaluation Criteria:
    1. Factual Accuracy: All factual claims in the response must be correct and verifiable.
    2. Grounding: The response must be based on information that can be reasonably inferred or verified from the user’s request or common knowledge. It should not invent details or make unsupported assumptions.
    3. No Hallucination: The response must not include fabricated facts, imaginary entities, or incorrect references. Check specific names, dates, numbers, and citations carefully.
    4. No Opinions or Beliefs: The response must avoid presenting personal beliefs, subjective opinions, or unverifiable statements.
    5. Uncertainty and Limitations: Responses that acknowledge uncertainty, express appropriate caveats, or state limitations of knowledge are acceptable and should not be penalized for truthfulness, as long as they don't make false claims.
    6. Temporal Accuracy: Consider whether factual claims are accurate for the relevant time period. Facts that were true historically but are no longer current should be evaluated in context.

  prompt_template: |
    Here is the data:
    [BEGIN DATA]
    ***
    [user request]:
    {request}
    ***
    [response]:
    {response}
    ***
    [END DATA]

  # Template variables to statically replace YAML placeholders
  template_variables:
    expected_response: If the response is entirely truthful and grounded, respond with 'Yes'. If not, respond with 'No'. If you are unsure, respond with 'Unsure'.

  # Output configuration
  response_format: JSON          # XML, JSON, or RAW
  judgment_type: ENUM            # BOOL, INT, FLOAT, TEXT, or ENUM
  include_explanation: True

  # For categorical (ENUM) judgments with custom scoring
  judgment_scores:
    "Yes": 1.0
    "Unsure": 0.5
    "No": 0.0

  # Few-shot examples to guide the judge
  examples:
    - question: "What is the capital of France?"
      response: "Paris"
      judgment: "Yes"
      explanation: "The response is factually correct, since the capital of France is Paris."
    - question: "Who is the president of the US in 2025?"
      response: "The president is Joe Biden."
      judgment: "No"
      explanation: "The answer is incorrect. The president of the US in 2025 is Donald J. Trump."

inference_config:
  # Model configuration
  model:
    model_name: "gpt-4o"

  # Inference engine
  engine: OPENAI

  # Generation parameters
  generation:
    max_new_tokens: 8192
    temperature: 1.0
```

## Rubric Judges

A {py:class}`~oumi.judges.simple_judge.SimpleJudge` answers one question per call. A
{py:class}`~oumi.judges.rubric_judge.RubricJudge` answers several — one judgment per
criterion, from a single inference call — plus one `aggregate_score` combining them.

Add a `rubric_judge_params` section to select it:

```yaml
judge_params:
  system_instruction: |
    You are evaluating an answer to a question.
    Judge each criterion strictly on its own terms.

  prompt_template: |
    [Question]: {question}
    [Answer]: {answer}

  response_format: JSON

rubric_judge_params:
  aggregation: WEIGHTED_MEAN

  criteria:
    - id: correctness
      description: The answer is factually correct.
      judgment_type: BOOL
      weight: 2.0

    - id: clarity
      description: How clearly the answer is written.
      judgment_type: ENUM
      judgment_scores:
        excellent: 1.0
        good: 0.5
        poor: 0.0
      include_explanation: false   # on by default; opt out to save tokens

inference_config:
  model:
    model_name: "gpt-4o"
  engine: OPENAI
  generation:
    max_new_tokens: 8192
    temperature: 0.0
```

Each criterion becomes one field in the judge's response, preceded by its explanation
unless you turn that off. The rubric above asks the model for exactly this:

```json
{
  "correctness_explanation": "The answer correctly states that 2+2 equals 4.",
  "correctness": "Yes",
  "clarity": "poor"
}
```

### Reading the Results

`judge()` returns one {py:class}`~oumi.judges.base_judge.JudgeOutput` per input row:

```python
from oumi.judges.rubric_judge import RubricJudge

output = RubricJudge("./my_rubric.yaml").judge(
    [{"question": "What is 2+2?", "answer": "its 4 i guess"}]
)[0]

output.field_values["correctness"]              # True — the judgment, typed
output.field_values["correctness_explanation"]  # "The answer correctly states that…"
output.field_values["clarity"]                  # "poor" — the label the judge chose
output.field_scores["clarity"]                  # 0.0 — that label's score
output.aggregate_score                          # 0.667 — (2 × 1.0 + 1 × 0.0) / 3
```

- `field_values` — what the judge said, converted to the criterion's `judgment_type`.
- `field_scores` — one entry per output field: the criterion's numeric score, or
  `None` for explanation fields and for criteria that carry no score.
- `aggregate_score` — the single score for the row; `None` for non-rubric judges.

### Criterion Parameters

| Parameter | Description |
|-----------|-------------|
| `id` | The criterion's output field name. Identifier-like: letters, digits and underscores, not starting with a digit. |
| `description` | What to assess. Written into the prompt, so phrase it as an instruction to the judge. |
| `judgment_type` | `BOOL` (default), `ENUM`, `INT`, `FLOAT`, or `TEXT`. See [Judgment Types](#judgment-types). |
| `judgment_scores` | For `ENUM`, maps each label to a score. See [Scoring](#scoring). |
| `include_explanation` | Emit a `{id}_explanation` field just before the judgment. Default `true`. |
| `weight` | Relative weight under `WEIGHTED_MEAN` aggregation. Default `1.0`. |

Criteria appear in the prompt, and in the judge's response, in the order you list
them. The names `explanation` and anything ending in `_explanation` are reserved for
the generated explanation fields.

Explanations are on by default because the judge writes them *before* the judgment, so
it reasons before committing. That matters most under guided decoding, where a
schema-constrained response leaves no other room to think.

### Scoring

A criterion feeds the aggregate only if it produces a number:

| `judgment_type` | Score |
|-----------------|-------|
| `BOOL` | `1.0` / `0.0`, automatically |
| `ENUM` | whatever `judgment_scores` maps the chosen label to |
| `INT`, `FLOAT`, `TEXT` | none, unless you supply `judgment_scores` |

Criteria without a score are still judged and reported — they just sit outside the
aggregate, and the judge names them in a warning when the config loads.

An `ENUM` label may map to `null`, meaning **this label carries no score**:

```yaml
judgment_scores:
  good: 1.0
  poor: 0.0
  not_applicable: null
```

The judge can still choose `not_applicable` and you will see it in `field_values`, but
the criterion then drops out of that row's aggregate — weight and all, from both the
numerator and the denominator. An N/A never drags the score down the way `0.0` would.
`null` is allowed only for `ENUM`; the other types parse their value out of the label,
so an unscored label could not be told apart from a failed parse.

### Aggregation

`aggregation` combines the per-criterion scores into `aggregate_score`:

| Value | Behavior |
|-------|----------|
| `WEIGHTED_MEAN` (default) | Weighted average. With the default weight of `1.0` everywhere this is a plain mean, so weights only matter when you want some criteria to count for more. |
| `MIN` | The lowest score — one failing criterion drags the row down. |
| `ALL` | `1.0` only if every criterion scores `1.0`, else `0.0`. The checklist case. |
| `NONE` | No aggregate is computed. |

Every mode ranges over the criteria that actually scored on that row, so an N/A or an
unscoreable criterion is skipped rather than counted as zero. If none scored,
`aggregate_score` is `None`. Only `WEIGHTED_MEAN` reads `weight`; setting weights under
the other modes has no effect, and the judge warns if you do.

### Constraints

- **`response_format` must be `JSON` or `XML`.** `RAW` cannot delimit one judgment per
  criterion. With `JSON`, [guided decoding](#guided-decoding) — on by default —
  guarantees that every criterion comes back and is well-formed.
- **Set `judgment_type`, `judgment_scores`, and `include_explanation` per criterion**,
  never on `judge_params` — a rubric judge rejects them there rather than ignoring them.
- **Few-shot `examples` must supply a value for every criterion**, explanation fields
  included.
- **A response the judge cannot parse yields `None` for every criterion** and no
  `aggregate_score`, with a warning naming the likely cause. The row is still returned,
  so one bad response never fails the batch. Usually the response was truncated — raise
  `max_new_tokens`. Detect these rows by checking for `None`.

## Configuration Loading

The Judge framework supports multiple ways to load configurations:

### Local File Path
```python
from oumi.judges.simple_judge import SimpleJudge

judge = SimpleJudge("./my_judge_config.yaml")
```

### Repository Path
```python
from oumi.judges.simple_judge import SimpleJudge

# Load from GitHub repository using oumi:// prefix
judge = SimpleJudge("oumi://configs/projects/judges/generic/truthfulness.yaml")
```

```python
# Load from GitHub repository using the judge's name
judge = SimpleJudge("generic/truthfulness")
```

### Programmatic Configuration
```python
from oumi.judges.simple_judge import SimpleJudge
from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.judge_params import JudgeParams
from oumi.core.configs.inference_config import InferenceConfig

judge_config = JudgeConfig(
    judge_params=JudgeParams(...),
    inference_config=InferenceConfig(...)
)

judge = SimpleJudge(judge_config)
```

## Parameter Override

You can override configuration parameters at runtime using the CLI or programmatically:

### CLI Override
```bash
oumi judge dataset \
    --config generic/truthfulness \
    --input dataset.jsonl \
    --judge_params.response_format XML
```

### Programmatic Override
```python
from oumi.core.configs.judge_config import JudgeConfig
from oumi.judges.simple_judge import SimpleJudge

judge_config = JudgeConfig.from_path("generic/truthfulness")
judge_config.judge_params.response_format = "XML"
judge = SimpleJudge(judge_config)
```
