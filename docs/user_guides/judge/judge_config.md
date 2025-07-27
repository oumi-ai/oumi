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
