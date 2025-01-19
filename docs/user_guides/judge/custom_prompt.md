# Custom Prompts

The simplest and most common way to customize a judge is by customizing the prompts and the few-shot examples.

## Judge Attribute Structure

{py:class}`~oumi.core.configs.JudgeAttribute` instances are defined in JSON files with the following structure. Each attribute has:
- `name`: A string identifier for the attribute
- `system_prompt`: The evaluation instructions for the judge
- `examples`: A list of example evaluations
- `value_type`: The output type (can be "bool", "categorical", or "likert-5")

```json
{
    "name": "string",
    "system_prompt": "string",
    "examples": [],
    "value_type": "string"
}
```

## Value Types

- `bool`: True/False evaluations
  Example: Code correctness, response helpfulness

- `categorical`: Multiple choice evaluations
  Example: Security levels (safe, warning, unsafe)

- `likert-5`: 5-point scale evaluations
  Example: Response quality (1=poor to 5=excellent)

## Writing System Prompts

When writing system prompts, follow these best practices:

1. Be explicit about evaluation criteria
2. Use clear, objective language
3. Include specific examples of what to look for
4. Define terms when necessary

Example of a good system prompt:

```text
Act as an expert code reviewer. Evaluate the code response based on:
1. Best Practices: Does it follow language conventions and patterns?
2. Security: Are there any security vulnerabilities?
3. Error Handling: Does it handle errors appropriately?
4. Documentation: Is it properly documented?

Provide your evaluation in XML format with explanations and True/False judgments.
```

## Customizing Examples

Examples are stored in JSON format. Each example should include:
- A user request
- An AI response
- The expected evaluation

```json
{
    "examples": [
        {
            "request": "Write a function to read a file safely",
            "response": "def read_file(filepath: str) -> str:\n    try:\n        with open(filepath, 'r') as f:\n            return f.read()\n    except (FileNotFoundError, PermissionError) as e:\n        raise e",
            "evaluation": {
                "label": true,
                "explanation": "The code follows best practices with proper error handling"
            }
        }
    ]
}
```

Here's a quick example of using a custom judge:

```python
from oumi.core.configs import JudgeConfig, JudgeAttribute
from oumi.judges.oumi_judge import OumiXmlJudge
from oumi.core.types.conversation import Conversation, Message, Role

# Load custom attribute
custom_attribute = JudgeAttribute.load("judges/helpful.json")

# Create judge config
judge_config = JudgeConfig(
    attributes={"helpful": custom_attribute},
    model=ModelParams(model_name="gpt-4")
)

# Initialize judge
judge = OumiXmlJudge(config=judge_config)

# Create conversation to evaluate
conversation = Conversation(messages=[
    Message(role=Role.USER, content="How do I handle exceptions in Python?"),
    Message(role=Role.ASSISTANT, content="Here's an example...")
])

# Get evaluation
results = judge.judge([conversation])
print(results)
```
