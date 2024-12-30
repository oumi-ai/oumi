# Custom Judges

## Introduction

### Why Custom Judges?

When evaluating AI model outputs, you often need to assess specific aspects of the responses that go beyond standard metrics. Custom judges in Oumi allow you to:

- Define precise evaluation criteria for your use case
- Implement domain-specific validation rules
- Create consistent evaluation frameworks across multiple models
- Automate quality assurance for AI outputs

### Common Use Cases

Custom judges are particularly valuable in scenarios such as:

1. **Code Quality Assessment**: Evaluate generated code for best practices, security, and documentation
2. **Content Moderation**: Check responses for safety, appropriateness, and adherence to guidelines
3. **Domain Expertise**: Validate technical accuracy in specialized fields like medicine or law
4. **Multi-criteria Evaluation**: Assess responses across multiple dimensions simultaneously

### Customization Levels Overview

The judge API provides three levels of customization, from simple to complex:

1. **Simple**: Modify prompts and examples
2. **Intermediate**: Configure inference engine and parameters
3. **Advanced**: Create custom judge attributes and templates

Choose the level that matches your needs - most use cases can be handled with simple customization.

## Level 1: Customizing Prompts and Examples

This is the simplest and most common way to customize a judge.

### Judge Attribute Structure

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

### Value Types

- `bool`: True/False evaluations
  Example: Code correctness, response helpfulness

- `categorical`: Multiple choice evaluations
  Example: Security levels (safe, warning, unsafe)

- `likert-5`: 5-point scale evaluations
  Example: Response quality (1=poor to 5=excellent)

### Writing System Prompts

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

### Customizing Examples

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

## Level 2: Customizing Model and Generation Parameters

This level allows you to change the underlying model and its parameters to balance speed, accuracy, and resource usage.

### Core Components

The inference configuration consists of several key parts:

1. **Model Configuration** (`ModelParams`):
   - Specifies the model to use
   - Controls model-specific settings
   - Defines trust and safety parameters

2. **Inference Engine** (`InferenceEngineType`):
   - Local: local models for speed and privacy (using vLLM, LlamaCPP, SGLang, etc.)
   - Remote: API-based models for accuracy (using OpenAI, Anthropic, Google, etc.)

3. **Generation Parameters** (`GenerationParams`):
   - Controls output generation settings
   - Sets performance constraints
   - Supports guided decoding and beam search
   - Configures sampling parameters like min_p

4. **Remote Parameters** (`RemoteParams`):
   - Manages API connections and authentication
   - Controls request rate limiting via politeness policy
   - Handles parallel processing with multiple workers

### Examples
#### 1. Fast Local Judge
```python
from oumi.core.configs import (
    GenerationParams,
    JudgeConfig,
    ModelParams,
    InferenceEngineType,
)
from oumi.judges.oumi_judge import OumiXmlJudge
from oumi.core.configs import JudgeAttribute

# Load existing attribute
judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"
helpful_attribute = JudgeAttribute.load(str(judges_directory / "helpful.json"))

# Configure for local GGUF model
local_config = JudgeConfig(
    attributes={"helpful": helpful_attribute},
    model=ModelParams(
        model_name="Qwen/Qwen2-0.5B-Instruct-GGUF",
    ),
    engine=InferenceEngineType.LLAMACPP,
    generation=GenerationParams(
        max_new_tokens=512,  # Smaller for faster processing
        temperature=0.0,
    )
)

# Initialize judge
local_judge = OumiXmlJudge(config=local_config)
```

#### 2. Accurate Remote Judge
```python
# Configure for GPT-4
remote_config = JudgeConfig(
    attributes={"helpful": helpful_attribute},
    model=ModelParams(
        model_name="gpt-4",
    ),
    engine=InferenceEngineType.REMOTE,
    generation=GenerationParams(
        max_new_tokens=2048,
        temperature=0.0,
    ),
    remote_params=RemoteParams(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key_env_varname="OPENAI_API_KEY",
        max_retries=3,
    )
)

# Initialize judge
remote_judge = OumiXmlJudge(config=remote_config)
```

#### 3. Using Different Models
```python
# Create test conversation
conversation = Conversation(messages=[
    Message(role=Role.USER, content="What is a Python decorator?"),
    Message(role=Role.ASSISTANT, content="""
    A decorator is a function that modifies another function. Here's an example:

    ```python
    def log_execution(func):
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"Finished {func.__name__}")
            return result
        return wrapper

    @log_execution
    def greet(name):
        return f"Hello, {name}!"
    ```
    """)
])

# Compare results
local_results = local_judge.judge([conversation])
remote_results = remote_judge.judge([conversation])

print("Local Judge Results:", local_results)
print("Remote Judge Results:", remote_results)
```

## Level 3: Custom Attributes and Templates

This is the most advanced level of customization, allowing you to modify how judges process inputs and outputs.

### Core Components

The judge implementation consists of several key components:

1. **Judge Configuration** (`JudgeConfig`):
   - Defines the model and inference settings
   - Contains attribute definitions
   - Configures remote API settings if applicable

2. **Judge Class** (`BaseJudge`):
   - Abstract base class for judge implementations
   - Handles core judging logic and data flow
   - Manages inference and attribute evaluation

3. **Input/Output Classes**:
   - `TemplatedMessage`: Base class for structured messages
   - `CustomJudgeInput`: Defines input format
   - `CustomJudgeOutput`: Defines output format

### Custom Judge Implementation

This example shows how to create a completely custom judge with specialized input/output formats.

#### 1. Define Custom Input/Output
```python
from typing import Optional, Dict, Any
from oumi.core.types.conversation import Role, TemplatedMessage
from oumi.judges.base_judge import BaseJudgeOutput

class CodeReviewInput(TemplatedMessage):
    """Custom input format for code review."""

    role: Role = Role.USER
    template: str = """
    <code_review>
        <file>{{ filename }}</file>
        <code>{{ code }}</code>
        <context>{{ context }}</context>
    </code_review>
    """

    filename: str
    code: str
    context: Optional[str] = None

class CodeReviewOutput(BaseJudgeOutput):
    """Custom output format for code review."""

    role: Role = Role.ASSISTANT
    template: str = """
    <evaluation>
        <best_practices>{{ best_practices }}</best_practices>
        <security>{{ security }}</security>
        <error_handling>{{ error_handling }}</error_handling>
        <documentation>{{ documentation }}</documentation>
        <overall_score>{{ overall_score }}</overall_score>
    </evaluation>
    """

    best_practices: bool
    security: bool
    error_handling: bool
    documentation: bool
    overall_score: float

    @property
    def label(self):
        """Convert to final score."""
        return self.overall_score
```

#### 2. Implement Custom Judge
```python
from oumi.judges.base_judge import BaseJudge

class CodeReviewJudge(BaseJudge):
    """Custom judge for code review."""

    def _transform_conversation_input(
        self, conversation: Conversation
    ) -> CodeReviewInput:
        """Transform conversation to code review input."""
        user_message = conversation.last_message(Role.USER)
        assistant_message = conversation.last_message(Role.ASSISTANT)

        if not all([user_message, assistant_message]):
            raise ValueError("Missing messages in conversation")

        # Extract filename from user message (assuming format: "Review file: xxx.py")
        filename = user_message.content.split(": ")[1]

        return CodeReviewInput(
            filename=filename,
            code=assistant_message.content,
            context=f"Review requested by user: {user_message.content}"
        )

    def _transform_dict_input(self, raw_input: dict[str, Any]) -> CodeReviewInput:
        """Transform dictionary to code review input."""
        return CodeReviewInput(**raw_input)

    def _transform_model_output(self, model_output) -> CodeReviewOutput:
        """Parse model output into code review format."""
        return CodeReviewOutput.from_xml_output(model_output)
```

#### 3. Create and Use Custom Judge
```python
# Create custom attribute for code review
code_review_attribute = JudgeAttribute(
    name="code_review",
    system_prompt="""
    Act as an expert code reviewer. Evaluate the provided code based on:
    1. Best Practices: Following language conventions
    2. Security: No obvious vulnerabilities
    3. Error Handling: Proper exception handling
    4. Documentation: Clear comments and docstrings

    Provide evaluations as True/False and an overall score from 0-1.
    """,
    examples=[
        {
            "template": "<code_review><file>{{ filename }}</file><code>{{ code }}</code></code_review>",
            "role": "user",
            "filename": "example.py",
            "code": "def process_data(data):\n    return data.strip().upper()"
        },
        {
            "template": "<evaluation><best_practices>{{ best_practices }}</best_practices><security>{{ security }}</security><error_handling>{{ error_handling }}</error_handling><documentation>{{ documentation }}</documentation><overall_score>{{ overall_score }}</overall_score></evaluation>",
            "role": "assistant",
            "best_practices": True,
            "security": True,
            "error_handling": False,
            "documentation": False,
            "overall_score": 0.5
        }
    ],
    value_type="float"
)

# Create judge config
judge_config = JudgeConfig(
    attributes={"code_review": code_review_attribute},
    model=ModelParams(model_name="gpt-4"),
    engine=InferenceEngineType.REMOTE,
    generation=GenerationParams(max_new_tokens=1024),
    remote_params=RemoteParams(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key_env_varname="OPENAI_API_KEY"
    )
)

# Initialize judge
code_review_judge = CodeReviewJudge(config=judge_config)

# Use the judge
conversation = Conversation(messages=[
    Message(role=Role.USER, content="Review file: data_processor.py"),
    Message(role=Role.ASSISTANT, content="""
    def process_data(filepath):
        data = {}
        with open(filepath) as f:
            data = json.load(f)
        return data
    """)
])

results = code_review_judge.judge([conversation])
print(results)
```

## API Reference

Complete documentation for key classes:

- {py:class}`~oumi.judges.base_judge.BaseJudge`: Abstract base class for judge implementations
- {py:class}`~oumi.core.configs.JudgeConfig`: Configuration container and validator
- {py:class}`~oumi.core.types.conversation.TemplatedMessage`: Base class for structured messages
- {py:class}`~oumi.core.configs.JudgeAttribute`: Defines evaluation criteria and examples

For detailed method signatures and usage examples, see the API Documentation.
