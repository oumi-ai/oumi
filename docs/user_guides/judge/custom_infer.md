# Customizing Model and Generation Parameters

This level allows you to change the underlying model and its parameters to balance speed, accuracy, and resource usage.

## Core Components

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

## Examples
### 1. Fast Local Judge
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

### 2. Accurate Remote Judge
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

### 3. Using Different Models
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
