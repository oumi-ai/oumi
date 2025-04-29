# Custom Evaluations

With Oumi, custom evaluations are effortless and powerful, giving you complete control over how model performance is assessed. Whether you're working with open- or closed-source models, setup is simple: just configure a few settings, no code changes required. Provide your dataset, select your models, and register an evaluation function tailored to the metrics that matter most to you, from accuracy and consistency to bias or domain-specific goals. Oumi handles the rest, including running inference, so you can focus on gaining insights, not managing infrastructure.

## Custom Evaluations Step-by-Step

Running custom evaluation involves three simple steps. First, define the evaluation configuration using a `YAML` file. Next, register your custom evaluation function to compute the metrics that matter to you. Finally, execute the evaluation using Oumi's {py:class}`~oumi.core.evaluation.Evaluator`, which orchestrates the entire process.

### Step 1: Defining Evaluation Configuration

The evaluation configuration is defined in a `YAML` file and parsed into a {py:class}`~oumi.core.configs.EvaluationTaskParams` object. Below is a simple example for evaluating GPT-4o. You can evaluate most open models (Llama, DeepSeek, Qwen, Phi, and others), closed models (Gemini, Claude, OpenAI), and cloud-hosted models (Vertex AI, Together, SambaNova, etc.) by simply updating the `model_name` and `inference_engine` fields. Example configurations for popular APIs are available at {gh}`src/oumi/configs/apis/`.

For custom evaluations, set `evaluation_backend` to `custom`, and assign `task_name` to the name of your registered custom evaluation function (see Step 2). For more details on setting the configuration file for evaluations, including custom models, refer to our {doc}`documentation </user_guides/evaluate/evaluation_config>`.

```yaml
model:
  model_name: "gpt-4o"

inference_engine: OPENAI

generation:
  max_new_tokens: 8192
  temperature: 0.0

tasks:
  - evaluation_backend: custom
    task_name: my_custom_evaluation
```

### Step 2: Defining Custom Evaluation Function

To define a custom evaluation function, simply register a Python function using the `@register_evaluation_function` decorator. Your function can optionally accept any of the reserved parameters below, depending on your needs:

- `config` ({py:class}`~oumi.core.configs.EvaluationConfig`): The full evaluation configuration defined in Step 1. Include this if you need access to platform-level settings or variables.
- `task_params` ({py:class}`~oumi.core.configs.EvaluationTaskParams`): Represents a specific evaluation task from the `YAML` file. If your configuration defines multiple tasks under `tasks`, this parameter will contain the metadata for the one currently being evaluated.
- `inference_engine` ({py:class}`~oumi.core.inference.BaseInferenceEngine`): An automatically generated engine for the model specified in the evaluation configuration (by `model_name`). Use its {py:obj}`infer() <oumi.core.inference.BaseInferenceEngine.infer>` method to run inference on a list of examples formatted as {class}`~oumi.core.types.conversation.Conversation`.
- User-defined inputs (e.g. `my_input`): You may also include any number of additional parameters of any type. These are passed in during execution (see Step 3).

```python
from oumi.core.registry import register_evaluation_function
from oumi.core.configs import EvaluationConfig, EvaluationTaskParams
from oumi.core.inference import BaseInferenceEngine

@register_evaluation_function("my_custom_evaluation")
def my_custom_evaluation(
    config: EvaluationConfig,
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    my_input
) -> dict[str, Any]
```

Your custom evaluation function is expected to return a dictionary where each key is a metric name and each value is the corresponding computed result.

### Step 3: Executing the Evaluation

Once you've defined your `YAML` configuration and registered the custom evaluation function (as specified by the `task_name` in your config), you can run the evaluation using the code snippet below.

The {py:class}`~oumi.core.evaluation.Evaluator`'s `evaluate` method requires the evaluation configuration ({py:class}`~oumi.core.configs.EvaluationConfig`) and also supports any number of user-defined variables passed as keyword arguments (e.g., `my_input` in the example below). These variable names must exactly match the parameters defined in your custom evaluation function's signature. Otherwise, a runtime error will occur.

The `evaluate` method returns a list of {py:class}`~oumi.core.evaluation.evaluation_result.EvaluationResult` objects, one for each task defined in the `tasks` section of your `YAML` file. Each result includes the dictionary returned by your evaluation function (`task_result`), along with useful metadata such as `start_time`, `elapsed_time_sec`, and more.

```python
from oumi.core.configs import EvaluationConfig
from oumi.core.evaluation import Evaluator

config = EvaluationConfig.from_yaml(<path/to/yaml/file>)
results = Evaluator().evaluate(config, my_input=<user_input>)
```

## Walk-through Example

This section walks through a simple example to demonstrate how to use custom evaluations in practice. Suppose you want to assess response verbosity (i.e., the average length of a model's responses, measured in number of characters) across multiple models.

To do this, assume you’ve prepared a dataset of user queries. A toy dataset (`my_conversations`) with two examples is shown below, formatted as a list of {class}`~oumi.core.types.conversation.Conversation` objects.

```python
from oumi.core.types.conversation import Conversation, Message, Role

my_conversations = [
    Conversation(
        messages=[
            Message(role=Role.USER, content="Hello there!"),
        ]
    ),
    Conversation(
        messages=[
            Message(role=Role.USER, content="How are you?"),
        ]
    ),
]
```

### Step 1: Defining the Evaluation Configuration

Start by defining a `YAML` configuration for each model you want to evaluate. The configuration specifies the model, inference engine, and links to the custom evaluation function via the `task_name`.

```python
gpt_4o_config = """
  model:
    model_name: "gpt-4o"

  inference_engine: OPENAI

  tasks:
  - evaluation_backend: custom
    task_name: model_verboseness_evaluation
"""
```

### Step 2: Defining Custom Evaluation Function

Next, define the evaluation function. Start by using the provided `inference_engine` to run inference and generate model responses. For each example, the engine appends a response (i.e., a {class}`~oumi.core.types.conversation.Message` with role {py:obj}`~oumi.core.types.conversation.Role`=`ASSISTANT`) to the end of the corresponding {class}`~oumi.core.types.conversation.Conversation`.

You can retrieve the model response from each `conversation` using the `last_message()` method, then compute the average character length across all responses, as shown in the example below.

```python
from oumi.core.registry import register_evaluation_function

@register_evaluation_function("model_verboseness_evaluation")
def model_verboseness_evaluation(inference_engine, conversations):
    # Run inference to generate the model responses.
    conversations = inference_engine.infer(conversations)

    aggregate_response_length = 0
    for conversation in conversations:
        # Extract the assistant's (model's) response from the conversation.
        response = conversation.last_message().content

        # Update the sum of lengths for all model responses.
        aggregate_response_length += len(response)

    return {"average_response_length": aggregate_response_length / len(conversations)}
```

### Step 3: Executing the Evaluation

Finally, run the evaluation using the code snippet below. This will execute inference and compute the verbosity metric based on your custom evaluation function.

```python
from oumi.core.configs import EvaluationConfig
from oumi.core.evaluation import Evaluator

config = EvaluationConfig.from_str(gpt_4o_config)
results = Evaluator().evaluate(config, conversations=my_conversations)
```


The response length can be retrieved from `results` as shown below. Since this walkthrough assumes a single task defined in the `tasks` section of the `YAML` config, we only examine the first item in the `results` list.

```python
result_dict = results[0].get_results()
print(f"Average length: {result_dict['average_response_length']}")
```
