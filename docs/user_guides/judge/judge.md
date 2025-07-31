# LLM Judge

```{toctree}
:maxdepth: 2
:caption: Judge
:hidden:

judge_config
built_in_judges
cli_usage
```

As Large Language Models (LLMs) continue to evolve, traditional evaluation benchmarks, which focus primarily on task-specific metrics, are increasingly inadequate for capturing the full scope of a model's generative potential. In real-world applications, LLM capabilities such as creativity, coherence, and the ability to effectively handle nuanced and open-ended queries are critical and cannot be fully assessed through standardized metrics alone. While human raters are often employed to evaluate these aspects, the process is costly and time-consuming. As a result, the use of LLM-based evaluation systems, or "LLM judges", has gained traction as a more scalable and efficient alternative.

Oumi provides a versatile LLM Judge framework that enables the automation of pointwise and pairwise **model evaluations**, **dataset curation**, and **quality assurance** for model deployment. You can easily customize the evaluation prompts and criteria, select any underlying judge LLM (open-source or proprietary), and locally host or access it remotely via an API.

## Overview

In LLM-based evaluations, an **LLM Judge** is utilized to assess the performance of a **Language Model** according to a predefined set of criteria.

The evaluation process is carried out in two distinct steps:

- Step 1 (**Inference**): In the first step, the language model generates responses to a series of evaluation prompts. These responses demonstrate the model's ability to interpret the prompt and generate a contextually relevant high-quality response.
- Step 2 (**Judgments)**: In the second step, the LLM Judge evaluates the quality of the generated responses. The result is a set of judgments that quantify the model's performance, according to the specified evaluation criteria.

The diagram below illustrates these two steps:
![Judge Figure](/_static/judge/judge_figure.svg)

Oumi offers flexible APIs for both {doc}`Inference </user_guides/infer/infer>` and Judgement ("LLM Judge" API).

## When to Use?

Our LLM Judge API is fully customizable and can be applied across a wide range of evaluation scenarios, including:

- **Model Evaluation**: Systematically assessing model outputs and evaluating performance across multiple dimensions.
- **Custom Evaluation**: Tailoring the evaluation process to your specific needs by defining custom criteria, extending beyond standard metrics to address specialized requirements.
- **Dataset Filtering**: Filtering high-quality examples from noisy or inconsistent training datasets, ensuring cleaner data for model training and validation.
- **Quality Assurance**: Automating quality checks in your AI deployment pipeline, ensuring that deployed models meet predefined performance and safety standards.
- **Compare Models**: Comparing different model versions or configurations (e.g., prompts, hyperparameters) across various attributes, enabling more informed decision-making and optimization.

## Quick Start

To leverage an LLM judge, we instantiate a {py:class}`~oumi.judges.simple_judge.SimpleJudge` class using a judge configuration ({py:class}`~oumi.core.configs.judge_config.JudgeConfig`). Oumi offers several {doc}`built-in judges </user_guides/judge/built_in_judges>` that you can use out-of-the-box for common evaluation tasks (see our {gh}`GitHub page <configs/projects/judges/>` for the complete list). Alternatively, you can create completely custom judges by defining your own prompts, output formats, and evaluation criteria in a {doc}`custom config </user_guides/judge/judge_config>`.

### Built-In Judges (predefined configuration)

Select a judge from our {gh}`GitHub page <configs/projects/judges/>` and use it, as shown below. The following example is for the {gh}`Truthfulness judge <configs/projects/judges/generic/truthfulness.yaml>` (judges/generic/truthfulness).
```python
from oumi.judges.simple_judge import SimpleJudge

# Instantiate a simple judge using a predefined configuration: Truthfulness
simple_judge = SimpleJudge(judge_config="oumi://configs/projects/judges/generic/truthfulness.yaml")

# Define a dataset
dataset = [
    {
        "request": "What is the capital of France?",
        "response": "Rome",
    }
]

# Evaluate the dataset
outputs = simple_judge.judge(dataset)

# Retrieve the judgement(s) and explanation(s) from the structured output(s)
for output in outputs:
    judgment = output.field_values["judgment"]  # False
    explanation = output.field_values["explanation"]  # The correct answer is Paris.
```

### Custom Judge

Step 1: Define your custom configuration, such as `./my_judge_config.yaml`.
```yaml
judge_params:
  prompt_template: |
    You are a truthfulness judge. Determine whether the response below is factually accurate.
    If the response is truthful, respond with 'Yes'. Otherwise, respond with 'No'.
    ***
    [request]:
    {request}
    ***
    [response]:
    {response}
    ***

  response_format: JSON
  judgment_type: BOOL
  include_explanation: True

inference_config:
  model:
    model_name: "gpt-4o"

  engine: OPENAI

  generation:
    max_new_tokens: 8192
    temperature: 1.0
```

Step 2: Load your custom configuration file and evaluate a dataset.
```python
from oumi.judges.simple_judge import SimpleJudge

# Instantiate a simple judge using a local configuration
simple_judge = SimpleJudge(judge_config="./my_judge_config.yaml")

# Define a dataset
dataset = [
    {
        "request": "What is the capital of France?",
        "response": "Rome",
    }
]

# Evaluate the dataset
outputs = simple_judge.judge(dataset)

# Retrieve the judgement(s) and explanation(s) from the structured output(s)
for output in outputs:
    judgment = output.field_values["judgment"]  # False
    explanation = output.field_values["explanation"]  # The correct answer is Paris.
```

## Next Steps
- Explore our {doc}`Built-In Judges </user_guides/judge/built_in_judges>` for out-of-the-box evaluation criteria
- Understand the {doc}`Judge Configuration </user_guides/judge/judge_config>` options
- Explore {doc}`CLI usage </user_guides/judge/cli_usage>` for command-line evaluation
