# LLM Judge

```{toctree}
:maxdepth: 2
:caption: Judge
:hidden:

built_in_judge
custom_prompt
custom_infer
```

As Large Language Models (LLMs) continue to evolve, traditional evaluation benchmarks, which focus primarily on task-specific metrics, are increasingly inadequate for capturing the full scope of a model's generative potential. In real-world applications, LLM capabilities such as creativity, coherence, and the ability to effectively handle nuanced and open-ended queries are critical and cannot be fully assessed through standardized metrics alone. While human raters are often employed to evaluate these aspects, the process is costly and time-consuming. As a result, the use of LLM-based evaluation systems, or "LLM judges", has gained traction as a more scalable and efficient alternative.

Oumi provides a versatile LLM Judge framework that enables the automation of pointwise and pairwise **model evaluations**, **dataset curation**, and **quality assurance** for model deployment. You can easily customize the {doc}`evaluation prompts and criteria </user_guides/judge/custom_prompt>`, select {doc}`any underlying judge LLM </user_guides/judge/custom_infer>` (open-source or proprietary), and locally host or access it remotely via an API.

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


## Oumi Offerrings

Oumi offers a {doc}`Built-In Judge </user_guides/judge/built_in_judge>` that you can use out-of-the-box. Alternatively, you can tailor the judge to your specific needs by customizing the {doc}`judgment prompts </user_guides/judge/custom_prompt>` or the {doc}`underlying judge model </user_guides/judge/custom_infer>` and its parameters.

### Built-In Judge

Our {doc}`Built-In Judge </user_guides/judge/built_in_judge>` evaluates model outputs across multiple key attributes. By default, it assesses outputs based on three essential dimensions: helpfulness, honesty, and safety. These attributes have been rigorously tested and validated to ensure strong alignment with human judgment and consistent performance in the evaluation of AI-generated content. The selection of these attributes has been carefully considered for their pivotal role in assessing the quality, trustworthiness, and ethical integrity of model outputs, ensuring they meet the highest standards for responsible real-world applications. However, the system is fully customizable, allowing you to {doc}`customize </user_guides/judge/custom_prompt>` attributes to better suit your specific project requirements.

A built-in judge is instantiated using a configuration class, {py:class}`~oumi.core.configs.JudgeConfig`. A selection of standard configurations is available on our {gh}`judge court <src/oumi/judges/judge_court.py>`. Depending on the desired capabilities for the underlying judge model, you can choose between local configurations ({py:func}`oumi_v1_xml_local_judge <oumi.judges.oumi_v1_xml_local_judge>`) or access more powerful models via a remote API, such as GPT-4 ({py:func}`oumi_v1_xml_local_judge <oumi.judges.oumi_v1_xml_gpt4o_judge>`) or Sonnet ({py:func}`oumi_v1_xml_local_judge <oumi.judges.oumi_v1_xml_claude_sonnet_judge>`).


(judge_quickstart_link)=
##### Quick Start

```python
from oumi.core.types import Conversation, Message, Role
from oumi.judges import OumiXmlJudge
from oumi.judges import oumi_v1_xml_local_judge as judge_local
from oumi.judges import oumi_v1_xml_gpt4o_judge as judge_gpt4o
from oumi.judges import oumi_v1_xml_claude_sonnet_judge as judge_sonnet
from oumi.judges import oumi_v1_xml_deepseek_r1_judge as judge_deepseek_r1

# Instantiate the judge.
judge = OumiXmlJudge(judge_local()) # alternatives: judge_gpt4o(), judge_sonnet(), judge_deepseek_r1()

# Define the `conversations` to be judged.
conversations = [
    Conversation(messages=[
      Message(role=Role.USER, content="What is Python?"),
      Message(role=Role.ASSISTANT, content="Python is a high-level programming language.")
   ])
]

results = judge.judge(conversations)
```

The `results` variable is a dictionary, where each key corresponds to an attribute name (`helpful`, `honest`, `safe`). The associated values include a `judgement` ("Yes" if the response meets the criteria, "No" otherwise) and an `explanation` provided by the judge model. For example, the result for `helpful` is represented as follows:

```
"helpful": {
   "fields": {
      "judgement": "Yes",
      "explanation": "The response is helpful because it provides a brief explanation of what Python is."
   },
   "label": True
}
```

### Custom Judge

Custom judges offer significant value in a variety of specialized scenarios, such as:

1. **Code Quality Assessment**: Evaluate generated code for adherence to best practices, security standards, and proper documentation.
2. **Content Moderation**: Assess responses for safety, appropriateness, and compliance with established guidelines.
3. **Domain Expertise**:  Ensure technical accuracy and precision in specialized fields such as medicine, law, or engineering.
4. **Multi-Criteria Evaluation**: Conduct comprehensive assessments of responses across multiple dimensions simultaneously.

This section provides an overview of the available customization options.

#### Customization Options

The LLM Judge framework offers a range of customization options to tailor the evaluation process to your specific needs. You can modify the judgment prompts and their corresponding few-shot examples, as well as choose the type of judgment the underlying model will provide (`bool`, `categorical`, or `likert-5`). For a comprehensive guide on these options, refer to the {doc}`Custom Prompts </user_guides/judge/custom_prompt>` page.

Additionally, you have the flexibility to select and configure the underlying judge model, allowing you to optimize for speed, accuracy, and resource efficiency. Models can be loaded from a local path (or downloaded from HuggingFace) and hosted locally, or you can choose from a variety of popular remote models (from providers such as OpenAI, Anthropic, and Google) by specifying the appropriate {py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`. Furthermore, all model ({py:class}`~oumi.core.configs.params.model_params.ModelParams`) and generation ({py:class}`~oumi.core.configs.params.generation_params.GenerationParams`) parameters are fully adjustable to suit your requirements. Detailed information on these configuration options can be found on the {doc}`Custom Model </user_guides/judge/custom_infer>` page.
