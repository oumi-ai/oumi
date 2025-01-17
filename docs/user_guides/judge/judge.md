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

## Overview

In LLM-based evaluations, an **LLM Judge** is utilized to assess the performance of a language **Language Model** according to a predefined set of criteria.

The evaluation process is carried out in two distinct steps:

- Step 1 (**Inference**): In the first step, the LLM generates responses to a series of evaluation prompts. These responses demonstrate the model's ability to interpret the prompt and generate a contextually relevant high-quality response.
- Step 2 (**Judgments)**: In the second step, the LLM Judge evaluates the quality of the generated responses. The result is a set of judgments that quantify the model's performance, according to the specified evaluation criteria.

The diagram below illustrates these two steps:

**IMAGE WILL BE ADDED HERE** ![Judge Figure](./figures/judge_figure.svg)

Oumi offers flexible APIs for both {doc}`Inference </user_guides/infer/infer>` and Judgement ("LLM Judge" API).

## When to Use?

Our LLM Judge API is fully customizable and can be applied across a wide range of evaluation scenarios, including:

- **Model Evaluation**: Systematically assessing model outputs and evaluating performance across multiple dimensions.
- **Custom Evaluation**: Tailoring the evaluation process to your specific needs by defining custom criteria, extending beyond standard metrics to address specialized requirements.
- **Dataset Filtering**: Filtering high-quality examples from noisy or inconsistent training datasets, ensuring cleaner data for model training and validation.
- **Quality Assurance**: Automating quality checks in your AI deployment pipeline, ensuring that deployed models meet predefined performance and safety standards.
- **Compare Models**: Comparing different model versions or configurations (e.g., prompts, hyperparameters) across various attributes, enabling more informed decision-making and optimization.


## Oumi Offerrings

Oumi offers a {doc}`Built-In Judge </user_guides/judge/built_in_judge>` that you can use out of the box, which evaluates model outputs based on multiple attributes such as helpfulness, honesty, and safety. Alternatively, you can tailor the judge to your specific project by customizing the {doc}`model prompts </user_guides/judge/custom_prompt>` or the {doc}`judge model and its generation parameters </user_guides/judge/custom_infer>`.

### Built-In Judge

Our built-in judge has been tested and validated for accuracy and performance. It comes with a pre-defined set of attributes, which can be easily customized. The underlying model can be either local (using a PyTorch or GGML/GGUF model) or we can call a remote API (e.g. OpenAI, Anthropic, Google, etc.). Let's explore both options, starting with the local implementation which is great for development and testing.

##### Quick Start with a local model

```{testcode} python
:skipif: True
from oumi.core.types import Conversation, Message, Role
from oumi.judges import OumiXmlJudge, oumi_v1_xml_local_judge

# Initialize the judge with local GGUF model
judge = OumiXmlJudge(oumi_v1_xml_local_judge())

# Judge conversations
conversations = [
    Conversation(messages=[
      Message(role=Role.USER, content="What is Python?"),
      Message(role=Role.ASSISTANT, content="Python is a high-level programming language.")
   ])
]

results = judge.judge(conversations)
```

##### Quick Start with a remote API

For more accurate results or when you need more powerful models, you might prefer using a remote API. Here's how to use GPT-4 as your judge:

```{testcode} python
:skipif: True
from oumi.core.types import Conversation, Message, Role
from oumi.judges import oumi_v1_xml_gpt4o_judge
from oumi.judges.oumi_judge import OumiXmlJudge

# Initialize judge with GPT-4
judge = OumiXmlJudge(oumi_v1_xml_gpt4o_judge())

# Judge conversations
conversations = [
    Conversation(messages=[
      Message(role=Role.USER, content="What is Python?"),
      Message(role=Role.ASSISTANT, content="Python is a high-level programming language.")
   ])
]

# Judge conversations
results = judge.judge(conversations)
```

### Custom Judges

When evaluating AI model outputs, you often need to assess specific aspects of the responses that go beyond standard metrics. Custom judges in Oumi allow you to:

- Define precise evaluation criteria for your use case
- Implement domain-specific validation rules
- Create consistent evaluation frameworks across multiple models
- Automate quality assurance for AI outputs

#### Common Use Cases

Custom judges are particularly valuable in scenarios such as:

1. **Code Quality Assessment**: Evaluate generated code for best practices, security, and documentation
2. **Content Moderation**: Check responses for safety, appropriateness, and adherence to guidelines
3. **Domain Expertise**: Validate technical accuracy in specialized fields like medicine or law
4. **Multi-criteria Evaluation**: Assess responses across multiple dimensions simultaneously

#### Customization Levels Overview

The judge API provides two levels of customization:

1. {doc}`Modify prompts and examples </user_guides/judge/custom_prompt>`
2. {doc}`Configure inference engine and parameters </user_guides/judge/custom_infer>`

Choose the level that matches your needs.

<!--

## API Reference

Complete documentation for key classes:

- {py:class}`~oumi.judges.base_judge.BaseJudge`: Abstract base class for judge implementations
- {py:class}`~oumi.core.configs.JudgeConfig`: Configuration container and validator
- {py:class}`~oumi.core.types.conversation.TemplatedMessage`: Base class for structured messages
- {py:class}`~oumi.core.configs.JudgeAttribute`: Defines evaluation criteria and examples

For detailed method signatures and usage examples, see the API Documentation.

-->
