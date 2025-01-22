# Custom Prompts


This section discusses how to customize Oumi's judge by defining personalized judgment prompts and incorporating few-shot examples. To achieve this, you must define a {py:class}`~oumi.core.configs.JudgeAttribute` for each dimension you wish to evaluate (e.g., instruction-following, safety, code correctness, or any domain-specific criteria). Additionally, you can specify the type of judgment the judge will provide (`bool`, `categorical`, or `likert-5`) for each attribute.

## Judge Attribute

A {py:class}`~oumi.core.configs.JudgeAttribute` can be defined either directly in Python or [imported from a JSON file](judge_attribute_json_import_link). It consists of the following data members:

- `name` (`string`): A name that serves as a unique identifier for the attribute.
- `system_prompt` (`string`): The system instruction that guides the judgment process.
- `examples` (`list`[{py:class}`~oumi.core.types.conversation.TemplatedMessage`]): A list of few-shot examples illustrating the expected judgment behavior.
- `value_type` ({py:class}`~oumi.core.configs.JudgeAttributeValueType`): The type of judgment output (`bool`, `categorical`, `likert-5`).

### System Prompt

When crafting system prompts, it's important to follow these best practices to ensure clear and effective evaluation:

1. **Be explicit about evaluation criteria**: Clearly outline the aspects or dimensions to be assessed.
2. **Use clear, objective language**: Ensure the prompt is unambiguous and easily interpretable by a language model.
3. **Provide specific examples**: Include concrete examples of what to look for to guide the evaluation.
4. **Define terms when necessary**: Clarify any technical or domain-specific terminology to avoid misinterpretation.

Example of a good system prompt:

```text
Act as an expert code reviewer. Evaluate the code response based on the following criteria:

1. Best Practices: Does the code adhere to established language conventions and patterns?
2. Security: Are there any security vulnerabilities or weaknesses?
3. Error Handling: Does the code appropriately handle potential errors?
4. Documentation: Is the code properly documented with clear explanations?

Provide your evaluation in XML format, including an explanation and a corresponding True/False judgment that indicates whether the response meets all of the evaluation criteria listed above.
```

### Examples

Few-shot examples play a critical role in guiding the judge's assessment process, not only by demonstrating how to evaluate inputs, but also by defining the expected response format.

```{tip} **Oumi recommendations**
- **XML format**: Based on our extensive experimentation, we have found that most language models (including GPTs, LLaMAs, and Gemini) are more likely to adhere to `XML` format when presented with a few examples illustrating the expected structure. While you can structure the judge’s input and output in any format you prefer, we strongly recommend using `XML` for optimal results.
- **Include explanations**: Depending on your specific use case, you may or may not require the judge to provide an explanation for its judgment. However, our research suggests that prompting the judge to generate an explanation, even when not strictly necessary, encourages more thoughtful reasoning. This typically results in more reliable and consistent judgments.
```

The table below outlines the supported input and output formats when judging an attribute.
|                  | Base (Flexible) | Recommended |
|------------------|-----------------|-------------|
Input class  | {py:class}`~oumi.core.types.conversation.TemplatedMessage` | {py:class}`~oumi.judges.oumi_judge.OumiJudgeInput`
Output class | {py:class}`~oumi.judges.base_judge.BaseJudgeOutput` | {py:class}`~oumi.judges.oumi_judge.OumiJudgeOutput`

Here's a simplified code snippet that demonstrates how to define few-shot examples with the recommended classes:
```
from oumi.judges.oumi_judge import OumiJudgeInput, OumiJudgeOutput

examples = [
    # 1st example
    OumiJudgeInput(
        request="Display all integer numbers from 0 to 100",
        response="for i in range(100): print(i)",
    ),
    OumiJudgeOutput(
        judgement="True",
        explanation="The code is incorrect since it displays numbers from 0 to 99, but does not display 100",
    ),
    # 2nd example
    ...
]
```

### Value Type

Our judge supports the following judgment value types ({py:class}`~oumi.core.configs.JudgeAttributeValueType`):

| Value Type | Description | Example |
|------------|-------------|---------|
| `bool`        | True/False evaluations      | Code correctness, response helpfulness   |
| `categorical` | Multiple choice evaluations | Security levels (safe, warning, unsafe)  |
| `likert-5`    | 5-point scale evaluations   | Response quality (1=poor to 5=excellent) |

### Putting Everything Together

After defining the system prompt, examples, and selecting the value type, you can create a  {py:class}`~oumi.core.configs.JudgeAttribute` as shown below:

```python
from oumi.core.configs.judge_config import JudgeAttribute, JudgeAttributeValueType

my_judge_attribute = JudgeAttribute(
    name="my_judge_attribute",
    system_prompt=system_prompt,
    examples=examples,
    value_type=JudgeAttributeValueType.BOOL,
)
```

Alternatively, you can load the attribute from a `JSON` file as follows (see {gh}`sample file <src/oumi/judges/oumi_v1/helpful.json>`):

(judge_attribute_json_import_link)=
```python
my_judge_attribute = JudgeAttribute.load("./my_judge_attribute.json")
```

## Custom Judge

### Definition

Once you have defined all the necessary attributes for your judge, the next step is to create a {py:class}`~oumi.core.configs.JudgeConfig`. This configuration will specify the attributes to be evaluated, as well as the underlying model through {py:class}`~oumi.core.configs.ModelParams`. For more information on configuring the model, please refer to the {doc}`Custom Model </user_guides/judge/custom_infer>` page.

```python
from oumi.core.configs import JudgeConfig, ModelParams
from oumi.judges.oumi_judge import OumiXmlJudge

# Create the judge's configuration file.
my_judge_config = JudgeConfig(
    attributes={"my_judge_attribute": my_judge_attribute},
    model=ModelParams(model_name="meta-llama/Llama-3.2-3B-Instruct")
)

# Instantiate the judge.
judge = OumiXmlJudge(config=judge_config)
```

### Usage

You can now execute judgments using the following code snippet:

```python
# Create examples to evaluate.
dataset = [
    OumiJudgeInput(
        request="Generate code to test a module that consists of ...",
        response="def test_module(..){...}",
    ),
    ...
]

# Evaluate.
results = judge.judge(dataset)
```

For overview of the `results` structure, please refer to [judge quickstart](judge_quickstart_link).
