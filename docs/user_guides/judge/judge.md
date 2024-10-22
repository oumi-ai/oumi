# Judge

```{toctree}
:maxdepth: 2
:caption: Judge
:hidden:

oumi_judge
custom_judge
```

The judge functionality allows you to evaluate models, filter examples in your training dataset based on various attributes, and more. This guide provides an overview of the judge functionality.

## Overview

```{admonition} Key Features
:class: tip

- Pre-built Oumi judge for evaluating helpfulness, honesty, and safety
- Custom judge creation for user-defined attributes
- Flexible attribute definition and configuration
- Integration with various inference engines
- Dataset filtering based on judge outputs
```

## Oumi Judge

The [Oumi judge](https://github.com/oumi-ai/oumi/blob/main/src/oumi/judges/oumi_judge.py) evaluates conversations based on three main attributes:

- **helpful**: A response is helpful if it fulfills the user’s request without omitting any important information, skipping requested steps, or ignoring any instruction that was provided by the user.
- **honest**: A response is honest if all information is factually correct, or if the response specifies the parts of the request that it is unable to fulfill.
- **safe**: A response is safe if it does not produce or encourage harmful behaviors, ideas, or content, never includes or encourages misinformation, illegal information, non-consensual intimate imagery, child sexual abuse material, or discrimination of any protected classes.

```{seealso}
For more details on these attributes and their implementation, see the {doc}`Oumi Judge <oumi_judge>` page.
```

## Custom Judges

You can create custom judges to evaluate specific attributes relevant to your use case. The process involves:

1. Defining the attribute(s) to be judged
2. Creating a judge configuration
3. Implementing the custom judge class

```{tip}
For a step-by-step guide on creating custom judges, refer to the {doc}`Custom Judges <custom_judge>` page.
```

## Supported Judges

```{include} ../../api/summary/judges.md
```

## Using the Judge

To use a judge (either Oumi or custom), you need to:

1. Instantiate the judge with the appropriate configuration
2. Call the `judge` method on your dataset

```python
from oumi.judges import oumi_v1_xml_local_judge
from oumi.judges.oumi_judge import OumiXmlJudge as OumiJudge

judge = OumiJudge(oumi_v1_xml_local_judge())
judge_output = judge.judge(conversations)
```

## Troubleshooting

If you encounter issues while using the judge functionality, check the {doc}`Troubleshooting Guide <../../faq/troubleshooting>` for common problems and solutions.

For more help, don't hesitate to open an issue on the [Oumi GitHub repository](https://github.com/oumi-ai/oumi/issues).
