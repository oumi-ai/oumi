# LLM Judges

```{toctree}
:maxdepth: 2
:caption: Judge
:hidden:

judges_court
custom_judges
```

## Overview

The Oumi Judge API provides a framework for evaluating model outputs based on multiple attributes such as helpfulness, honesty, and safety.

We provide pre-built {doc}`judges <judges_court>` that you can use out of the box, or you can easily implement your own {doc}`custom judges <custom_judges>`, tailored to your specific project.

The judges API is flexible and can be run locally using a PyTorch or GGML/GGUF model, or you can use a remote API (e.g. OpenAI, Anthropic, Google, etc.).

### When to Use

The Oumi Judge API is particularly useful in the following scenarios:

- **Model Evaluation**: When you need to systematically evaluate the quality of AI model outputs across multiple dimensions
- **Custom Evaluation**: When you need specialized evaluation criteria beyond standard metrics
- **Quality Assurance**: For implementing automated quality checks in your AI deployment pipeline
- **Compare Models**: For comparing different model versions or implementations (prompts, hyper-parameters, etc.) across multiple attributes

## Quick Start

Let's start with the built-in judges, which provide a great foundation for most common evaluation needs.

Built-in judges are judges that have been tested and validated for accuracy and performance. They come with a set of attributes that are evaluated based on the model's output.

### Using Built-in Judges (Oumi Judge V1)

The Oumi Judge V1 comes in two flavors: local and remote. Let's explore both options, starting with the local implementation which is great for development and testing.

#### With a local model

```{testcode}
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

#### With a remote API

For more accurate results or when you need more powerful models, you might prefer using a remote API. Here's how to use GPT-4 as your judge:

```{testcode}
from oumi.judges import oumi_v1_xml_gpt4o_judge
from oumi.judges.oumi_judge import OumiXmlJudge

# Initialize judge with GPT-4
judge = OumiXmlJudge(oumi_v1_xml_gpt4o_judge())

# Judge conversations
results = judge.judge(conversations)
```

### Building your own judge

While our built-in judges cover many common use cases, you might have specific evaluation needs that require a custom implementation.

See {doc}`custom_judges` for more information.
