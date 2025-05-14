# Training Methods

(training-methods)=
## Introduction

Oumi supports several training methods to accommodate different use cases.

Here's a quick comparison:

| Method | Use Case | Data Required | Compute | Key Features |
|--------|----------|---------------|---------|--------------|
| [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft) | Task adaptation | Input-output pairs | Low | Fine-tunes pre-trained models on specific tasks by providing labeled conversations. |
| [Vision-Language SFT](#vision-language-sft) | Multimodal tasks | Image-text pairs | Moderate | Extends SFT to handle both images and text, enabling image understanding problems. |
| [Pretraining](#pretraining) | Domain adaptation | Raw text | Very High | Trains a language model from scratch or adapts it to a new domain using large amounts of unlabeled text. |
| [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo) | Preference learning | Preference pairs | Low | Trains a model to align with human preferences by providing pairs of preferred and rejected outputs. |

(supervised-fine-tuning-sft)=
## Supervised Fine-Tuning (SFT)

### Overview
Supervised Fine-Tuning (SFT) is the most common approach for adapting a pre-trained language model to specific downstream tasks. This involves fine-tuning the model's parameters on a labeled dataset of input-output pairs, effectively teaching the model to perform the desired task. SFT is effective for a wide range of tasks, including:


- **Question answering:** Answering questions based on given context or knowledge. This could be used to build chatbots that can answer questions about a specific domain or provide general knowledge.
- **Agent development:** Training language models to act as agents that can interact with their environment and perform tasks autonomously. This involves fine-tuning the model on data that demonstrates how to complete tasks, communicate effectively, and make decisions.
- **Tool use:** Fine-tuning models to effectively use external tools (e.g., calculators, APIs, databases) to augment their capabilities. This involves training on data that shows how to call tools, interpret their outputs, and integrate them into problem-solving.
- **Structured data extraction:** Training models to extract structured information from unstructured text. This can be used to extract entities, relationships, or key events from documents, enabling automated data analysis and knowledge base construction.
- **Text generation:** Generating coherent text, code, scripts, email, etc. based on a prompt.

### Data Format
SFT uses the {class}`~oumi.core.types.conversation.Conversation` format, which represents a conversation between a user and an assistant. Each turn in the conversation is represented by a message with a role ("user" or "assistant") and content.

```python
{
    "messages": [
        {
            "role": "user",
            "content": "What is machine learning?"
        },
        {
            "role": "assistant",
            "content": "Machine learning is a type of artificial intelligence that allows software applications to become more accurate in predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values."
        }
    ]
}
```


See {doc}`/resources/datasets/sft_datasets` for available SFT datasets.

### Configuration
The `data` section in the configuration file specifies the dataset to use for training. The `training` section defines various training parameters.

```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "/path/to/data"
        split: "train"
    collator_name: "text_with_padding"

training:
  trainer_type: "TRL_SFT"
```
See the {gh}`🔧 Model Finetuning Guide <notebooks/Oumi - Finetuning Tutorial.ipynb>` notebook for a complete example.

(vision-language-sft)=
## Vision-Language SFT

### Overview
Vision-Language SFT extends the concept of Supervised Fine-Tuning to handle both images and text. This enables the model to understand and reason about visual information, opening up a wide range of multimodal applications:

- **Image-based instruction following:** Following instructions that involve both text and images. For example, the model could be instructed to analyze and generate a report based on an image of a table.
- **Multimodal Agent Development:** Training agents that can perceive and act in the real world through vision and language. This could include tasks like navigating a physical space, interacting with objects, or following complex instructions.
- **Structured Data Extraction from Images:** Extracting structured data from images, such as tables, forms, or diagrams. This could be used to automate data entry or to extract information from scanned documents.


### Data Format
Vision-Language SFT uses the {class}`~oumi.core.types.conversation.Conversation` format with additional support for images. The `image` field contains the path to the image file.

::::{tab-set-code}
:::{code-block} JSON

{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "content": "https://oumi.ai/the_great_wave_off_kanagawa.jpg"
        },
        {
          "type": "text",
          "content": "What is in this image?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "The image is a traditional Japanese ukiyo-e print."
    }
  ]
}
:::

:::{code-block} python

from oumi.core.types.conversation import Conversation, ContentItem, Message, Role, Type

Conversation(
    messages=[
        Message(
            role=Role.USER,
            content=[
                ContentItem(
                    type=Type.IMAGE_URL,
                    content="https://oumi.ai/the_great_wave_off_kanagawa.jpg"
                ),
                ContentItem(type=Type.TEXT, content="What is in this image?"),
            ],
        ),
        Message(
            role=Role.ASSISTANT,
            content="The image is a traditional Japanese ukiyo-e print."
        )
    ]
)
:::
::::

See {doc}`/resources/datasets/vl_sft_datasets` for available vision-language datasets.

### Configuration
The configuration for Vision-Language SFT is similar to SFT, but with additional parameters for handling images.

```yaml
model:
  model_name: "meta-llama/Llama-3.2-11B-Vision-Instruct"
  chat_template: "llama3-instruct"
  freeze_layers: ["vision_encoder"]  # Freeze specific layers to only train the language model, and not the vision encoder

data:
  train:
    collator_name: "vision_language_with_padding" # Visual features collator
    datasets:
      - dataset_name: "vl_sft_jsonl"
        dataset_path: "/path/to/data"
        trust_remote_code: False # Set to true if needed for model-specific processors
        dataset_kwargs:
          processor_name: "meta-llama/Llama-3.2-11B-Vision-Instruct" # Feature generator

training:
  trainer_type: "TRL_SFT"
```

See the {gh}`🖼️ Oumi Multimodal <notebooks/Oumi - Vision Language Models.ipynb>` notebook for a complete example.

(pretraining)=
## Pretraining

### Overview
The most common pretraining method is Causal Language Modeling (CLM), where the model predicts the next token in a sequence, given the preceding tokens.
Pretraining is the process of training a language model from scratch, or continuing training on a pre-trained model, using large amounts of unlabeled text data. This is a computationally expensive process, but it can result in models with strong general language understanding capabilities.

Pretraining is typically used for:
- **Training models from scratch:** This involves training a new language model from scratch on a massive text corpus.
- **Continuing training on new data:** This involves taking a pre-trained model and continuing to train it on additional data, to improve its performance on existing tasks.
- **Domain adaptation:** This involves adapting a pre-trained model to a specific domain, such as scientific literature or legal documents.


### Data Format
Pretraining uses the {class}`~oumi.core.datasets.BasePretrainingDataset` format, which simply contains the text to be used for training.

```python
{
    "text": "Document text for pretraining..."
}
```

See {doc}`/resources/datasets/pretraining_datasets` section on pretraining datasets.

### Configuration
The configuration for pretraining specifies the dataset and the pretraining approach to use.

```yaml
data:
  train:
    datasets:
      - dataset_name: "text"
        dataset_path: "/path/to/corpus"
        streaming: true  # Stream data from disk and/or network
    pack: true  # Pack multiple documents into a single sequence
    max_length: 2048  # Maximum sequence length

training:
  trainer_type: "OUMI"
```

**Explanation of Configuration Parameters:**

* `streaming`: If set to `true`, the data will be streamed from disk, which is useful for large datasets that don't fit in memory.
* `pack`: If set to `true`, multiple documents will be packed into a single sequence, which can improve efficiency.

(direct-preference-optimization-dpo)=
## Direct Preference Optimization (DPO)

### Overview
Direct Preference Optimization (DPO) is a technique for training language models to align with human preferences. It involves presenting the model with pairs of outputs (e.g., two different responses to the same prompt) and training it to prefer the output that humans prefer. DPO offers several advantages:

- **Training with human preferences:** DPO allows you to directly incorporate human feedback into the training process, leading to models that generate more desirable outputs.
- **Improving output quality without reward models:** Unlike reinforcement learning methods, DPO doesn't require a separate reward model to evaluate the quality of outputs.


### Data Format
DPO uses the {class}`~oumi.core.datasets.BaseExperimentalDpoDataset` format, which includes the prompt, the chosen output, and the rejected output.

```python
{
    "messages": [
        {
            "role": "user",
            "content": "Write a story about a robot"
        }
    ],
    "chosen": {
        "messages": [
            {
                "role": "assistant",
                "content": "In the year 2045, a robot named..."
            }
        ]
    },
    "rejected": {
        "messages": [
            {
                "role": "assistant",
                "content": "There was this robot who..."
            }
        ]
    }
}
```

See {doc}`/resources/datasets/preference_datasets` section on preference datasets.

### Configuration
The configuration for DPO specifies the training parameters and the DPO settings.

```yaml

data:
  train:
    datasets:
      - dataset_name: "preference_pairs_jsonl"
        dataset_path: "/path/to/data"
    collator_name: "dpo_with_padding"

training:
  trainer_type: "TRL_DPO"  # Use the TRL DPO trainer

```

## Next Steps

- Explore {doc}`configuration options </user_guides/train/configuration>`
- Set up {doc}`monitoring tools </user_guides/train/monitoring>`
