# Chat Formats

Oumi supports multiple popular input formats for representing conversations and messages for AI applications. For example, you can use the `Oumi Conversation` format, which is an extension of OpenAI's JSON format, or the `Alpaca Instruction` format, which is a simplified format that is easier to work with for single-turn instruction models.

The chat data representation is designed to be:
- **Type-Safe**: Built on pydantic models for strong type checking and validation
- **Flexible**: Supports various content types including text, images, and multimodal conversations
- **Extensible**: Easy to add metadata and custom attributes
- **Standardized**: Follows common conventions for AI conversation formats

In this guide, we will look at examples of each supported format, and how to use each format in oumi. This is essential for using your own chat data with oumi `train`, `infer`, and `judge` commands.

## Using Your Own Chat Data
In general, to use your own data with Oumi, you need to convert it into a format that can be loaded by one of the existing {py:mod}`oumi.datasets`  classes.

For chat datasets in particular, which are used for {doc}`Supervised Fine-tuning </resources/datasets/sft_datasets>`, {doc}`Preference Tuning </resources/datasets/preference_datasets>`, {doc}`Vision-Language SFT </resources/datasets/vl_sft_datasets>` training, and for all {doc}`inference tasks </user_guides/infer/infer>`, we recommend using {py:class}`~oumi.datasets.sft.sft_jsonlines.TextSftJsonLinesDataset` (registered as `text_sft`) for text-only conversations, or {py:class}`~oumi.datasets.vision_language.vision_jsonlines.VLJsonlinesDataset` (registered as `vl_sft`) for multimodal conversations.

### Individual Example Formats
::::{tab-set}
:::{tab-item} Conversation Format
The conversation format is used internally by all the SFT, Preference Tuning, and Vision-Language dataset classes. Each example is a JSON object with a list of messages:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in Seattle today?"
    },
    {
      "role": "assistant",
      "content": "I apologize, but I don't have access to real-time weather information for Seattle."
    }
  ]
}
```

This format is:
- Used by internally by Oumi for most tasks. We recommend using it by default for your chat data.
- Supports multi-turn conversations with system messages
- Can handle multimodal content (text + images)
- Allows for additional metadata
:::

:::{tab-item} Alpaca Format
The Alpaca format can be used as an alternative to the conversation format. Each example is a JSON object with instruction/input/output fields:

```json
{
  "instruction": "What's the weather like in Seattle today?",
  "input": "",
  "output": "I apologize, but I don't have access to real-time weather information for Seattle."
}
```

This format is:
- Simple and straightforward for single-turn interactions
- However, it does not support system messages or multi-turn conversations
:::
::::

### Loading Your Data
Once you have converted each individual example into a `Conversation` or `Alpaca` object, and saved it to a file (with the appropriate extension: `.jsonl` for `jsonlines`, or `.json` for `json`), you can easily load it for training or inference.

For example, let's look at a training config:

::::{tab-set}
:::{tab-item} Text Conversations
```yaml
data:
  train:
    datasets:
      - dataset_name: text_sft
        dataset_path: path/to/conversations.jsonl
```

This dataset class:
- Handles both conversation (similar to OpenAI's format) and instruction (alpaca) formats
- Supports system messages and multi-turn conversations
:::

:::{tab-item} Vision-Language
```yaml
data:
  train:
    datasets:
      - dataset_name: vl_sft
        dataset_path: path/to/multimodal.jsonl
```

This dataset class:
- Handles conversations with both text and images (similar to OpenAI's format)
- Supports multiple image formats (URL, local path, base64)
- Can process image-text pairs and multi-turn visual conversations
:::
::::

You can also load it with the python API:

```python
from oumi.datasets import TextSftJsonLinesDataset, VLJsonlinesDataset

text_dataset = TextSftJsonLinesDataset(dataset_path="path/to/conversations.jsonl")
print(text_dataset.conversation(0))  # prints the first conversation

vl_dataset = VLJsonlinesDataset(dataset_path="path/to/multimodal.jsonl")
print(vl_dataset.conversation(0))  # prints the first conversation
```

Or build examples from scratch:

```python
from oumi.core.types.conversation import Conversation, Message, Role

conversation = Conversation(
    messages=[Message(role=Role.USER, content="Hi there!")],
    metadata={"timestamp": "2025-01-01"}
)
```

## Examples

Let's looks at some examples of how these formats look in practice. You can further directly inspect small _sample_ datasets in these formats available [here](https://github.com/oumi-ai/oumi/tree/main/data/dataset_examples).

### Multi-turn with System Message

````{dropdown} Example with System Message
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant with knowledge about architecture."
    },
    {
      "role": "user",
      "content": "Tell me about the Golden Gate Bridge."
    },
    {
      "role": "assistant",
      "content": "The Golden Gate Bridge is an iconic suspension bridge in San Francisco."
    }
  ]
}
```
````

### Multimodal Conversation

````{dropdown} Example with Image and Text
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "content": "https://example.com/image_of_dog.jpg"
        },
        {
          "type": "text",
          "content": "What breed is this dog?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "This appears to be a Shih Tzu puppy."
    }
  ]
}
```
````

### Conversation with Metadata

````{dropdown} Example with Additional Metadata
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How can I make good espresso at home?"
    },
    {
      "role": "assistant",
      "content": "Here are some key tips for making espresso at home:\n1. Use freshly roasted beans\n2. Grind just before brewing\n3. Use the right pressure\n4. Maintain proper temperature"
    }
  ],
  "metadata": {
    "category": "coffee_brewing",
    "timestamp": "2025-01-11T11:22:00Z"
  }
}
```
````


## Core Data Structures

Oumi uses structured data formats implemented with pydantic models for robust type checking and validation:

### Message Format

The basic unit of conversation is the `Message` class, which represents a single message in a conversation. Key attributes include:

- `id`: Optional unique identifier for the message
- `role`: The role of the entity sending the message
- `content`: Text content of the message for text messages, or a list of `ContentItem`-s for multimodal messages e.g., an image and text content items.


```python
from oumi.core.types.conversation import Message, Role

message = Message(
    role=Role.USER,
    content="Hello, how can I help you?"
)
```

Available roles:

- `Role.SYSTEM`: System instructions
- `Role.USER`: User messages
- `Role.ASSISTANT`: AI assistant responses
- `Role.TOOL`: Tool/function calls

### Conversation

The `Conversation` class represents a sequence of messages. Key attributes include:

- `conversation_id`: Optional unique identifier for the conversation
- `messages`: List of `Message` objects that make up the conversation
- `metadata`: Optional dictionary for storing additional information about the conversation

### Content Types

For multimodal content, use `ContentItem` with appropriate types:

```python
from oumi.core.types.conversation import ContentItem, Type

# Text content
text_content = ContentItem(
    type=Type.TEXT,
    content="What's in this image?"
)

# Image content
image_content = ContentItem(
    type=Type.IMAGE_URL,
    content="https://example.com/image.jpg"
)
```

Available types:

- `Type.TEXT`: Text content
- `Type.IMAGE_PATH`: Local image path
- `Type.IMAGE_URL`: Remote image URL
- `Type.IMAGE_BINARY`: Raw image data


### ContentItem

The `ContentItem` class represents a single type part of content used in multimodal messages in a conversation. Key attributes include:

- `type`: The type of the content
- `content`: Optional text content (used for content text items, or to store image URL or path for `IMAGE_URL` and `IMAGE_PATH` content items respectively).
- `binary`: Optional binary data for the content item (used for images)

Either `content` or `binary` must be provided when creating a `ContentItem` instance.


## Working with Conversations

### Creating Conversations

```python
from oumi.core.types.conversation import Conversation, Message, Role

conversation = Conversation(
    messages=[
        Message(role=Role.USER, content="Hi there!"),
        Message(role=Role.ASSISTANT, content="Hello! How can I help?")
    ],
    metadata={"source": "customer_support"}
)
```

```python
>>> from oumi.core.types.conversation import ContentItem, Message, Role
>>> # Create a simple text message
>>> text_message = Message(role=Role.USER, content="Hello, world!")
>>> text_message.role
<Role.USER: 'user'>
>>> text_message.content
'Hello, world!'

>>> # Create an image message
>>> image_message = Message(role=Role.USER, content=[ContentItem(type=Type.IMAGE_BINARY, binary=b"image_bytes")])
>>> image_message.type
<Type.IMAGE_BINARY: 'image_binary'>

```

### Conversation Methods

```python
# Get first message of a specific role
first_user = conversation.first_message(role=Role.USER)

# Get all messages from a role
assistant_msgs = conversation.filter_messages(role=Role.ASSISTANT)

# Get the last message
last_msg = conversation.last_message()
```

```python
>>> from oumi.core.types.conversation import ContentItem, Conversation, Message, Role
>>> # Create a conversation with multiple messages
>>> conversation = Conversation(
...     messages=[
...         Message(role=Role.USER, content="Hi there!"),
...         Message(role=Role.ASSISTANT, content="Hello! How can I help?"),
...         Message(role=Role.USER, content="What's the weather?")
...     ],
...     metadata={"source": "customer_support"}
... )

>>> # Get the first user message
>>> first_user = conversation.first_message(role=Role.USER)
>>> first_user.content
'Hi there!'

>>> # Get all assistant messages
>>> assistant_msgs = conversation.filter_messages(role=Role.ASSISTANT)
>>> len(assistant_msgs)
1
>>> assistant_msgs[0].content
'Hello! How can I help?'

>>> # Get the last message
>>> last_msg = conversation.last_message()
>>> last_msg.content
"What's the weather?"

```

### Serialization

```python
# Convert to JSON
json_data = conversation.to_json()

# Load from JSON
restored = Conversation.from_json(json_data)
```

```python
>>> from oumi.core.types.conversation import ContentItem, Conversation, Message, Role
>>> # Serialize to JSON
>>> conversation = Conversation(
...     messages=[Message(role=Role.USER, content="Hello!")],
...     metadata={"timestamp": "2025-01-01"}
... )
>>> json_data = conversation.to_json()
>>> print(json_data)
{"messages":[{"content":"Hello!","role":"user"}],"metadata":{"timestamp":"2025-01-01"}}

>>> # Deserialize from JSON
>>> restored = Conversation.from_json(json_data)
>>> restored.messages[0].content
'Hello!'
>>> restored.metadata["timestamp"]
'2025-01-01'

```

## Data Validation

Oumi uses pydantic models to automatically validate:

- Message role values
- Content type consistency
- Required fields presence
- Data type correctness
