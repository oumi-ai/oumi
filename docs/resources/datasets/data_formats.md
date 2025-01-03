# Supported Formats

Oumi uses a structured data format for representing conversations and messages.

This format is implemented using pydantic models, which provide type checking and data validation.

Let's look at some examples of compatible datasets and how to work with them.

## Supported File Formats

### JSONL Format

The primary format for conversation datasets is JSONL (JSON Lines). Each line contains a complete conversation:

````{dropdown} Basic Conversation Example
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
````

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

The basic unit of conversation is the `Message` class:

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

### Message

The `Message` class represents a single message in a conversation. Key attributes include:

- `id`: Optional unique identifier for the message
- `role`: The role of the entity sending the message
- `content`: Text content of the message for text messages, or a list of `ContentItem`-s for multimodal messages e.g., an image and text content items.

### ContentItem

The `ContentItem` class represents a single type part of content used in multimodal messages in a conversation. Key attributes include:

- `type`: The type of the content
- `content`: Optional text content (used for content text items, or to store image URL or path for `IMAGE_URL` and `IMAGE_PATH` content items respectively).
- `binary`: Optional binary data for the content item (used for images)

Either `content` or `binary` must be provided when creating a `ContentItem` instance.

### Conversation

The `Conversation` class represents a sequence of messages. Key attributes include:

- `conversation_id`: Optional unique identifier for the conversation
- `messages`: List of `Message` objects that make up the conversation
- `metadata`: Optional dictionary for storing additional information about the conversation

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
>>> from oumi.core.types.conversation import ContentItem, Message, Role, Type
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
