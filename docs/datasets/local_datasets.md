# Supported Formats

Oumi uses a structured data format for representing conversations and messages.

This format is implemented using pydantic models, which provide type checking and data validation.

Let's look at some examples of compatible datasets and how to work with them.

## JSONL Datasets

Oumi can work with JSONL (JSON Lines) datasets that follow the structure defined by the `Conversation` and `Message` classes.

### Example 1: Simple Text Conversation

```json
{"messages": [
    {"role": "user", "content": "What's the weather like in Seattle today?"},
    {"role": "assistant", "content": "I apologize, but I don't have access to real-time weather information for Seattle."},
    {"role": "user", "content": "I see. Can you tell me about Seattle's typical weather patterns?"},
    {"role": "assistant", "content": "Certainly! Seattle is known for its mild, but wet climate."}
]}
```

### Example 2: Multi-turn Conversation with System Message

```json
{"messages": [
    {"role": "system", "content": "You are a helpful AI assistant with knowledge about architecture."},
    {"role": "user", "content": "Hi, can you tell me about the Golden Gate Bridge?"},
    {"role": "assistant", "content": "Sure! The Golden Gate Bridge is an iconic suspension bridge in San Francisco."},
    {"role": "user", "content": "How long is the Golden Gate Bridge?"},
]}
```

### Example 3: Conversation with an Image

```json
{"messages": [
    {"role": "user", "type": "image_url", "content": "https://example.com/image_of_dog.jpg"},
    {"role": "user", "content": "What breed is this dog?"},
    {"role": "assistant", "content": "Thank you for the image. This appears to be a Shih Tzu puppy."}
]}
```

### Example 4: Conversation with Metadata

```json
{
    "messages": [
    {"role": "user", "content": "How can I make good espresso at home?"},
    {"role": "assistant", "content": "Making good espresso at home requires some equipment and practice, but here are some key tips:\n\n1. Use freshly roasted, high-quality coffee beans.\n2. Grind the beans just before brewing, using a fine, consistent grind.\n3. Use the right amount of coffee - typically 18-20 grams for a double shot.\n4. Tamp the grounds evenly with about 30 pounds of pressure.\n5. Ensure your espresso machine is clean and at the correct temperature (usually around 200°F).\n6. Aim for an extraction time of 25-30 seconds for a 1-2 oz shot.\n7. Look for a golden-brown crema on top of your espresso.\n8. Experiment and adjust based on taste - espresso making is part science, part art!"}
    ],
    "metadata": {"category": "coffee_brewing", "timestamp": "2024-05-11T11:22:00Z"}
}
```

## Python API

### Role and Type Enums

```python
>>> from oumi.core.types.conversation import Role, Type
>>> list(Role)  # Show all available roles
[<Role.SYSTEM: 'system'>, <Role.USER: 'user'>, <Role.ASSISTANT: 'assistant'>, <Role.TOOL: 'tool'>]
>>> list(Type)  # Show all available types
[<Type.TEXT: 'text'>, <Type.IMAGE_PATH: 'image_path'>, <Type.IMAGE_URL: 'image_url'>, <Type.IMAGE_BINARY: 'image_binary'>]

```

### Message

The `Message` class represents a single message in a conversation. Key attributes include:

- `id`: Optional unique identifier for the message
- `content`: Optional text content of the message
- `binary`: Optional binary data for the message (used for images)
- `role`: The role of the entity sending the message
- `type`: The type of the message content

Either `content` or `binary` must be provided when creating a `Message` instance.

### Conversation

The `Conversation` class represents a sequence of messages. Key attributes include:

- `conversation_id`: Optional unique identifier for the conversation
- `messages`: List of `Message` objects that make up the conversation
- `metadata`: Optional dictionary for storing additional information about the conversation

### Creating Messages

```python
>>> from oumi.core.types.conversation import Message, Role, Type
>>> # Create a simple text message
>>> text_message = Message(content="Hello, world!", role=Role.USER)
>>> text_message.content
'Hello, world!'
>>> text_message.role
<Role.USER: 'user'>

>>> # Create an image message
>>> image_message = Message(binary=b"image_data", role=Role.USER, type=Type.IMAGE_BINARY)
>>> image_message.type
<Type.IMAGE_BINARY: 'image_binary'>

```

### Creating and Working with Conversations

```python
>>> from oumi.core.types.conversation import Conversation, Message, Role
>>> # Create a conversation with multiple messages
>>> conversation = Conversation(
...     messages=[
...         Message(content="Hi there!", role=Role.USER),
...         Message(content="Hello! How can I help?", role=Role.ASSISTANT),
...         Message(content="What's the weather?", role=Role.USER)
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
>>> from oumi.core.types.conversation import Conversation, Message, Role
>>> # Serialize to JSON
>>> conversation = Conversation(
...     messages=[Message(content="Hello!", role=Role.USER)],
...     metadata={"timestamp": "2024-01-01"}
... )
>>> json_data = conversation.to_json()
>>> print(json_data)
{"messages":[{"content":"Hello!","role":"user"}],"metadata":{"timestamp":"2024-01-01"}}

>>> # Deserialize from JSON
>>> restored = Conversation.from_json(json_data)
>>> restored.messages[0].content
'Hello!'
>>> restored.metadata["timestamp"]
'2024-01-01'

```

## Data Validation

The pydantic models automatically validate the data.
