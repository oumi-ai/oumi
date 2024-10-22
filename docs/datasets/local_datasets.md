# Supported Formats

Oumi uses a structured data format for representing conversations and messages.

This format is implemented using pydantic models, which provide type checking and data validation.

Let's look at some examples of compatible JSONL datasets.

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

### Role

The `Role` enum defines the possible roles for entities in a conversation:

- `SYSTEM`: Represents a system message
- `USER`: Represents a user message
- `ASSISTANT`: Represents an assistant message
- `TOOL`: Represents a tool message

### Type

The `Type` enum defines the possible types of message content:

- `TEXT`: Represents a text message
- `IMAGE_PATH`: Represents an image referenced by its file path
- `IMAGE_URL`: Represents an image referenced by its URL
- `IMAGE_BINARY`: Represents an image stored as binary data

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

## Usage

### Creating Messages

```python
from oumi.core.types.turn import Message, Role, Type

text_message = Message(content="Hello, world!", role=Role.USER)
image_message = Message(binary=b"image_data", role=Role.USER, type=Type.IMAGE_BINARY)
```

### Creating Conversations

```python
from oumi.core.types.turn import Conversation

conversation = Conversation(
    messages=[text_message, image_message],
    metadata={"source": "customer_support"}  # Add any metadata here
)
```

### Working with Conversations

```python
# Get the first user message
first_user_message = conversation.first_message(role=Role.USER)

# Get all assistant messages
assistant_messages = conversation.filter_messages(role=Role.ASSISTANT)

# Get the last message (any role)
last_message = conversation.last_message()
```

## Data Validation

The pydantic models automatically validate the data

## Serialization

These pydantic models can be easily serialized to and deserialized from JSON, making them suitable for storage and transmission:

```python
json_data = conversation.model_dump_json()
restored_conversation = Conversation.model_validate_json(json_data)
```
