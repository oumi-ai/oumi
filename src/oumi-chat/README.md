# Oumi Chat

Interactive chat interface for Oumi models with advanced features including command system, multimodal support, and rich terminal UI.

## Features

- **Interactive Chat Interface**: Rich terminal-based chat interface with customizable styling
- **Command System**: Powerful command system for managing conversations, models, and parameters
- **Multimodal Support**: Support for images and other media types
- **Conversation Management**: Branch, save, load, and export conversations
- **Macro System**: Reusable prompt templates using Jinja2
- **System Monitoring**: Real-time system resource monitoring during inference
- **Thinking Display**: Display model reasoning with thinking tags
- **Enhanced Input**: Multi-line input, command history, and autocompletion

## Installation

```bash
pip install oumi-chat
```

Or install as part of the main Oumi package:

```bash
pip install oumi[interactive]
```

## Usage

### Command Line

```bash
# Start a chat session
oumi chat --config path/to/config.yaml

# With system prompt
oumi chat --config path/to/config.yaml --system-prompt "You are a helpful assistant"

# With an image (for multimodal models)
oumi chat --config path/to/config.yaml --image path/to/image.jpg
```

### Python API

```python
from oumi_chat import CommandContext, CommandRouter, EnhancedInput
from oumi import InferenceConfig
from oumi.builders.inference_engines import build_inference_engine

# Load configuration
config = InferenceConfig.from_yaml("path/to/config.yaml")

# Create inference engine
engine = build_inference_engine(config)

# Initialize chat components
console = Console()
conversation_history = []
system_monitor = SystemMonitor()

command_context = CommandContext(
    console, config, conversation_history, engine, system_monitor
)
command_router = CommandRouter(command_context)

# Start chat loop
input_handler = EnhancedInput(console)
# ... (see examples for full implementation)
```

## Commands

Once in a chat session, use `/help()` to see all available commands:

- `/help()` - Show available commands
- `/save(filename)` - Save conversation
- `/load(filename)` - Load conversation
- `/export(format, filename)` - Export conversation (json, txt, pdf, html, markdown)
- `/branch()` - Create conversation branch
- `/switch(branch_name)` - Switch to a branch
- `/compact()` - Compact conversation history
- `/regen()` - Regenerate last response
- `/macro(name, **args)` - Load and apply a macro template
- `/fetch(url)` - Fetch web content
- `/attach(filepath)` - Attach a file to the conversation
- And many more...

## Configuration

Oumi Chat supports extensive styling configuration. Create a YAML config file:

```yaml
style:
  theme: "monokai"  # Available: dark, minimal, monokai, neon
  use_emoji: true
  expand_panels: false
  user_prompt_style: "bold blue"
  # ... (see examples for full configuration)
```

## Macros

Create reusable prompt templates using Jinja2 in the `macros/` directory:

```jinja
{# macros/creative_writing.jinja #}
You are a creative writing assistant specializing in {{ genre }}.
Help the user write {{ format }} with attention to {{ style }}.
```

Use in chat:

```
/macro(creative_writing, genre="science fiction", format="short stories", style="vivid imagery")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/oumi-chat/

# Run linting
ruff check .
```

## License

Apache License 2.0 - see LICENSE file for details.
