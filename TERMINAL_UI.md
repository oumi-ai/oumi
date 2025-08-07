# Oumi Terminal UI - Beautiful Frontend for Inference

This document describes the new terminal UI enhancements for Oumi inference, providing beautiful and user-friendly interfaces for interacting with your models.

## 🌟 Features

### ✨ Enhanced Interactive Mode
- **Rich formatting** with syntax highlighting for code blocks
- **Markdown rendering** for properly formatted responses
- **Panels and borders** for clear conversation separation
- **Loading indicators** with animated spinners
- **Error handling** with styled error messages
- **Model information** display on startup

### 🚀 HTTP Server Mode
- **OpenAI-compatible API** endpoints
- **Drop-in replacement** for OpenAI API calls
- **Cross-origin support** for web clients
- **Health check endpoints** for monitoring

### 🤖 AIChat Integration
- **Professional TUI client** with advanced features
- **Automatic server management**
- **Configuration backup** and restoration
- **Multi-model support** and switching

## 🚀 Quick Start

### Option 1: Enhanced Interactive Mode (Recommended)

The simplest way to get a beautiful chat interface:

```bash
# Basic usage
oumi infer -i -c configs/recipes/smollm/inference/135m_infer.yaml

# With system prompt
oumi infer -i -c my_config.yaml --system-prompt "You are a helpful coding assistant."
```

**Features:**
- 🎨 Beautiful Rich-formatted output
- 📱 Responsive terminal interface
- ⚡ Direct model access (no HTTP overhead)
- 🔧 Built into Oumi CLI

### Option 2: Using the Wrapper Script

For the most user-friendly experience:

```bash
# Enhanced interactive mode
./scripts/oumi-chat -c configs/recipes/smollm/inference/135m_infer.yaml

# Use AIChat TUI (requires aichat installed)
./scripts/oumi-chat -c configs/recipes/smollm/inference/135m_infer.yaml --use-aichat

# Custom system prompt
./scripts/oumi-chat -c my_config.yaml -s "You are a helpful assistant."
```

### Option 3: HTTP Server Mode

For compatibility with existing OpenAI-compatible tools:

```bash
# Start server
oumi infer --server-mode -c configs/recipes/smollm/inference/135m_infer.yaml

# Custom host/port
oumi infer --server-mode --host localhost --port 9000 -c my_config.yaml

# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oumi-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 📋 Available Options

### CLI Options

```bash
oumi infer [OPTIONS] -c CONFIG_FILE

Options:
  -i, --interactive           Run in enhanced interactive session
  --server-mode              Run as HTTP server (OpenAI-compatible)
  --host HOST                Server host (default: 0.0.0.0)
  --port PORT                Server port (default: 8000)
  --system-prompt TEXT       System prompt for the model
  --image PATH               Input image for vision models
```

### Wrapper Script Options

```bash
oumi-chat [OPTIONS] -c CONFIG_FILE

Options:
  -c, --config CONFIG        Path to inference configuration (required)
  -h, --host HOST           Server host (default: 0.0.0.0)
  -p, --port PORT           Server port (default: 8000)
  -s, --system-prompt TEXT  System prompt for the model
  --use-aichat              Use AIChat TUI client
  --aichat-model MODEL      Model name for AIChat
  --help                    Show help message
```

## 🔗 OpenAI API Compatibility

The server mode provides these OpenAI-compatible endpoints:

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions

### Example Usage with Python OpenAI Client

```python
from openai import OpenAI

# Point to your Oumi server
client = OpenAI(
    api_key="dummy-key-not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="oumi-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## 🛠️ Integration with External TUI Clients

### AIChat Setup

1. **Install AIChat:**
   ```bash
   # Via cargo
   cargo install aichat

   # Via Homebrew (macOS)
   brew install aichat

   # Via package manager (Arch Linux)
   pacman -S aichat
   ```

2. **Use with wrapper script:**
   ```bash
   ./scripts/oumi-chat -c my_config.yaml --use-aichat
   ```

   The script will:
   - Automatically start the Oumi server
   - Configure AIChat to use your Oumi server
   - Launch the AIChat TUI
   - Clean up when you exit

### Manual AIChat Configuration

If you prefer manual setup:

1. **Start Oumi server:**
   ```bash
   oumi infer --server-mode -c your_config.yaml
   ```

2. **Configure AIChat** (`~/.config/aichat/config.yaml`):
   ```yaml
   model: openai:oumi-model
   clients:
   - type: openai
     api_base: http://localhost:8000/v1
     api_key: dummy-key
     models:
     - name: oumi-model
       max_input_tokens: 4096
       max_output_tokens: 4096
   ```

3. **Run AIChat:**
   ```bash
   aichat --model openai:oumi-model
   ```

## 🎨 UI Features

### Enhanced Interactive Mode

**Conversation Display:**
- User messages with blue styling
- Assistant responses in bordered panels
- Automatic markdown rendering for code blocks
- Syntax highlighting for programming languages
- Loading spinner during inference

**Error Handling:**
- Styled error panels with red borders
- Clear error messages and suggestions
- Graceful handling of interruptions (Ctrl+C/Ctrl+D)

**Model Information:**
- Model name and engine type display on startup
- System prompt confirmation
- Connection status indicators

### AIChat TUI Features

When using AIChat with the wrapper script:

- **Advanced editing:** Multi-line input with syntax highlighting
- **History management:** Persistent conversation history
- **Model switching:** Switch between models with `.model` command
- **Session management:** Save and load conversation sessions
- **Customizable themes:** Light and dark theme support
- **Keyboard shortcuts:** Emacs-style editing keybindings

## 🔧 Configuration

### Model Configuration

Any standard Oumi inference config works. Example:

```yaml
model:
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  model_max_length: 2048
  torch_dtype_str: "bfloat16"
  attn_implementation: "sdpa"

generation:
  max_new_tokens: 500
  temperature: 0.7

engine: NATIVE
```

### Server Configuration

Server settings can be customized:

```bash
# Custom host and port
oumi infer --server-mode --host 127.0.0.1 --port 9000

# With logging
oumi infer --server-mode --log-level DEBUG
```

## 🚦 Troubleshooting

### Common Issues

**"Rich not found" errors:**
- Rich is included in Oumi dependencies and should be available

**"Server failed to start":**
- Check if port is already in use: `lsof -i :8000`
- Try different port: `--port 8001`
- Check firewall settings for the host/port

**"AIChat not found":**
- Install AIChat: `cargo install aichat`
- Or use enhanced interactive mode instead

**Model loading errors:**
- Ensure model config is valid
- Check available GPU memory
- Try smaller models for testing

### Performance Tips

**For faster inference:**
- Use VLLM engine when available
- Enable GPU acceleration
- Reduce `max_new_tokens` for quicker responses

**For better UI responsiveness:**
- Use smaller models during development
- Enable appropriate logging levels
- Close unnecessary terminal applications

## 📝 Examples

### Complete Examples

**1. Quick chat with SmolLM:**
```bash
oumi infer -i -c configs/recipes/smollm/inference/135m_infer.yaml
```

**2. Code assistant with system prompt:**
```bash
oumi infer -i -c configs/recipes/llama3_2/inference/3b_infer.yaml \
  --system-prompt "You are an expert Python developer. Provide clear, well-commented code examples."
```

**3. Server for external tools:**
```bash
oumi infer --server-mode --host 0.0.0.0 --port 8000 \
  -c configs/recipes/qwen3/inference/8b_infer.yaml
```

**4. AIChat integration:**
```bash
./scripts/oumi-chat -c configs/recipes/llama3_2/inference/3b_infer.yaml --use-aichat
```

### Client Examples

**Python OpenAI client:**
```python
import openai
client = openai.OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

chat = client.chat.completions.create(
    model="oumi-model",
    messages=[{"role": "user", "content": "Explain Python decorators"}]
)
print(chat.choices[0].message.content)
```

**Curl:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oumi-model",
    "messages": [{"role": "user", "content": "Hello world!"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

## 🎯 Next Steps

1. **Try the enhanced interactive mode** - Start with basic chat
2. **Experiment with different models** - Test various model sizes
3. **Set up AIChat integration** - For the best TUI experience
4. **Build custom clients** - Use the OpenAI-compatible API
5. **Create wrapper scripts** - Customize for your workflow

The new terminal UI makes Oumi inference more accessible and enjoyable to use, whether you're doing quick experiments, building applications, or integrating with existing tools.
