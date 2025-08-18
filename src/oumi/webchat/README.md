# Oumi WebChat

A beautiful, full-featured web interface for Oumi's interactive chat functionality. WebChat provides access to all 27+ interactive commands through a modern browser interface with real-time conversation branching, file attachments, and system monitoring.

## Features

### 🌐 Web-Based Interface
- **Modern UI**: Built with Gradio 5.0+ for responsive, professional design
- **Real-time Communication**: WebSocket-based for instant message streaming
- **Mobile Responsive**: Optimized for desktop, tablet, and mobile devices
- **Dark/Light Themes**: Automatic theme switching based on system preferences

### 💬 Full Chat Functionality
- **Complete Feature Parity**: All CLI interactive commands available in web UI
- **Streaming Responses**: Real-time message generation with thinking indicators
- **Rich Formatting**: Markdown rendering, code syntax highlighting, LaTeX support
- **Message History**: Full conversation persistence with export capabilities

### 🌿 Interactive Branch Tree
- **Visual Branching**: D3.js-powered tree visualization of conversation paths
- **Click Navigation**: Switch between branches with single clicks
- **Branch Management**: Create, delete, and rename branches through UI
- **Context Switching**: Seamless conversation state management across branches

### 📎 Advanced File Support
- **Multi-format Uploads**: Images, PDFs, CSV, JSON, text files, and more
- **Drag & Drop**: Intuitive file attachment interface
- **Preview Generation**: Inline preview of uploaded content
- **Bulk Operations**: Handle multiple file attachments simultaneously

### ⚙️ System Integration
- **Live Monitoring**: Real-time GPU/CPU/memory usage display  
- **Context Window Tracking**: Visual progress bars for token usage
- **Model Switching**: Dynamic model/engine changes during conversation
- **Parameter Control**: Adjust temperature, max_tokens, and other settings live

### 🔧 Developer Features
- **Command Palette**: Searchable command interface (Cmd/Ctrl+K)
- **Auto-completion**: Real-time command and parameter suggestions
- **Export Options**: PDF, JSON, CSV, HTML, Markdown export formats
- **API Integration**: RESTful and WebSocket APIs for custom integrations

## Quick Start

### Installation

Install WebChat dependencies:
```bash
pip install 'oumi[webchat]'
```

### Basic Usage

Launch with a configuration file:
```bash
oumi webchat -c configs/recipes/llama3_1/inference/8b_infer.yaml
```

This starts:
- **Backend server** at `http://localhost:8000`
- **Frontend interface** at `http://localhost:7860`

### Advanced Usage

#### Backend Only (Development)
```bash
oumi webchat-server -c config.yaml --port 8000
```

#### Custom Ports
```bash
oumi webchat -c config.yaml --backend-port 8000 --frontend-port 7860
```

#### Public Sharing
```bash
oumi webchat -c config.yaml --share
```

#### With System Prompt
```bash
oumi webchat -c config.yaml --system-prompt "You are a helpful assistant."
```

## Architecture

### Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │◄──►│  WebChat Server  │◄──►│ Oumi Core       │
│  (Frontend)     │    │  (Backend)       │    │                 │
│                 │    │                  │    │                 │
│ • Chat Interface│    │ • WebSocket API  │    │ • Command System│
│ • Branch Tree   │    │ • Session Mgmt   │    │ • Inference     │
│ • File Upload   │    │ • Command Router │    │ • Branch Manager│
│ • System Monitor│    │ • Real-time Sync │    │ • Thinking Proc │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Files

- `server.py` - Extended WebSocket server with command support
- `interface.py` - Main Gradio interface and UI components  
- `components/branch_tree.py` - Interactive D3.js branch visualization
- `utils/gradio_helpers.py` - Utility functions for Gradio integration

## Interactive Commands

All 27 CLI commands are available in WebChat:

### Basic Commands
- `/help()` - Show available commands
- `/clear()` - Clear conversation history
- `/delete()` - Delete last conversation turn
- `/regen()` - Regenerate last response

### File Operations
- `/attach(filename)` - Attach files to conversation
- `/save(filename)` - Export conversation to various formats
- `/import(filename)` - Import previous conversations

### Branch Operations
- `/branch(name)` - Create new conversation branch
- `/switch(name)` - Switch to different branch
- `/branches()` - List all branches with previews
- `/branch_delete(name)` - Delete a branch

### Advanced Features
- `/swap(model)` - Change models during conversation
- `/set(parameter=value)` - Adjust generation parameters
- `/fetch(url)` - Fetch web content into conversation
- `/shell(command)` - Execute shell commands (if enabled)
- `/render(filename)` - Create asciinema recordings

## Configuration

WebChat uses standard Oumi inference configurations:

```yaml
# Basic configuration
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  model_max_length: 4096

generation:
  max_new_tokens: 2048
  temperature: 0.7

engine: VLLM

# WebChat-specific styling (optional)
style:
  use_emoji: true
  expand_panels: true
  theme: "default"  # or "dark", "neon", "minimal"
```

## API Reference

### WebSocket API

Connect to `ws://localhost:8000/v1/oumi/ws?session_id=<session_id>`

#### Message Types

**Send Messages:**
```json
{
  "type": "chat_message",
  "message": "Hello, world!"
}

{
  "type": "command", 
  "command": "/help()"
}

{
  "type": "get_branches"
}
```

**Receive Messages:**
```json
{
  "type": "assistant_message",
  "content": "Hello! How can I help you?",
  "timestamp": 1234567890
}

{
  "type": "branches_update",
  "branches": [...],
  "current_branch": "main"
}
```

### REST API

#### Command Execution
```bash
curl -X POST http://localhost:8000/v1/oumi/command \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "command": "help", "args": []}'
```

#### Branch Operations
```bash
# List branches
curl http://localhost:8000/v1/oumi/branches?session_id=test

# Switch branch
curl -X POST http://localhost:8000/v1/oumi/branches \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "action": "switch", "branch_id": "main"}'
```

## Development

### Running in Development Mode

1. Install development dependencies:
   ```bash
   pip install -e '.[dev,webchat]'
   ```

2. Start backend server:
   ```bash
   python -c "
   from oumi.webchat.server import run_webchat_server
   from oumi.core.configs import InferenceConfig
   config = InferenceConfig.from_yaml_file('config.yaml')
   run_webchat_server(config, host='localhost', port=8000)
   "
   ```

3. Start frontend (separate terminal):
   ```bash
   python -c "
   from oumi.webchat.interface import launch_webchat
   from oumi.core.configs import InferenceConfig
   config = InferenceConfig.from_yaml_file('config.yaml')  
   launch_webchat(config, server_url='http://localhost:8000', server_port=7860)
   "
   ```

### Adding Custom Components

WebChat uses Gradio's HTML component for custom features. To add new functionality:

1. Create component in `components/`
2. Add JavaScript with D3.js or vanilla JS
3. Integrate with WebSocket API for real-time updates
4. Add to main interface in `interface.py`

### Testing

Run the demo script:
```bash
python demo_webchat.py --run
```

## Troubleshooting

### Common Issues

**WebSocket Connection Failed**
- Ensure backend server is running on correct port
- Check firewall settings for WebSocket connections
- Verify CORS configuration for your domain

**Branch Tree Not Loading**
- Check browser console for JavaScript errors
- Ensure D3.js is loading from CDN
- Verify WebSocket messages for branch data

**File Upload Issues**  
- Check file size limits (default: 100MB)
- Ensure proper MIME type detection
- Verify upload directory permissions

**Model Loading Errors**
- Ensure model is properly configured in YAML
- Check GPU memory availability
- Verify model access permissions (for gated models)

### Debug Mode

Enable debug logging:
```bash
OUMI_LOG_LEVEL=DEBUG oumi webchat -c config.yaml
```

### Performance Tips

- Use vLLM engine for models >7B parameters
- Enable GPU monitoring for resource tracking
- Use `/compact()` command to manage long conversations
- Consider using quantized models for faster inference

## Contributing

WebChat is part of the main Oumi project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Submit a pull request

See the main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for detailed guidelines.

## License

WebChat is licensed under the same Apache 2.0 license as Oumi. See [LICENSE](../../../LICENSE) for details.