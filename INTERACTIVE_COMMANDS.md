# Oumi Interactive Commands & Input System

This document describes the enhanced interactive system for `oumi infer`, including commands and multi-line input support.

## Multi-Line Input System ✅ COMPLETE

### Input Modes

#### Single-line Mode (Default)
- Press **Enter** to send your message
- Type `/ml` to switch to multi-line mode
- Perfect for quick questions and commands

#### Multi-line Mode
- Press **Enter** to add a new line
- Press **Enter** on an empty line to send your message
- Type `/sl` to switch back to single-line mode
- Great for longer messages, code, or formatted text

### Mode Switching
```
You: /ml                    # Switch to multi-line mode
📝 Switched to multi-line input mode
Press Enter to add new lines, empty line to submit

You: This is my first line
   │: This is my second line
   │: This is my third line
   │: (empty line submits)

You: /sl                    # Switch back to single-line
✏️ Switched to single-line input mode
```

### Visual Indicators
- **Single-line**: `You: (single-line)`
- **Multi-line**: First line shows `You:`, continuation lines show `│:`
- Help messages guide you through mode switching
- Clear feedback when modes change

## Phase 1: Core Command System ✅ COMPLETE

### Available Commands

#### Basic Commands
- **`/help()`** - Display comprehensive help with all available commands and usage examples
- **`/exit()`** - Gracefully exit the interactive chat session

### Command Syntax

Commands must:
- Start with `/`
- Be the first thing in your input message
- Use parentheses for arguments: `/command(arg1, arg2)`
- Use `key=value` format for parameters: `/set(temperature=0.8)`
- Commands are case-insensitive

### Examples
```
/help()                     # Show help
/exit()                     # Exit chat
/attach(document.pdf)       # Attach file (Phase 2)
/set(temperature=0.8)       # Adjust parameters (Phase 4)
/save(conversation.pdf)     # Save to PDF (Phase 5)
```

### Integration

The command system is fully integrated with the multi-line input system:
- Commands work in both single-line and multi-line modes
- `/help()` shows both command and input mode information
- All commands are processed before inference
- Seamless integration with styling themes
- Mode switching commands (`/ml`, `/sl`) are handled by the input system

### Testing

Test the command system with:
```bash
oumi infer -i -c configs/recipes/qwen2_5/inference/3b_infer_command_test.yaml
```

Try these commands:
- `/help()` - See full documentation
- `/exit()` - Exit gracefully
- `/invalid()` - See error handling

## Phase 2: File Attachments (Planned)

### Commands (Coming Soon)
- **`/attach(path)`** - Attach files to conversation
  - **Images**: JPG, PNG, BMP, TIFF, WEBP
  - **Documents**: PDF (text + images), TXT, MD, JSON, CSV
  - **Code**: Python, YAML, etc.

### Implementation Features
- Auto file-type detection
- Image processing and encoding
- PDF text extraction with page images
- CSV/JSON structured data formatting
- Error handling for missing/invalid files

## Phase 3: Conversation Management (Planned)

### Commands (Coming Soon)
- **`/delete()`** - Delete the previous conversation turn
- **`/regen()`** - Regenerate the last assistant response

### Implementation Features
- Multi-turn conversation tracking
- Conversation history manipulation
- Response regeneration with same parameters
- Undo/redo functionality

## Phase 4: Parameter Adjustment (Planned)

### Commands (Coming Soon)
- **`/set(param=value)`** - Adjust generation parameters dynamically

### Supported Parameters
- `temperature` - Control randomness (0.0-2.0)
- `top_p` - Nucleus sampling (0.0-1.0)
- `max_tokens` - Maximum response length
- `sampling` - Enable/disable sampling (true/false)

### Examples
```
/set(temperature=0.8)       # More creative
/set(temperature=0.1)       # More focused
/set(top_p=0.9)            # Nucleus sampling
/set(max_tokens=2048)      # Longer responses
/set(sampling=true)        # Enable sampling
```

## Phase 5: Conversation Export (Planned)

### Commands (Coming Soon)
- **`/save(path)`** - Export conversation to formatted PDF

### Implementation Features
- PDF generation with proper formatting
- Conversation turn organization
- Syntax highlighting for code blocks
- Image embedding for visual content
- Metadata inclusion (model, parameters, timestamps)

## Architecture

### Components

1. **`CommandParser`** (`src/oumi/core/commands/command_parser.py`)
   - Regex-based command parsing
   - Argument extraction (positional + keyword)
   - Command validation
   - Syntax error handling

2. **`CommandHandler`** (`src/oumi/core/commands/command_handler.py`)
   - Command execution logic
   - Integration with styling system
   - Error and success feedback
   - Modular handler methods

3. **Integration** (`src/oumi/infer.py`)
   - Command detection in input loop
   - Results processing
   - Flow control (continue/exit)
   - Error display

### Error Handling

The system provides comprehensive error handling:
- Invalid command syntax detection
- Unknown command validation
- Parameter type validation
- File access error handling
- Graceful fallbacks with user feedback

### Styling Integration

Commands respect the active style theme:
- Help panels use configured colors
- Error messages match error styling
- Success feedback follows theme
- Emoji usage respects `use_emoji` setting

### Testing

Unit tests cover:
- Command parsing accuracy
- Validation logic
- Handler execution
- Integration flow
- Error scenarios

## Future Enhancements

- **Command aliases**: `/h` for `/help()`, `/q` for `/exit()`
- **Command history**: Up arrow to repeat previous commands
- **Batch commands**: `/multi(help,set(temp=0.8),attach(file.pdf))`
- **Conditional commands**: `/if(turn>5,save(backup.pdf))`
- **Scheduled commands**: `/after(10min,save(session.pdf))`

## Configuration

Commands work with all existing inference configs and styling themes. No additional configuration required - the system is enabled by default in interactive mode.

For custom styling, commands respect these style parameters:
- `assistant_title_style` - Help panel titles
- `error_style` - Error message text
- `error_border_style` - Error panel borders
- `use_emoji` - Emoji in command feedback
- `expand_panels` - Panel width behavior
