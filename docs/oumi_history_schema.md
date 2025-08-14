# Oumi Conversation History Schema

This document describes the complete schema for Oumi's comprehensive conversation history format used by `/save_history()` and `/import_history()` commands.

## Overview

The Oumi conversation history format preserves the complete state of interactive chat sessions, including:
- **Full conversation tree** with all branches and their relationships
- **Complete configuration** including model, generation, and style parameters
- **Session metadata** with timestamps, IDs, and statistics
- **Attachment information** from files used during the conversation
- **Command history** (placeholder for future implementation)

## File Format

- **Format**: JSON
- **Extension**: `.json` (automatically added if not present)
- **Encoding**: UTF-8
- **Schema Version**: 1.0.0

## Schema Structure

### Root Object

```json
{
  "schema_version": "1.0.0",
  "format": "oumi_conversation_history", 
  "created_at": "2025-01-15T14:30:45.123456",
  "source": "oumi_interactive_chat",
  "session": { ... },
  "configuration": { ... },
  "branches": { ... },
  "command_history": [ ... ],
  "attachments": [ ... ],
  "statistics": { ... }
}
```

### Session Information

Tracks session-level metadata:

```json
"session": {
  "chat_id": "DeepSeek-R1-Distill-Qwen-32B-GGUF_20250115_143045",
  "current_branch_id": "main",
  "total_session_time": null,
  "oumi_version": "latest"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `chat_id` | string | Unique session identifier (format: `{model}_{timestamp}`) |
| `current_branch_id` | string | ID of the active conversation branch |
| `total_session_time` | number/null | Total session duration in seconds (future) |
| `oumi_version` | string | Oumi version used to create the history |

### Configuration

Complete model and inference configuration:

```json
"configuration": {
  "model": {
    "model_name": "microsoft/DialoGPT-medium", 
    "model_max_length": "2048",
    "torch_dtype_str": "float16",
    "attn_implementation": "sdpa"
  },
  "generation": {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "sampling": true
  },
  "engine": "VLLM",
  "style": {
    "user_prompt_style": "bold blue",
    "assistant_title_style": "bold cyan", 
    "use_emoji": true,
    "expand_panels": false
  },
  "inference_params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "sampling": true
  }
}
```

### Conversation Branches

Complete conversation tree with all branches and their relationships:

```json
"branches": {
  "main": {
    "id": "main",
    "name": "Main",
    "created_at": "2025-01-15T14:30:45.123456",
    "last_active": "2025-01-15T14:35:20.789012",
    "parent_branch_id": null,
    "branch_point_index": 0,
    "conversation_history": [
      {
        "role": "user",
        "content": "Hello! How are you today?"
      },
      {
        "role": "assistant", 
        "content": "Hello! I'm doing well, thank you for asking..."
      }
    ],
    "model_name": "microsoft/DialoGPT-medium",
    "engine_type": "VLLM",
    "model_config": { ... },
    "generation_config": { ... }
  },
  "experiment_1": {
    "id": "experiment_1",
    "name": "Alternative Response",
    "created_at": "2025-01-15T14:32:10.456789",
    "last_active": "2025-01-15T14:34:55.123456",
    "parent_branch_id": "main",
    "branch_point_index": 1,
    "conversation_history": [ ... ],
    "model_name": "microsoft/DialoGPT-medium",
    "engine_type": "VLLM",
    "model_config": { ... },
    "generation_config": { ... }
  }
}
```

#### Branch Object Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique branch identifier |
| `name` | string/null | Human-readable branch name |
| `created_at` | string | ISO 8601 timestamp of branch creation |
| `last_active` | string | ISO 8601 timestamp of last activity |
| `parent_branch_id` | string/null | ID of parent branch (null for root) |
| `branch_point_index` | number | Index in parent where branch diverged |
| `conversation_history` | array | Complete message history for this branch |
| `model_name` | string/null | Model used in this branch |
| `engine_type` | string/null | Inference engine used |
| `model_config` | object/null | Serialized model configuration |
| `generation_config` | object/null | Serialized generation parameters |

#### Message Format

Each message in `conversation_history` follows this structure:

```json
{
  "role": "user" | "assistant" | "attachment" | "system",
  "content": "Message content...",
  "timestamp": "2025-01-15T14:30:45.123456",
  "metadata": { 
    "thinking_content": "...",
    "raw_thinking": "...",
    "word_count": 150
  }
}
```

### Command History

Record of commands executed during the session:

```json
"command_history": [
  {
    "command": "branch",
    "args": ["experiment"],
    "timestamp": "2025-01-15T14:32:10.456789",
    "success": true,
    "result": "Created branch 'experiment'"
  },
  {
    "command": "set",
    "args": ["temperature=0.8"],
    "timestamp": "2025-01-15T14:33:15.789012", 
    "success": true,
    "result": "Updated temperature to 0.8"
  },
  {
    "note": "Command history tracking not yet implemented",
    "timestamp": "2025-01-15T14:30:45.123456",
    "type": "system_note"
  }
]
```

**Note**: Command history tracking is planned for future implementation. Current exports include a placeholder note.

### Attachments

Metadata about files attached during the conversation:

```json
"attachments": [
  {
    "filename": "research_paper.pdf",
    "file_type": "pdf",
    "size_bytes": 1024000,
    "timestamp": "2025-01-15T14:31:30.123456",
    "content_preview": "This paper presents a novel approach to..."
  },
  {
    "filename": "data.csv",
    "file_type": "csv", 
    "size_bytes": 50000,
    "timestamp": "2025-01-15T14:32:45.789012",
    "content_preview": "Name,Age,City\nJohn,25,New York\nJane,30,..."
  }
]
```

### Statistics

Session-wide statistics and metrics:

```json
"statistics": {
  "total_branches": 3,
  "total_messages": 45,
  "total_user_messages": 22,
  "total_assistant_messages": 21,
  "estimated_tokens": 12500,
  "created_at": "2025-01-15T14:30:45.123456"
}
```

## Usage Examples

### Saving Complete History

```bash
# Save all branches, config, and metadata
/save_history(my_project_complete.json)

# Output example:
# Saved complete conversation history to my_project_complete.json
# 📊 Saved: 3 branches, 45 messages, 0 commands, full config & metadata
```

### Importing Complete History

```bash
# Restore entire conversation state
/import_history(my_project_complete.json)

# Output example:
# Restored conversation history from my_project_complete.json  
# 📊 Restored: 3 branches, 45 messages, 0 commands, config & metadata
```

### After Import

Once imported, you can:
- Use `/branches()` to see all restored branches
- Use `/switch(branch_name)` to navigate between branches
- Continue conversations from any branch
- All original configuration and metadata is preserved

## Schema Validation

The import process validates:
- **Required fields**: `schema_version`, `format`, `branches`
- **Format identifier**: Must be `"oumi_conversation_history"`  
- **Branch structure**: Each branch must have `id` and `conversation_history`
- **Data types**: Conversation history must be arrays, etc.

Invalid files will be rejected with descriptive error messages.

## Differences from Simple `/save()`

| Feature | `/save()` | `/save_history()` |
|---------|-----------|-------------------|
| **Branches** | Current only | Complete tree |
| **Configuration** | Basic | Full model/generation/style |
| **Commands** | None | History tracking (future) |
| **Attachments** | Limited | Complete metadata |
| **Metadata** | Minimal | Session statistics |
| **Restoration** | Basic import | Complete state restoration |

## File Size Considerations

History files can be large for complex conversations:
- **Branches**: Each branch stores complete message history
- **Attachments**: Content previews included (not full content)
- **Configuration**: All parameters serialized
- **Metadata**: Comprehensive statistics included

For very large conversations (>100MB), consider:
- Periodic cleanup with `/clear_thoughts()` before saving
- Using `/compact()` to reduce context size
- Saving individual branches separately if needed

## Compatibility

- **Forward compatible**: Newer Oumi versions can read older schema versions
- **Backward compatible**: Schema includes version field for handling changes
- **Cross-platform**: JSON format works across all operating systems
- **External tools**: Standard JSON can be processed by other applications

## Security Notes

History files may contain:
- **Sensitive conversations**: Consider encryption for confidential chats
- **Model configurations**: May reveal infrastructure details
- **File metadata**: Attachment filenames and previews included
- **Session IDs**: Unique identifiers that could be tracked

Handle history files with the same security considerations as your original conversations.