# PocketDoc Model Family

This directory contains configurations for PocketDoc's fine-tuned models.

## PersonalityEngine V1.2.0 24B

PocketDoc Dan's PersonalityEngine V1.2.0 is a 24B parameter model fine-tuned for personality-driven conversations. It's based on Mistral Small 24B Base 2501 and optimized for creating engaging, character-driven interactions.

### Model Details
- **Base Model**: Mistral Small 24B Base 2501
- **Fine-tuned by**: PocketDoc Dan
- **Version**: 1.2.0
- **Parameters**: 24B
- **Context Length**: 32,768 tokens
- **Specialization**: Personality-driven conversation and character roleplay

### Available Configurations

#### Inference Configurations
- `personalityengine_24b_infer.yaml` - NATIVE engine (CPU/GPU compatible)
- `personalityengine_24b_vllm_infer.yaml` - vLLM engine (high-performance GPU)
- `personalityengine_24b_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `personalityengine_24b_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS/CPU optimized)

### Usage Examples

```bash
# Native inference
oumi infer -i -c configs/recipes/pocketdoc/inference/personalityengine_24b_infer.yaml

# High-performance GPU inference
oumi infer -i -c configs/recipes/pocketdoc/inference/personalityengine_24b_vllm_infer.yaml

# Quantized GPU inference
oumi infer -i -c configs/recipes/pocketdoc/inference/personalityengine_24b_gguf_infer.yaml

# macOS CPU/Metal inference
oumi infer -i -c configs/recipes/pocketdoc/inference/personalityengine_24b_gguf_macos_infer.yaml
```

### Model Source
- **HuggingFace**: [PocketDoc/Dans-PersonalityEngine-V1.2.0-24b](https://huggingface.co/PocketDoc/Dans-PersonalityEngine-V1.2.0-24b)
- **GGUF Quantizations**: [bartowski/PocketDoc_Dans-PersonalityEngine-V1.2.0-24b-GGUF](https://huggingface.co/bartowski/PocketDoc_Dans-PersonalityEngine-V1.2.0-24b-GGUF)

### Recommended Use Cases
- Interactive character roleplay
- Personality-driven chatbots
- Creative writing assistance with character consistency
- Dialogue generation for games and storytelling
- Conversational AI with distinctive personality traits