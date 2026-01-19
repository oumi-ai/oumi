# Oumi Deploy - Model Deployment CLI

Deploy trained models to inference providers (Together.ai, Fireworks.ai) for production use.

## Quick Start

### Deploy a model with config file

```bash
# Deploy to Together.ai
oumi deploy up --config configs/examples/deploy/together_deploy.yaml

# Deploy to Fireworks.ai
oumi deploy up --config configs/examples/deploy/fireworks_deploy.yaml

# Deploy a LoRA adapter
oumi deploy up --config configs/examples/deploy/lora_adapter_deploy.yaml
```

### Deploy a model with CLI arguments

```bash
# Upload + create endpoint in one command
oumi deploy up \
  --model-path s3://my-bucket/models/my-model/ \
  --provider together \
  --hardware nvidia_a100_80gb

# With CLI overrides
oumi deploy up \
  --config configs/examples/deploy/together_deploy.yaml \
  --hardware nvidia_h100_80gb
```

## Commands

### Phase 1 - Core Commands

#### Upload a model

```bash
oumi deploy upload \
  --model-path s3://bucket/model/ \
  --provider together \
  --model-name "my-model" \
  --wait
```

#### Create an endpoint

```bash
oumi deploy create-endpoint \
  --model-id "together-model-id" \
  --provider together \
  --hardware nvidia_a100_80gb \
  --gpu-count 2 \
  --min-replicas 1 \
  --max-replicas 4 \
  --wait
```

#### Check deployment status

```bash
# Get status once
oumi deploy status \
  --endpoint-id "ep-123" \
  --provider together

# Watch until ready
oumi deploy status \
  --endpoint-id "ep-123" \
  --provider together \
  --watch
```

#### List all deployments

```bash
oumi deploy list --provider together
```

#### List all uploaded models

```bash
# List models from all providers with API keys configured (default)
oumi deploy list-models

# List models from a specific provider
oumi deploy list-models --provider together

# Include public platform models too
oumi deploy list-models --all

# List only ongoing uploads (pending/processing)
oumi deploy list-models --status pending

# List failed uploads
oumi deploy list-models --status failed

# List ready models
oumi deploy list-models --status ready

# Combine filters: specific provider + status
oumi deploy list-models --provider together --status pending
```

This command shows:
- Model ID
- Model Name
- Status (ready, pending, failed, etc.)
- Model Type (full or adapter)
- Provider
- Created timestamp
- Status summary by count

**Filtering Options:**
- **Default**: Shows models from all providers with API keys configured (Together.ai and/or Fireworks.ai). Only shows your custom uploaded models/fine-tuning jobs (excludes public platform models). Results are sorted by creation date (most recent first).
- **`--provider <provider>`**: Limit to a specific provider (together or fireworks)
- **`--all`**: Include public platform models (like meta-llama, mistralai, etc.)
- **`--status <status>`**: Filter by status
  - `pending` - Models/jobs pending or queued
  - `running` - Currently training/uploading
  - `completed` - Successfully completed
  - `failed` - Failed uploads/training
  - `cancelled` - Cancelled jobs

**Note**: The command lists:
- Your [model upload jobs](https://docs.together.ai/reference/getjob) (job-* IDs) - Direct model uploads
- Your [fine-tuning jobs](https://docs.together.ai/reference/get_fine-tunes) (ft-* IDs) - Fine-tuning tasks

Both have proper status tracking. Use this to monitor your model uploads and training progress.

#### Delete an endpoint

```bash
oumi deploy delete \
  --endpoint-id "ep-123" \
  --provider together
```

#### Delete an uploaded model

```bash
# Delete a Fireworks model (fully supported)
oumi deploy delete-model \
  --model-id "my-model" \
  --provider fireworks

# Delete with auto-confirmation
oumi deploy delete-model \
  --model-id "my-model" \
  --provider fireworks \
  --force

# Together.ai limitation - models must be deleted via dashboard
oumi deploy delete-model \
  --model-id "model-id" \
  --provider together
# Error: Together.ai does not currently support model deletion via their API
```

**Note:**
- **Fireworks.ai**: Full model deletion support via API
- **Together.ai**: Model deletion not supported via API. Models must be deleted through the [Together.ai dashboard](https://api.together.xyz/playground)
- Always delete endpoints before deleting models to avoid dangling references

### Phase 2 - Enhanced Commands

#### One-command deploy

```bash
# Deploy with config file
oumi deploy up --config deploy_config.yaml

# Deploy with CLI args
oumi deploy up \
  --model-path s3://bucket/model/ \
  --provider together \
  --hardware nvidia_a100_80gb \
  --wait
```

#### List available hardware

```bash
# List all hardware for a provider
oumi deploy list-hardware --provider together

# Filter by model compatibility
oumi deploy list-hardware \
  --provider fireworks \
  --model-id "my-model-id"
```

#### Test endpoint

```bash
oumi deploy test \
  --endpoint-id "ep-123" \
  --provider together \
  --prompt "Hello, how are you?" \
  --max-tokens 100
```

## Configuration File Format

```yaml
# Model source (required)
model_source: s3://my-bucket/models/my-model/

# Provider (required): "together" or "fireworks"
provider: together

# Model name (required)
model_name: my-finetuned-model-v1

# Model type: "full" or "adapter"
model_type: full

# Base model (required if model_type is "adapter")
# base_model: meta-llama/Llama-2-7b-hf

# Hardware configuration (required)
hardware:
  accelerator: nvidia_a100_80gb
  count: 1

# Autoscaling configuration (required)
autoscaling:
  min_replicas: 1
  max_replicas: 2

# Optional: test prompts to run after deployment
test_prompts:
  - "Test prompt 1"
  - "Test prompt 2"
```

## Providers

### Together.ai

**Setup:**
```bash
export TOGETHER_API_KEY="your-api-key"
```

**Available Hardware:**
- nvidia_a100_80gb
- nvidia_h100_80gb
- (Run `oumi deploy list-hardware --provider together` for full list)

### Fireworks.ai

**Setup:**
```bash
export FIREWORKS_API_KEY="your-api-key"
export FIREWORKS_ACCOUNT_ID="your-account-id"
```

**Available Hardware:**
- nvidia_a100_80gb
- nvidia_h100_80gb
- nvidia_h200_141gb
- amd_mi300x
- (Run `oumi deploy list-hardware --provider fireworks` for full list)

## Common Workflows

### Deploy a fine-tuned model

1. Train your model using `oumi train`
2. Upload to S3/GCS
3. Deploy:

```bash
oumi deploy up \
  --model-path s3://my-bucket/models/my-finetuned-model/ \
  --provider together \
  --hardware nvidia_a100_80gb \
  --wait
```

### Deploy a LoRA adapter

```bash
oumi deploy up \
  --model-path s3://my-bucket/adapters/my-lora/ \
  --provider together \
  --model-name "my-lora-adapter" \
  --model-type adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --wait
```

### Update autoscaling settings

Currently, you need to delete and recreate the endpoint. Future versions will support `oumi deploy update`.

### Monitor deployment

```bash
# Watch deployment progress
oumi deploy status \
  --endpoint-id "ep-123" \
  --provider together \
  --watch

# List all deployments
oumi deploy list --provider together

# List all uploaded models (including pending) from all configured providers
oumi deploy list-models

# List models from a specific provider
oumi deploy list-models --provider together

# Check ongoing uploads across all providers
oumi deploy list-models --status pending
```

### Clean up

```bash
# Delete endpoint only
oumi deploy delete \
  --endpoint-id "ep-123" \
  --provider together \
  --force

# Delete uploaded model (Fireworks only)
oumi deploy delete-model \
  --model-id "model-123" \
  --provider fireworks \
  --force

# Full cleanup (delete both endpoint and model)
# 1. Delete endpoint first
oumi deploy delete \
  --endpoint-id "ep-123" \
  --provider fireworks \
  --force

# 2. Then delete model
oumi deploy delete-model \
  --model-id "model-123" \
  --provider fireworks \
  --force
```

**Note:** Together.ai does not support programmatic model deletion. Models must be deleted through the [Together.ai dashboard](https://api.together.xyz/playground).

## Examples

See the example config files in this directory:
- `together_deploy.yaml` - Deploy to Together.ai
- `fireworks_deploy.yaml` - Deploy to Fireworks.ai
- `lora_adapter_deploy.yaml` - Deploy a LoRA adapter

## Troubleshooting

### Authentication errors

Make sure your API keys are set:
```bash
# Together.ai
export TOGETHER_API_KEY="your-key"

# Fireworks.ai
export FIREWORKS_API_KEY="your-key"
export FIREWORKS_ACCOUNT_ID="your-account-id"
```

### Model upload fails

- Check model format (should be HuggingFace format)
- Verify S3/GCS permissions
- Check model size limits for the provider

### Endpoint creation fails

- Verify model is ready (use `--wait` when uploading)
- Check hardware availability with `oumi deploy list-hardware`
- Review provider-specific requirements

## Future Enhancements

Coming soon:
- Cost estimation before deployment
- Interactive hardware selection
- Endpoint update (autoscaling, hardware)
- Multi-provider deployment
- Canary deployments
- Integrated monitoring and alerting
