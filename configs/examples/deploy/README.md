# Oumi Deploy - Model Deployment CLI

Deploy trained models to Fireworks.ai for production inference.

## Quick Start

### Deploy a model with config file

```bash
# Deploy to Fireworks.ai
oumi deploy up --config configs/examples/deploy/fireworks_deploy.yaml
```

### Deploy a model with CLI arguments

```bash
# Upload + create endpoint in one command
oumi deploy up \
  --model-path s3://my-bucket/models/my-model/ \
  --provider fireworks \
  --hardware nvidia_a100_80gb
```

## Commands

### Upload a model

```bash
oumi deploy upload \
  --model-path s3://bucket/model/ \
  --provider fireworks \
  --model-name "my-model" \
  --wait
```

### Create an endpoint

```bash
oumi deploy create-endpoint \
  --model-id "fireworks-model-id" \
  --provider fireworks \
  --hardware nvidia_a100_80gb \
  --gpu-count 2 \
  --min-replicas 1 \
  --max-replicas 4 \
  --wait
```

### Check deployment status

```bash
# Get status once
oumi deploy status \
  --endpoint-id "ep-123" \
  --provider fireworks

# Watch until ready
oumi deploy status \
  --endpoint-id "ep-123" \
  --provider fireworks \
  --watch
```

### List all deployments

```bash
oumi deploy list --provider fireworks
```

### List all uploaded models

```bash
# List models from Fireworks
oumi deploy list-models --provider fireworks

# Include public platform models too
oumi deploy list-models --provider fireworks --all

# List only ongoing uploads (pending/processing)
oumi deploy list-models --provider fireworks --status pending

# List ready models
oumi deploy list-models --provider fireworks --status ready
```

### Start / Stop endpoints (cost savings)

```bash
# Stop an endpoint (scales to 0 replicas)
oumi deploy stop --endpoint-id ep-123 --provider fireworks

# Start a stopped endpoint
oumi deploy start --endpoint-id ep-123 --provider fireworks --min-replicas 2
```

**Note:** Start/stop is currently only supported by providers that allow
scaling to zero replicas.

### Delete an endpoint

```bash
oumi deploy delete \
  --endpoint-id "ep-123" \
  --provider fireworks
```

### Delete an uploaded model

```bash
# Delete a Fireworks model
oumi deploy delete-model \
  --model-id "my-model" \
  --provider fireworks

# Delete with auto-confirmation
oumi deploy delete-model \
  --model-id "my-model" \
  --provider fireworks \
  --force
```

### List available hardware

```bash
oumi deploy list-hardware --provider fireworks

# Filter by model compatibility
oumi deploy list-hardware \
  --provider fireworks \
  --model-id "my-model-id"
```

**Note:** Fireworks hardware list is hardcoded (version `2026-01`). Run the
command above to see the current set of supported accelerators.

### Test endpoint

```bash
oumi deploy test \
  --endpoint-id "ep-123" \
  --provider fireworks \
  --prompt "Hello, how are you?" \
  --max-tokens 100
```

The test command sends an OpenAI-compatible chat completion request to the
deployed endpoint and prints the response.

### One-command deploy

```bash
# Deploy with config file
oumi deploy up --config configs/examples/deploy/fireworks_deploy.yaml

# Deploy with CLI args
oumi deploy up \
  --model-path s3://bucket/model/ \
  --provider fireworks \
  --hardware nvidia_a100_80gb \
  --wait
```

## Configuration File Format

```yaml
# Model source (required)
model_source: s3://my-bucket/models/my-model/

# Provider (required): "fireworks"
provider: fireworks

# Model name (required)
model_name: my-finetuned-model-v1

# Model type: "full" or "adapter"
model_type: full

# Base model (required if model_type is "adapter")
# base_model: accounts/fireworks/models/llama-v3p1-8b-instruct

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

## Provider Setup

### Fireworks.ai

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
  --provider fireworks \
  --hardware nvidia_a100_80gb \
  --wait
```

### Clean up

```bash
# Delete endpoint first
oumi deploy delete \
  --endpoint-id "ep-123" \
  --provider fireworks \
  --force

# Then delete model
oumi deploy delete-model \
  --model-id "model-123" \
  --provider fireworks \
  --force
```

## Examples

See the example config files in this directory:
- `fireworks_deploy.yaml` - Deploy to Fireworks.ai

## Troubleshooting

### Authentication errors

Make sure your API keys are set:
```bash
export FIREWORKS_API_KEY="your-key"
export FIREWORKS_ACCOUNT_ID="your-account-id"
```

### Model upload fails

- Check model format (should be HuggingFace format)
- Verify S3/GCS permissions
- Check model size limits for the provider

### Endpoint creation fails

- Verify model is ready (use `--wait` when uploading)
- Check hardware availability with `oumi deploy list-hardware --provider fireworks`
- Review provider-specific requirements
