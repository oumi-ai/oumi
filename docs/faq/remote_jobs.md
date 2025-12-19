# Remote Jobs FAQ

Common questions and solutions for running jobs on cloud and remote clusters.

## Setup Issues

### How do I set up cloud credentials?

Install the cloud provider extras and configure credentials:

```bash
# Install cloud dependencies
pip install "oumi[gcp]"     # Google Cloud
pip install "oumi[aws]"     # AWS
pip install "oumi[azure]"   # Azure
pip install "oumi[lambda]"  # Lambda Labs
pip install "oumi[runpod]"  # RunPod

# Check which clouds are enabled
sky check
```

Follow the provider-specific setup in [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup).

### sky check shows my cloud as not enabled

Make sure you have:

1. Installed the cloud SDK (e.g., `gcloud`, `aws`, `az`)
2. Authenticated with the cloud provider
3. Set up proper IAM permissions

```bash
# GCP
gcloud auth login
gcloud auth application-default login

# AWS
aws configure

# Azure
az login
```

### How do I check my cloud quotas?

```bash
# View available resources
sky show-gpus

# Check specific cloud
sky show-gpus --cloud gcp
sky show-gpus --cloud aws
```

## File Mount Issues

### Job fails with "File mount source does not exist"

```text
ValueError: File mount source '~/.netrc' does not exist locally.
```

This means your config references a file that doesn't exist. Common files:

| File | Purpose | Solution |
|------|---------|----------|
| `~/.netrc` | WandB credentials | Run `wandb login` or disable WandB |
| `~/.cache/huggingface/token` | HuggingFace auth | Run `huggingface-cli login` |
| `~/.aws/credentials` | AWS access | Run `aws configure` |

Or remove the file mount from your job config if not needed.

### How do I mount cloud storage?

Add storage mounts to your job config:

```yaml
storage_mounts:
  /gcs_dir:
    source: gs://your-bucket-name
    store: gcs
  /s3_dir:
    source: s3://your-bucket-name
    store: s3
```

### How do I save outputs to cloud storage?

Configure your training to save to the mounted path:

```yaml
storage_mounts:
  /output:
    source: gs://my-bucket/training-outputs
    store: gcs

run: |
  oumi train -c config.yaml --training.output_dir=/output
```

## Job Management

### How do I submit a job?

Use the `oumi launch` command:

```bash
oumi launch up -c your_job.yaml
```

### How do I check job status?

```bash
# List all jobs
oumi launch status

# View specific cluster
sky status
```

### How do I view job logs?

```bash
# Stream logs from a running job
sky logs cluster-name

# Or SSH into the cluster
sky ssh cluster-name
```

### How do I cancel a running job?

```bash
# Stop the cluster (preserves data)
sky stop cluster-name

# Terminate completely
sky down cluster-name
```

### How do I resume a stopped job?

```bash
# Restart a stopped cluster
sky start cluster-name
```

## Resource Configuration

### How do I request specific GPUs?

Specify accelerators in your job config:

```yaml
resources:
  cloud: gcp
  accelerators: A100:4         # 4x A100 GPUs
  # Or specific GPU type
  accelerators: "A100-80GB:2"  # 2x A100 80GB
```

### What GPUs are available?

Check available GPUs and their prices:

```bash
sky show-gpus

# Common GPU types:
# - A100, A100-80GB
# - H100
# - V100
# - T4
# - L4
```

### How do I use spot instances?

Enable spot instances for cost savings (with preemption risk):

```yaml
resources:
  use_spot: true
```

```{warning}
Spot instances can be preempted! Always save checkpoints to cloud storage.
```

### How do I run multi-node training?

Set `num_nodes` in your job config:

```yaml
num_nodes: 4  # 4 nodes

run: |
  torchrun --nproc_per_node=8 --nnodes=4 \
    --node_rank=$RANK --master_addr=$MASTER_ADDR \
    oumi train -c config.yaml
```

## Cost Management

### How do I estimate job costs?

Use `sky cost-report` to view historical costs:

```bash
sky cost-report
```

### How do I minimize costs?

1. **Use spot instances** when possible
2. **Auto-stop idle clusters**:

    ```yaml
    resources:
      idle_minutes_to_autostop: 30
    ```

3. **Right-size resources**: Don't request more GPUs than needed
4. **Use smaller instance types** for development

### How do I set up auto-stop?

Prevent runaway costs with auto-stop:

```yaml
resources:
  idle_minutes_to_autostop: 30  # Stop after 30 min idle
```

Or set cluster-wide:

```bash
sky autostop cluster-name -i 30
```

## Debugging

### Job fails during setup

Check the setup logs:

```bash
sky logs cluster-name --setup
```

Common issues:

1. **Package installation failures**: Check pip/conda commands
2. **Disk space**: Increase `disk_size` in config
3. **Network issues**: Retry with `--retry-until-up`

### Job hangs or is very slow

1. **Check GPU utilization**:

    ```bash
    sky ssh cluster-name
    nvidia-smi
    ```

2. **Check for network bottlenecks** (slow data loading)
3. **Verify you got the expected hardware**

### How do I debug interactively?

SSH into the cluster:

```bash
sky ssh cluster-name

# Activate the Oumi environment
source ~/oumi-env/bin/activate

# Run commands interactively
oumi train -c config.yaml
```

## HuggingFace Integration

### How do I access gated models on the cloud?

Mount your HuggingFace token:

```yaml
file_mounts:
  ~/.cache/huggingface/token: ~/.cache/huggingface/token
```

Make sure you've accepted the model's license on HuggingFace and run:

```bash
huggingface-cli login
```

## WandB Integration

### How do I enable WandB logging on remote jobs?

1. Create `~/.netrc` with your WandB credentials:

    ```bash
    wandb login
    ```

2. Mount it in your job config:

    ```yaml
    file_mounts:
      ~/.netrc: ~/.netrc

    envs:
      WANDB_PROJECT: my-project
    ```

## See Also

- {doc}`/user_guides/launch/launch` - Remote jobs guide
- {doc}`/user_guides/launch/custom_cluster` - Custom cluster setup
- {doc}`/user_guides/launch/kubernetes` - Kubernetes deployment
- {doc}`troubleshooting` - General troubleshooting
