# Oumi v0.5: Hyperparameter Tuning, AWS Bedrock, and Advanced RL Training

We're thrilled to announce Oumi v0.5, our most feature-rich release yet! This version introduces powerful hyperparameter optimization, seamless AWS integration, knowledge distillation capabilities, and enhanced reinforcement learning workflows. Whether you're fine-tuning on HPC clusters or scaling with cloud infrastructure, Oumi v0.5 has you covered.

## What's New in v0.5

### 1. Hyperparameter Tuning with `oumi tune`

Finding the right hyperparameters can be the difference between a mediocre model and state-of-the-art performance. Oumi v0.5 introduces `oumi tune`, a built-in hyperparameter search module powered by Optuna that makes systematic optimization effortless.

#### Quick Start

First, install the tuning dependencies:
```bash
pip install oumi[tune]
```

Create a tuning configuration file (e.g., `tune.yaml`):
```yaml
model:
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  model_max_length: 2048
  torch_dtype_str: "bfloat16"

data:
  train:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train[90%:]"
  validation:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train[:10%]"

tuning:
  n_trials: 10

  # Define hyperparameters to search
  tunable_training_params:
    optimizer:
      type: categorical
      choices: ["adamw_torch", "sgd", "adafactor"]

    learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-2

    warmup_ratio:
      type: uniform
      low: 0.0
      high: 0.3

    gradient_accumulation_steps:
      type: int
      low: 1
      high: 8

  # Fixed parameters (not tuned)
  fixed_training_params:
    trainer_type: TRL_SFT
    per_device_train_batch_size: 1
    num_train_epochs: 3
    max_steps: 1000

  # Optimize for best validation loss and token accuracy
  evaluation_metrics: ["eval_loss", "eval_mean_token_accuracy"]
  evaluation_direction: ["minimize", "maximize"]

  tuner_type: OPTUNA
  tuner_sampler: "TPESampler"
```

Run tuning with a single command:
```bash
oumi tune -c tune.yaml
```

#### Python API

Prefer Python? Use the programmatic API:

```python
from oumi import tune
from oumi.core.configs import TuningConfig, TuningParams, ModelParams

config = TuningConfig(
    model=ModelParams(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_max_length=1024,
    ),
    tuning=TuningParams(
        n_trials=10,
        tunable_training_params={
            "learning_rate": {
                "type": "loguniform",
                "low": 1e-5,
                "high": 1e-4,
            },
            "per_device_train_batch_size": {
                "type": "categorical",
                "choices": [2, 4, 8],
            },
        },
        tunable_peft_params={
            "lora_r": {
                "type": "categorical",
                "choices": [4, 8, 16],
            },
        },
        evaluation_metrics=["eval_loss"],
        evaluation_direction=["minimize"],
        output_dir="./tuning_output",
    ),
)

tune(config)
```

Results are saved to `{output_dir}/trials_results.csv` with detailed metrics for each trial. The best model checkpoints are automatically saved for immediate use.

---

### 2. AWS Bedrock Integration

Deploy and scale your inference workloads with AWS Bedrock, now fully integrated into Oumi. Access Claude, Llama, and other foundation models through AWS infrastructure without managing your own inference servers.

#### Setup

Install boto3 and configure AWS credentials:
```bash
pip install boto3
export AWS_REGION=us-east-1
```

Configure your AWS credentials via AWS CLI, environment variables, or IAM roles.

#### Basic Inference

```python
from oumi.inference import BedrockInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams, GenerationParams
from oumi.core.types.conversation import Conversation, Message, Role

# Initialize the engine
engine = BedrockInferenceEngine(
    model_params=ModelParams(
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0"
    ),
    generation_params=GenerationParams(
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    ),
    remote_params=RemoteParams(
        num_workers=10,          # Parallel requests
        max_retries=3,
        connection_timeout=300.0
    )
)

# Run inference
conversations = [
    Conversation(messages=[
        Message(content="You are a helpful assistant.", role=Role.SYSTEM),
        Message(content="Explain quantum computing in simple terms.", role=Role.USER)
    ])
]

results = engine.infer(conversations)
print(results[0].message.content)
```

#### Multimodal Support

Bedrock integration supports text and images, including S3 URIs:

```python
from oumi.core.types.conversation import ContentItem, ImageContentItem, TextContentItem

conversation = Conversation(messages=[
    Message(
        content=[
            ImageContentItem(content="s3://my-bucket/diagram.jpg"),
            TextContentItem(content="Explain what's happening in this image")
        ],
        role=Role.USER
    )
])

results = engine.infer([conversation])
```

#### CLI Usage

```bash
oumi infer --engine BEDROCK \
  --model.model_name anthropic.claude-3-5-sonnet-20240620-v1:0 \
  --generation.max_new_tokens 1024 \
  --generation.temperature 0.7
```

---

### 3. Knowledge Distillation with GKD Trainer

Model compression just got easier with support for Generalized Knowledge Distillation (GKD). Train smaller, faster models that maintain the capabilities of larger teachers using on-policy distillation.

#### What is GKD?

GKD implements on-policy distillation where the student model generates outputs and learns from teacher corrections in real-time. This approach is based on the paper ["On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"](https://arxiv.org/abs/2306.13649).

#### Configuration

Create a GKD training config:

```yaml
model:
  # Student model - smaller model to train
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  model_max_length: 1024
  torch_dtype_str: "bfloat16"

data:
  train:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        dataset_kwargs:
          return_conversations: True
          return_conversations_format: "dict"  # Required for GKD

training:
  trainer_type: "TRL_GKD"
  output_dir: "output/gkd_distilled"

  max_steps: 1000
  per_device_train_batch_size: 4
  learning_rate: 1e-4

  # GKD-specific parameters
  gkd:
    # Teacher model - larger model to distill from
    teacher_model_name_or_path: "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Generation settings
    temperature: 0.9
    max_new_tokens: 512

    # Distillation parameters
    lmbda: 0.5          # 50% on-policy, 50% off-policy data
    beta: 0.5           # Jensen-Shannon divergence (symmetric)
    disable_dropout: True
    seq_kd: False       # Use token-level KD (recommended)
```

Run training:
```bash
oumi train -c gkd_config.yaml
```

#### Python API

```python
from oumi import train
from oumi.core.configs import TrainingConfig, TrainingParams, ModelParams
from oumi.core.configs.params.gkd_params import GkdParams
from oumi.core.configs import TrainerType

config = TrainingConfig(
    model=ModelParams(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_max_length=512,
    ),
    training=TrainingParams(
        trainer_type=TrainerType.TRL_GKD,
        max_steps=1000,
        per_device_train_batch_size=4,
        output_dir="output/gkd_training",
        gkd=GkdParams(
            teacher_model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
            temperature=0.9,
            lmbda=0.5,          # Balance on-policy/off-policy
            beta=0.5,           # Symmetric JSD
            max_new_tokens=128,
        ),
    ),
)

train(config)
```

**Key Parameters:**
- `lmbda`: Controls the mix of on-policy (student-generated) vs. off-policy (dataset) examples
- `beta`: Interpolation coefficient for JSD (0.0 = KL divergence, 0.5 = symmetric JSD, 1.0 = reverse KL)
- `temperature`: Sampling temperature for generation

---

### 4. OpenEnv Reinforcement Learning Training

Take your RL workflows to the next level with OpenEnv integration. Train models using environment-based rewards with automatic visualization and vLLM acceleration.

#### GRPO Training with Custom Rewards

Set up Group Relative Policy Optimization with custom reward functions:

```yaml
model:
  model_name: "Qwen/Qwen2-0.5B-Instruct"
  model_max_length: 2048

data:
  train:
    datasets:
      - dataset_name: "trl-lib/ultrafeedback-prompt"
        split: "train[:1000]"

training:
  trainer_type: "TRL_GRPO"
  output_dir: "output/grpo_training"

  # Reward function from environment
  reward_functions: ["env_reward"]

  max_steps: 500
  per_device_train_batch_size: 4

  grpo:
    use_vllm: True                          # Enable vLLM acceleration
    rollout_function: "custom_rollout"      # Your custom rollout
    max_completion_length: 2048
    num_generations: 8                      # Generations per prompt
    temperature: 0.9
    epsilon: 0.2                            # GRPO clipping epsilon

  # Enable reward tracking and visualization
  enable_wandb: True
```

#### Custom Rollout Function

Define custom rollout logic for environment interaction:

```python
from oumi.core.registry import register, RegistryType

@register("custom_rollout", RegistryType.ROLLOUT_FUNCTION)
def custom_rollout(prompts, args, processing_class):
    """
    Custom rollout function that interacts with an environment.

    1. Generate completions via vLLM
    2. Step through environment to get rewards
    3. Return completions with environment rewards
    """
    # Generate completions
    completions = generate_with_vllm(prompts, args)

    # Get rewards from environment
    env_rewards = []
    for prompt, completion in zip(prompts, completions):
        reward = environment.step(prompt, completion)
        env_rewards.append(reward)

    # Attach rewards to completions
    for completion, reward in zip(completions, env_rewards):
        completion.extra_fields["env_reward"] = reward

    return completions
```

#### Custom Reward Function

```python
@register("env_reward", RegistryType.REWARD_FUNCTION)
def reward_from_env(completions, **kwargs):
    """Extract environment rewards from completions."""
    env_rewards = kwargs.get("env_reward", [])
    return [float(reward) for reward in env_rewards]
```

#### Reward Visualization

When you enable W&B logging, Oumi automatically tracks:
- Average rewards over training steps
- Reward standard deviation
- Individual reward function values
- Completion length statistics
- KL divergence from reference policy

Train and watch your model improve:
```bash
oumi train -c grpo_config.yaml
```

Check out the example notebook at `notebooks/Oumi - OpenEnv GRPO with trl.ipynb` for a complete walkthrough.

---

### 5. HPC Support: NERSC Perlmutter

Scale your training to one of the world's fastest supercomputers. Oumi launcher now supports NERSC Perlmutter with optimized SLURM configurations.

#### Launch Training on Perlmutter

```yaml
launcher:
  cluster: NERSC_PERLMUTTER
  num_nodes: 4
  gpus_per_node: 4
  time_limit: "2:00:00"
  account: "your_allocation"

model:
  model_name: "meta-llama/Llama-2-7b-hf"

training:
  trainer_type: TRL_SFT
  output_dir: "/pscratch/sd/u/username/output"
  # ... training config
```

Launch your job:
```bash
oumi launch -c perlmutter_config.yaml
```

#### Enhanced Logging

New logging features make debugging easier:
- **Job log trailing**: `oumi logs tail <job_id>` to follow logs in real-time
- **Dedicated logs command**: `oumi logs show <job_id>` for comprehensive log viewing
- **Lazy cloud initialization**: Faster launcher startup times

---

## Additional Improvements

### Data & Synthesis
- New synthesis documentation with example configurations
- Oumi dataset support in synthesis workflows
- Improved error handling for document processing
- Refactored analysis module with enhanced statistics computation

### Model Support
- Added Qwen3 VL 4B model configurations
- Exposed `chat_template_kwargs` in ModelParams for fine-grained template control
- Dictionary support for image IDs in multimodal workflows

### Developer Experience
- Updated BaseConfig to support non-primitive field types
- Unique inference scratch filenames via hashing
- Optional `stdout_file` parameter in SLURM client

### Bug Fixes
- Fixed NaN values in dataset analyzer for single-conversation datasets
- Resolved SLURM environment variable issues (PMI_RANK → SLURM_PROCID)
- Corrected GKD trainer initialization issues
- Made `chat_template_kwargs` and `image_id_map` optional for backwards compatibility
- Fixed SkyPilot and GPU test suite failures

### Dependency Updates
- Upgraded transformers: 4.56 → 4.57
- Upgraded TRL: 0.24.0 → 0.25
- Improved SkyPilot compatibility

---

## New Contributors

A huge welcome to our new contributors who helped make v0.5 possible:
- @gbladislau
- @oumiandy
- @AliliRayane

Thank you for your contributions!

---

## Get Started with Oumi v0.5

### Installation

```bash
# Core installation
pip install oumi

# With hyperparameter tuning
pip install oumi[tune]

# Full installation with all features
pip install oumi[all]
```

### Documentation

- [Hyperparameter Tuning Guide](https://oumi.ai/docs/en/latest/user_guides/tune/tune.html)
- [GKD Training Documentation](https://huggingface.co/docs/trl/main/en/gkd_trainer)
- [Inference Engines](https://oumi.ai/docs/en/latest/user_guides/infer/inference_engines.html)
- [GRPO Training Methods](https://oumi.ai/docs/en/latest/user_guides/train/training_methods.html)

### Example Configs

Check out the example configs in the repository:
- `configs/recipes/smollm/tuning/135m/tune.yaml` - Hyperparameter tuning
- `configs/examples/gkd/train.yaml` - Knowledge distillation
- `configs/examples/grpo_verl_gsm8k/train.yaml` - GRPO training
- `notebooks/Oumi - OpenEnv GRPO with trl.ipynb` - OpenEnv RL tutorial

### Full Changelog

For a complete list of changes, see the [full changelog](https://github.com/oumi-ai/oumi/compare/v0.4.0...0.5).

---

## What's Next?

We're constantly improving Oumi based on your feedback. Have ideas or feature requests? Open an issue on [GitHub](https://github.com/oumi-ai/oumi/issues) or join our community discussions.

Happy training!

— The Oumi Team
