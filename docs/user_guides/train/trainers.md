# Trainers

## Overview

Oumi provides a flexible training framework with multiple trainer implementations to suit different needs. The trainers are designed to handle various tasks, from simple supervised fine-tuning to more advanced techniques like Direct Preference Optimization (DPO).

The main trainers available are:

1. HuggingFace Trainers:
   - {external:py:obj}`trl.SFTTrainer`: Supervised Fine-Tuning Trainer
   - {external:py:obj}`trl.DPOTrainer`: Direct Preference Optimization Trainer
   - {external:py:obj}`transformers.Trainer`: HuggingFace's default Trainer
2. Custom Trainers
   - {py:obj}`oumi.core.trainers.Trainer`: Trainer implementation for pretraining.

All trainers in Oumi inherit from the {py:obj}`oumi.core.trainers.BaseTrainer` class, which defines the common interface for trainers. Stay tuned in future releases for even more training options.

```{note}
The choice of trainer depends on your specific use case, model architecture, and training requirements.
```

### Example usage

```python
from oumi.core.configs import TrainingConfig, TrainerType

config = TrainingConfig(
    # ... other config options ...
    training=TrainingParams(trainer_type=TrainerType.TRL_SFT)
)
```

Or, in your yaml config:

```yaml
training:
  trainer_type: "TRL_SFT"
```

### HuggingFace Trainers

#### Transformers Trainer

The {py:obj}`~oumi.core.trainers.HuggingFaceTrainer` is a wrapper around the HuggingFace `Trainer` class from the `transformers` library. It provides a familiar interface for those accustomed to the HuggingFace ecosystem and seamless integration with HuggingFace models and datasets.

For more details on the underlying HuggingFace Transformers Trainer, refer to the [Trainer documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).

#### TRL SFT Trainer

The TRL SFT (Supervised Fine-Tuning) Trainer is accessed through the {py:obj}`oumi.core.trainers.HuggingFaceTrainer` class. It's specifically designed for fine-tuning language models on instruction-following and conversational tasks.

To use the TRL SFT Trainer, set the `trainer_type` in your {py:obj}`~oumi.core.configs.TrainingConfig` to {py:obj}`~oumi.core.configs.TrainerType.TRL_SFT`.

For more information on the TRL library and SFT training, see the [TRL documentation](https://huggingface.co/docs/trl/sft_trainer).

#### DPO Trainer

The DPO (Direct Preference Optimization) Trainer is accessed through the {py:obj}`oumi.core.trainers.HuggingFaceTrainer` class. It implements the DPO algorithm for fine-tuning language models based on human preferences.

To use the DPO Trainer, set the `trainer_type` in your {py:obj}`~oumi.core.configs.TrainingConfig` to {py:obj}`~oumi.core.configs.TrainerType.TRL_DPO`.

For more details on the DPO algorithm and its implementation, refer to the [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer).

## Oumi Trainers

The {py:obj}`oumi.core.trainers.Trainer`, also known as the Oumi Trainer, is a custom implementation that provides fine-grained control over the training loop for advanced and experimental use cases.

To use the Oumi Trainer, set the `trainer_type` in your {py:obj}`~oumi.core.configs.TrainingConfig` to {py:obj}`~oumi.core.configs.TrainerType.OUMI`.

You can also manually instantiate the Oumi Trainer with your own callbacks and data collators:

```python
from oumi.core.trainers import Trainer
from oumi.core.configs import TrainingParams

params = TrainingParams(...)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

```{warning}
The Oumi Trainer is still under active development and may undergo changes in future releases.
```
