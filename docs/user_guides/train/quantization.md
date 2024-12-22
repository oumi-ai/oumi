# Peft and Quantization

## Overview

Model quantization is a technique used to reduce the memory footprint and computational requirements of large language models. Oumi supports various quantization methods to help you train and deploy models more efficiently.

```{note}
Quantization can significantly reduce model size and increase inference speed, but it may also impact model performance. Always evaluate your model's performance after quantization.
```

## Using bitsandbytes

Oumi integrates with the `bitsandbytes`, `peft`, and `transformers` libraries to provide efficient quantization methods.

To enable `bitsandbytes` quantization in Oumi, simply install the package and configure your model to use quantization (see the next section for details).

```bash
# Option 1: Install GPU dependencies including bitsandbytes
pip install oumi[gpu]

# Option 2: Install bitsandbytes separately
pip install bitsandbytes
```

## Setting Quantization Parameters

Quantization settings can be configured in the {py:obj}`~oumi.core.configs.PeftParams` class. The main quantization-related parameters are:

- {py:obj}`~oumi.core.configs.PeftParams.q_lora`: Enables QLoRA (Quantized Low-Rank Adaptation)
- {py:obj}`~oumi.core.configs.PeftParams.q_lora_bits`: Number of bits for quantization (e.g., 4 for 4-bit quantization)
- {py:obj}`~oumi.core.configs.PeftParams.bnb_4bit_quant_type`: Quantization type (e.g., "nf4" for 4-bit normal float)

Example configuration:

```python
from oumi.core.configs import TrainingConfig, TrainingParams, ModelParams, PeftParams

config = TrainingConfig(
    model=ModelParams(
        model_name="your_model_name",
        torch_dtype_str="bfloat16",
    ),
    training=TrainingParams(
        use_peft=True,
        # ... other training parameters ...
    ),
    peft=PeftParams(
        q_lora=True,
        q_lora_bits=4,
        bnb_4bit_quant_type="nf4",
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        lora_target_modules=[
            "q_proj",
            "v_proj",
        ],
    ),
    # ... other config options ...
)
```

## Training with QLoRA

QLoRA is a technique that allows you to train models with quantized weights and optimizer states. This significantly reduces the memory footprint of the model during training.

To enable QLoRA:

1. Set `q_lora=True` in your {py:obj}`oumi.core.configs.PeftParams`
2. Configure other QLora parameters as needed

```python
from oumi.core.configs import TrainingConfig, TrainingParams, PeftParams

config = TrainingConfig(
    training=TrainingParams(use_peft=True),
    peft=PeftParams(
        q_lora=True,
        q_lora_bits=4,
        bnb_4bit_quant_type="nf4",
        lora_r=64,
        lora_alpha=128,
    ),
)
```

## Sample Configurations

- [Llama 8B QLoRA](../../configs/recipes/llama3_1/sft/8b_qlora/train.yaml)
- [Llama 3.2 3B QLoRA](../../configs/recipes/llama3_2/sft/3b_qlora/train.yaml)

```{seealso}
For more advanced quantization techniques and configurations, refer to the [bitsandbytes documentation](https://github.com/TimDettmers/bitsandbytes) and the [PEFT library documentation](https://huggingface.co/docs/peft/index).
```
