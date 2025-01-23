# Custom Model

The quality of the judgments is heavily influenced by the underlying model that powers our judge, as well as how well it aligns with your specific use case. Selecting the right model involves balancing factors such as speed, accuracy, resource usage, and cost. Smaller models, such as [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) or [Llama-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), are often more cost-effective and can be hosted locally on modest GPUs. In contrast, larger models typically require more powerful hardware (e.g., A100 GPUs), which may necessitate access to a {doc}`remote cluster </user_guides/launch/launch>`. Then, proprietary models from companies like OpenAI, Google, or Anthropic may offer additional performance benefits, but at the cost of per-token pricing.

The choice of the judge model should also align with your specific application needs. For instance, Anthropic's models are renowned for their safety focus, with Claude being particularly adept at coding tasks. OpenAI's ChatGPT is widely regarded as the leader in conversational AI, while Google’s MedPaLM is increasingly recognized for its effectiveness in medical domains. Regardless of your use case or budget, our judge framework provides flexibility, allowing you to select any model, host it on your preferred platform (or connect via a remote API), all while optimizing configuration to best suit your needs.

## Core Components

A custom judge is defined using a {py:class}`~oumi.core.configs.JudgeConfig`, which specifies the attribute(s) to be evaluated (for more details, refer to the {doc}`Custom Prompts </user_guides/judge/custom_prompt>` page) along with the underlying model. The model definition includes several key components:

1. **Model Parameters** ({py:class}`~oumi.core.configs.ModelParams`):
   - **Model selection**: Specify the model to be used, either loaded from a local path or sourced from a platform like HuggingFace.
   - **Model-specific settings**: Configure essential components such as the tokenizer, chat template, attention mechanism, and more.
   - **Adapter integration**: Apply an adapter model on top of the base model for task-specific customization.
   - **Model sharding**: Enable model sharding across GPUs, allowing for efficient distribution of model layers across available devices.

2. **Inference Engine** ({py:class}`~oumi.core.configs.InferenceEngineType`):
   - **Local engines**: Host models locally for improved speed and privacy (e.g., via vLLM, LlamaCPP, SGLang).
   - **Remote engines**: Query API-based models for enhanced accuracy (e.g., OpenAI, Anthropic, Google).

3. **Generation Parameters** ({py:class}`~oumi.core.configs.GenerationParams`):
   - **Key settings**: Control key aspects of the model’s output, such as maximum token length, temperature, stop tokens, and others.
   - **Advanced techniques**: Leverage guided decoding and beam search to refine output quality and improved coherence.
   - **Performance options**: Tweak performance-related settings like batch size, number of beams for beam search.
   - **Repetition Penalties**: Enforce token frequency or presence penalties to enhance diversity in the generated text.
   - **Sampling parameters**: Adjust sampling parameters (min_p, top_p, etc.) to refine the generation behavior.

4. **Remote Parameters** ({py:class}`~oumi.core.configs.RemoteParams`):
   - **API connection**: Manage API connections and handle authentication for remote model access.
   - **Rate limiting**: Control request rate limits based on politeness policies to ensure fair usage and prevent overloading services.
   - **Parallel processing**: Enable parallel processing across multiple workers, optimizing performance and reducing response time.

## Examples

Below are two example configurations: one for a local judge utilizing a quantized Llama 3B, and another for a remote judge powered by a GPT model. These examples serve as a starting point for evaluating performance and accuracy trade-offs.

### Fast Local Judge

```python
from oumi.core.configs import (
    JudgeAttribute,
    JudgeConfig,
    ModelParams,
    GenerationParams,
    InferenceEngineType,
)
from oumi.judges.oumi_judge import OumiXmlJudge

# Load an existing attribute.
my_attribute = JudgeAttribute.load("<oumi src>/judges/oumi_v1/helpful.json")

# Create a judge configuration for a local GGUF model.
local_config = JudgeConfig(
    attributes={"my_attribute": my_attribute},
    model=ModelParams(
        model_name="bartowski/Llama-3.2-3B-Instruct-GGUF",
        model_kwargs={"filename": "Llama-3.2-3B-Instruct-Q8_0.gguf"},  # 3.42 GB
        model_max_length=4096,
        torch_dtype_str="bfloat16",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    generation=GenerationParams(
        max_new_tokens=4096,
        batch_size=4,
        seed=1234,
        temperature=0.5
    )
    engine=InferenceEngineType.LLAMACPP,
)

# Instantiate the judge.
judge = OumiXmlJudge(config=local_config)
```

### Accurate Remote Judge

```python
from oumi.core.configs import (
    JudgeAttribute,
    JudgeConfig,
    ModelParams,
    GenerationParams,
    InferenceEngineType,
    RemoteParams,
)
from oumi.judges.oumi_judge import OumiXmlJudge

# Load an existing attribute.
my_attribute = JudgeAttribute.load("<oumi src>/judges/oumi_v1/helpful.json")

# Create a judge configuration for for GPT-4
remote_config = JudgeConfig(
    attributes={"my_attribute": my_attribute},
    model=ModelParams(model_name="gpt-4"),
    generation=GenerationParams(
        max_new_tokens=2048,
        temperature=0.0,
    ),
    engine=InferenceEngineType.REMOTE,
    remote_params=RemoteParams(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key_env_varname="OPENAI_API_KEY",
        max_retries=3,
    )
)

# Instantiate the judge.
judge = OumiXmlJudge(config=remote_config)
```
