from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference import VLLMInferenceEngine
from tests.markers import requires_cuda_initialized, requires_gpus


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=5, use_sampling=False, temperature=0.0, min_p=0.0, seed=42
        )
    )


@requires_cuda_initialized()
@requires_gpus()
def test_qwen_think_block_with_enable_thinking_true():
    convo = Conversation(
        messages=[Message(content="why is the sky blue?", role=Role.USER)]
    )
    engine = VLLMInferenceEngine(_get_default_model_params(), tensor_parallel_size=1)
    inference_config = _get_default_inference_config()
    outputs = engine.infer([convo], inference_config=inference_config)
    output = outputs[-1].messages[-1].content
    print(output)
    assert "<think>" in output


@requires_cuda_initialized()
@requires_gpus()
def test_qwen_no_think_block_with_enable_thinking_false():
    convo = Conversation(
        messages=[Message(content="why is the sky blue?", role=Role.USER)]
    )

    engine = VLLMInferenceEngine(_get_default_model_params(), tensor_parallel_size=1)
    inference_config = _get_default_inference_config()
    inference_config.model.chat_template_kwargs = {"enable_thinking": False}

    outputs = engine.infer([convo], inference_config=inference_config)
    output = outputs[-1].messages[-1].content
    print(output)
    assert "<think>" not in output
