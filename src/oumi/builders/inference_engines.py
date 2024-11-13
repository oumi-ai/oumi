from oumi.core.configs import InferenceEngineType, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
    SGLangInferenceEngine,
    VLLMInferenceEngine,
)


def build_inference_engine(
    engine_type: InferenceEngineType, model_params: ModelParams
) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config."""
    if engine_type == InferenceEngineType.NATIVE:
        return NativeTextInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.VLLM:
        return VLLMInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.SGLANG:
        return SGLangInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.LLAMACPP:
        return LlamaCppInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.ANTHROPIC:
        return AnthropicInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.REMOTE:
        return RemoteInferenceEngine(model_params)

    raise ValueError(f"Unsupported inference engine: {engine_type}")
