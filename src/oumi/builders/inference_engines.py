from typing import Optional

from oumi.core.configs import InferenceEngineType, ModelParams, RemoteParams
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import (
    AnthropicInferenceEngine,
    DeepSeekInferenceEngine,
    GoogleGeminiInferenceEngine,
    GoogleVertexInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    OpenAIInferenceEngine,
    ParasailInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SGLangInferenceEngine,
    TogetherInferenceEngine,
    VLLMInferenceEngine,
)


def build_inference_engine(
    engine_type: InferenceEngineType,
    model_params: ModelParams,
    remote_params: Optional[RemoteParams],
) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config."""
    if engine_type == InferenceEngineType.NATIVE:
        return NativeTextInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.VLLM:
        return VLLMInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.LLAMACPP:
        return LlamaCppInferenceEngine(model_params)
    elif engine_type == InferenceEngineType.DEEPSEEK:
        return DeepSeekInferenceEngine(model_params, remote_params)
    elif engine_type == InferenceEngineType.PARASAIL:
        return ParasailInferenceEngine(model_params, remote_params)
    elif engine_type == InferenceEngineType.TOGETHER:
        return TogetherInferenceEngine(model_params, remote_params)
    elif engine_type == InferenceEngineType.OPENAI:
        return OpenAIInferenceEngine(model_params, remote_params)
    elif engine_type == InferenceEngineType.ANTHROPIC:
        return AnthropicInferenceEngine(model_params, remote_params)
    elif engine_type == InferenceEngineType.GOOGLE_GEMINI:
        return GoogleGeminiInferenceEngine(model_params, remote_params)
    elif engine_type in (
        InferenceEngineType.REMOTE_VLLM,
        InferenceEngineType.SGLANG,
        InferenceEngineType.REMOTE,
        InferenceEngineType.GOOGLE_VERTEX,
    ):
        # These inference engines do not have a default remote params configuration,
        # so we need to check that remote_params is provided.
        if remote_params is None:
            raise ValueError(
                "remote_params must be configured "
                f"for the '{engine_type}' inference engine in inference config."
            )
        if engine_type == InferenceEngineType.REMOTE_VLLM:
            return RemoteVLLMInferenceEngine(model_params, remote_params)
        elif engine_type == InferenceEngineType.SGLANG:
            return SGLangInferenceEngine(model_params, remote_params)
        elif engine_type == InferenceEngineType.GOOGLE_VERTEX:
            return GoogleVertexInferenceEngine(model_params, remote_params)
        else:
            return RemoteInferenceEngine(model_params, remote_params)

    raise ValueError(f"Unsupported inference engine: {engine_type}")
