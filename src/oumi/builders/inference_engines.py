from typing import Optional

from oumi.core.configs import InferenceEngineType, ModelParams, RemoteParams
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SGLangInferenceEngine,
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
    elif engine_type in (
        InferenceEngineType.REMOTE_VLLM,
        InferenceEngineType.SGLANG,
        InferenceEngineType.ANTHROPIC,
        InferenceEngineType.REMOTE,
    ):
        if remote_params is None:
            raise ValueError(
                "remote_params must be configured "
                f"for the '{engine_type}' inference engine in inference config."
            )
        if engine_type == InferenceEngineType.REMOTE_VLLM:
            return RemoteVLLMInferenceEngine(model_params, remote_params)
        elif engine_type == InferenceEngineType.SGLANG:
            return SGLangInferenceEngine(model_params, remote_params)
        elif engine_type == InferenceEngineType.ANTHROPIC:
            return AnthropicInferenceEngine(model_params, remote_params)
        else:
            assert engine_type == InferenceEngineType.REMOTE
            return RemoteInferenceEngine(model_params, remote_params)

    raise ValueError(f"Unsupported inference engine: {engine_type}")
