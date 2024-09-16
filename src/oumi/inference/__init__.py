"""Inference module for the LeMa (Learning Machines) library.

This module provides various implementations for running model inference.
"""

from oumi.inference.native_text_inference_engine import NativeTextInferenceEngine
from oumi.inference.vllm_inference_engine import VLLMInferenceEngine

__all__ = [
    "NativeTextInferenceEngine",
    "VLLMInferenceEngine",
]
