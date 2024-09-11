"""Inference module for the LeMa (Learning Machines) library.

This module provides various implementations for running model inference.
"""

from lema.inference.native_text_inference_engine import NativeTextInferenceEngine
from lema.inference.vllm_inference_engine import VLLMInferenceEngine

__all__ = [
    "NativeTextInferenceEngine",
    "VLLMInferenceEngine",
]
