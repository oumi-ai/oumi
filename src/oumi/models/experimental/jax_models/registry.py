# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX Model Registry - OFFICIAL models supported by jax-llm-examples.

Based on upstream supported models with TPU requirements.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class JAXModelInfo:
    """Information about a supported JAX model."""

    model_id: str  # HuggingFace model ID
    architecture: str  # Which JAX implementation to use
    description: str
    size_gb: float | None = None
    requires_auth: bool = False
    recommended_hardware: str | None = None
    notes: str | None = None


# OFFICIAL SUPPORTED MODELS from jax-llm-examples
# ALL models are designed for TPU ("currently runs on TPU, GPU support in-progress")
SUPPORTED_MODELS: dict[str, JAXModelInfo] = {
    # LLAMA3_JAX Architecture (supports Llama 3.1 + DeepSeek R1 Distilled)
    "llama-3.1-8b-instruct": JAXModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        architecture="llama3_jax",
        description="Llama 3.1 8B Instruct model with GQA attention",
        size_gb=16.0,
        requires_auth=True,
        recommended_hardware="TPU v5e-16+",
        notes=(
            "Requires HuggingFace auth. Multi-host TPU cluster for optimal performance."
        ),
    ),
    "llama-3.1-70b-instruct": JAXModelInfo(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        architecture="llama3_jax",
        description="Llama 3.1 70B Instruct model with GQA attention",
        size_gb=140.0,
        requires_auth=True,
        recommended_hardware="TPU v5e-16+",
        notes="Requires HuggingFace auth. Multi-host TPU cluster required.",
    ),
    "llama-3.1-405b-instruct": JAXModelInfo(
        model_id="meta-llama/Llama-3.1-405B-Instruct",
        architecture="llama3_jax",
        description="Llama 3.1 405B Instruct model with GQA attention",
        size_gb=810.0,
        requires_auth=True,
        recommended_hardware="TPU v5e-16+ (multiple)",
        notes="Requires HuggingFace auth. Large-scale multi-host TPU cluster required.",
    ),
    "deepseek-r1-distill-llama-8b": JAXModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        architecture="llama3_jax",  # Uses Llama architecture, not native DeepSeek R1
        description="DeepSeek R1 knowledge distilled into Llama 3.1 8B architecture",
        size_gb=16.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-16+",
        notes="Distilled model using Llama 3.1 architecture for compatibility.",
    ),
    "deepseek-r1-distill-llama-70b": JAXModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        architecture="llama3_jax",  # Uses Llama architecture, not native DeepSeek R1
        description="DeepSeek R1 knowledge distilled into Llama 3.1 70B architecture",
        size_gb=140.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-16+",
        notes="Distilled model using Llama 3.1 architecture for compatibility.",
    ),
    # QWEN3_JAX Architecture
    "qwen3-0.6b": JAXModelInfo(
        model_id="Qwen/Qwen3-0.6B",
        architecture="qwen3_jax",
        description="Qwen 3 0.6B model with MLA attention",
        size_gb=1.2,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Scout supported with defaults.",
    ),
    "qwen3-1.7b": JAXModelInfo(
        model_id="Qwen/Qwen3-1.7B",
        architecture="qwen3_jax",
        description="Qwen 3 1.7B model with MLA attention",
        size_gb=3.4,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Scout supported with defaults.",
    ),
    "qwen3-4b": JAXModelInfo(
        model_id="Qwen/Qwen3-4B",
        architecture="qwen3_jax",
        description="Qwen 3 4B model with MLA attention",
        size_gb=8.0,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Scout supported with defaults.",
    ),
    "qwen3-8b": JAXModelInfo(
        model_id="Qwen/Qwen3-8B",
        architecture="qwen3_jax",
        description="Qwen 3 8B model with MLA attention",
        size_gb=16.0,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Scout supported with defaults.",
    ),
    "qwen3-14b": JAXModelInfo(
        model_id="Qwen/Qwen3-14B",
        architecture="qwen3_jax",
        description="Qwen 3 14B model with MLA attention",
        size_gb=28.0,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Scout supported with defaults.",
    ),
    "qwen3-32b": JAXModelInfo(
        model_id="Qwen/Qwen3-32B",
        architecture="qwen3_jax",
        description="Qwen 3 32B model with MLA attention",
        size_gb=64.0,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Scout supported with defaults.",
    ),
    "qwen3-30b-a3b": JAXModelInfo(
        model_id="Qwen/Qwen3-30B-A3B",
        architecture="qwen3_jax",
        description="Qwen 3 30B MoE model with 3B active parameters",
        size_gb=60.0,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="MoE model. Maverick's defaults tuning in progress.",
    ),
    "qwen3-235b-a22b": JAXModelInfo(
        model_id="Qwen/Qwen3-235B-A22B",
        architecture="qwen3_jax",
        description="Qwen 3 235B MoE model with 22B active parameters",
        size_gb=470.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-16+",
        notes="Large MoE model. Maverick's defaults tuning in progress.",
    ),
    # LLAMA4_JAX Architecture
    "llama-4-scout-17b-16e-instruct": JAXModelInfo(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        architecture="llama4_jax",
        description="Llama 4 Scout 17B model with 16 experts",
        size_gb=34.0,
        requires_auth=True,
        recommended_hardware="TPU",
        notes="Requires HuggingFace auth. MoE architecture.",
    ),
    "llama-4-maverick-17b-128e-instruct": JAXModelInfo(
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        architecture="llama4_jax",
        description="Llama 4 Maverick 17B model with 128 experts",
        size_gb=34.0,
        requires_auth=True,
        recommended_hardware="TPU",
        notes="Requires HuggingFace auth. Large MoE architecture.",
    ),
    # DEEPSEEK_R1_JAX Architecture (native, not distilled)
    "deepseek-r1": JAXModelInfo(
        model_id="deepseek-ai/DeepSeek-R1",
        architecture="deepseek_r1_jax",
        description="DeepSeek R1 with native MLA attention and MoE routing",
        size_gb=671.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-64+",
        notes="Full DeepSeek R1 model with expert parallelism.",
    ),
    # KIMI_K2_JAX Architecture
    "kimi-k2": JAXModelInfo(
        model_id="moonshotai/Kimi-K2-Instruct",
        architecture="kimi_k2_jax",
        description="Kimi K2 1T parameter model with MLA attention",
        size_gb=1000.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-64+",
        notes="Very large model requiring multi-host TPU cluster.",
    ),
    # GPT_OSS_JAX Architecture
    "gpt-oss-20b": JAXModelInfo(
        model_id="openai/gpt-oss-20b",
        architecture="gpt_oss_jax",
        description="GPT OSS 20B with sliding window attention and MoE",
        size_gb=40.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-16+",
    ),
    "gpt-oss-120b": JAXModelInfo(
        model_id="openai/gpt-oss-120b",
        architecture="gpt_oss_jax",
        description="GPT OSS 120B with sliding window attention and MoE",
        size_gb=240.0,
        requires_auth=False,
        recommended_hardware="TPU v5e-16+",
    ),
    # NEMOTRON3_JAX Architecture
    "nemotron3-nano": JAXModelInfo(
        model_id="nvidia/Nemotron-3-Nano",
        architecture="nemotron3_jax",
        description="NVIDIA Nemotron 3 Nano hybrid Mamba-Transformer model",
        size_gb=8.0,
        requires_auth=False,
        recommended_hardware="TPU",
        notes="Hybrid Mamba-Transformer architecture.",
    ),
}


def get_supported_models() -> dict[str, JAXModelInfo]:
    """Get all supported JAX models."""
    return SUPPORTED_MODELS


def get_models_by_architecture() -> dict[str, list[str]]:
    """Group models by their JAX architecture."""
    arch_models = {}
    for model_key, info in SUPPORTED_MODELS.items():
        if info.architecture not in arch_models:
            arch_models[info.architecture] = []
        arch_models[info.architecture].append(model_key)
    return arch_models


def get_model_info(model_key: str) -> JAXModelInfo | None:
    """Get information about a specific model."""
    return SUPPORTED_MODELS.get(model_key)


def list_models_by_size(max_size_gb: float | None = None) -> list[str]:
    """List models under a certain size."""
    if max_size_gb is None:
        return list(SUPPORTED_MODELS.keys())

    return [
        key
        for key, info in SUPPORTED_MODELS.items()
        if info.size_gb is not None and info.size_gb <= max_size_gb
    ]


def list_models_no_auth() -> list[str]:
    """List models that don't require authentication."""
    return [key for key, info in SUPPORTED_MODELS.items() if not info.requires_auth]


def get_implementation_module(architecture: str) -> str:
    """Get the Python module path for a given architecture."""
    arch_modules = {
        "llama3_jax": "oumi.models.experimental.jax_models.llama3.llama3_jax",
        "qwen3_jax": "oumi.models.experimental.jax_models.qwen3.qwen3_jax",
        "llama4_jax": "oumi.models.experimental.jax_models.llama4.llama4_jax",
        "deepseek_r1_jax": (
            "oumi.models.experimental.jax_models.deepseek_r1_jax.deepseek_r1_jax"
        ),
        "kimi_k2_jax": "oumi.models.experimental.jax_models.kimi_k2.kimi_k2_jax",
        "gpt_oss_jax": "oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax",
        "nemotron3_jax": "oumi.models.experimental.jax_models.nemotron3.nemotron3_jax",
    }
    return arch_modules.get(
        architecture, f"oumi.models.experimental.jax_models.{architecture}"
    )


def get_implementation_path(architecture: str) -> str:
    """Get the filesystem path for a given architecture."""
    arch_paths = {
        "llama3_jax": "llama3/llama3_jax",
        "qwen3_jax": "qwen3/qwen3_jax",
        "llama4_jax": "llama4/llama4_jax",
        "deepseek_r1_jax": "deepseek_r1_jax/deepseek_r1_jax",
        "kimi_k2_jax": "kimi_k2/kimi_k2_jax",
        "gpt_oss_jax": "gpt_oss/gpt_oss_jax",
        "nemotron3_jax": "nemotron3/nemotron3_jax",
    }
    base_path = Path(__file__).parent
    return str(base_path / arch_paths.get(architecture, architecture))


def validate_model_name(model_key: str) -> bool:
    """Validate if a model key is supported."""
    return model_key in SUPPORTED_MODELS


def get_recommended_model(
    max_size_gb: float | None = None, requires_no_auth: bool | None = None
) -> str | None:
    """Get a recommended model based on constraints."""
    candidates = list(SUPPORTED_MODELS.keys())

    if max_size_gb is not None:
        candidates = [
            k
            for k in candidates
            if SUPPORTED_MODELS[k].size_gb is not None
            and SUPPORTED_MODELS[k].size_gb <= max_size_gb
        ]

    if requires_no_auth is True:
        candidates = [k for k in candidates if not SUPPORTED_MODELS[k].requires_auth]

    if not candidates:
        return None

    # Prefer smaller models first
    candidates.sort(key=lambda k: SUPPORTED_MODELS[k].size_gb or 0)
    return candidates[0]


def list_supported_architectures() -> list[str]:
    """List all supported JAX architectures."""
    architectures = set()
    for info in SUPPORTED_MODELS.values():
        architectures.add(info.architecture)
    return sorted(list(architectures))
