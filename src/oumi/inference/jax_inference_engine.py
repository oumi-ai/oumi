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

"""JAX inference engine for running LLM inference via jax-llm-examples models.

Supports Llama 3, Llama 4, DeepSeek R1, Qwen 3, Kimi K2, GPT-OSS, and Nemotron 3.
Each model follows the upstream jax-llm-examples prefill/decode pattern.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

from oumi.builders import build_tokenizer
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.sharding import PartitionSpec as P
except ModuleNotFoundError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    random = None  # type: ignore[assignment]
    P = None  # type: ignore[assignment]

# Maps model architecture keywords to their module paths and loading patterns.
# Each entry defines: (module_import_path, architecture_family)
_MODEL_ARCHITECTURES = {
    "llama3": (
        "oumi.models.experimental.jax_models.llama3.llama3_jax",
        "llama3",
    ),
    "llama-3": (
        "oumi.models.experimental.jax_models.llama3.llama3_jax",
        "llama3",
    ),
    "llama4": (
        "oumi.models.experimental.jax_models.llama4.llama4_jax",
        "llama4",
    ),
    "llama-4": (
        "oumi.models.experimental.jax_models.llama4.llama4_jax",
        "llama4",
    ),
    "deepseek": (
        "oumi.models.experimental.jax_models.deepseek_r1_jax.deepseek_r1_jax",
        "deepseek_r1",
    ),
    "qwen3": (
        "oumi.models.experimental.jax_models.qwen3.qwen3_jax",
        "qwen3",
    ),
    "qwen-3": (
        "oumi.models.experimental.jax_models.qwen3.qwen3_jax",
        "qwen3",
    ),
    "kimi": (
        "oumi.models.experimental.jax_models.kimi_k2.kimi_k2_jax",
        "kimi_k2",
    ),
    "gpt-oss": (
        "oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax",
        "gpt_oss",
    ),
    "gpt_oss": (
        "oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax",
        "gpt_oss",
    ),
    "nemotron": (
        "oumi.models.experimental.jax_models.nemotron3.nemotron3_jax",
        "nemotron3",
    ),
}

# Models that use cache.iter for first-token extraction (vs cache.length)
_USES_CACHE_ITER = {"llama3", "gpt_oss", "qwen3", "nemotron3"}

# Models that use chkpt_utils.load_model instead of model.load_pytree
_USES_CHKPT_LOAD_MODEL = {"deepseek_r1", "kimi_k2"}

# Models that use optimal_formats for weight loading
_USES_OPTIMAL_FORMATS = {"gpt_oss", "nemotron3"}

# Models that require set_mesh context manager
_USES_SET_MESH = {"llama3", "gpt_oss", "qwen3", "llama4", "nemotron3"}

# Models that use hf_to_jax_config (vs default Config() or llama_to_jax_config)
_HF_CONFIG_MODELS = {"gpt_oss", "qwen3", "llama4", "nemotron3"}
_LLAMA_CONFIG_MODELS = {"llama3"}
_DEFAULT_CONFIG_MODELS = {"deepseek_r1", "kimi_k2"}

# KVCache init: llama3 uses 3 args, all others use 4 (including max_seq_len)
_KVCACHE_3_ARGS = {"llama3"}


class JAXInferenceEngine(BaseInferenceEngine):
    """Engine for running inference with JAX models from jax-llm-examples.

    This engine loads and runs models vendored from Google's jax-llm-examples
    repository, supporting Llama 3/4, DeepSeek R1, Qwen 3, Kimi K2, GPT-OSS,
    and Nemotron 3. Models are served through Oumi's standard inference pipeline.

    The engine handles the full prefill-then-decode autoregressive generation loop
    following each model's upstream patterns.
    """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        tensor_parallel_size: int = -1,
        quantization: str | None = None,
        max_seq_len: int = 2048,
    ):
        """Initializes the JAX inference engine.

        Args:
            model_params: The model parameters. ``model_name`` should be a path
                to a local checkpoint directory containing ``config.json`` and
                converted JAX weights, or a HuggingFace model ID that maps to
                a known architecture.
            generation_params: Parameters for generation.
            tensor_parallel_size: Number of devices for tensor parallelism.
                -1 means use all available devices.
            quantization: Quantization mode (e.g., "int8"). Applied via each
                model's native quantization flags.
            max_seq_len: Maximum sequence length for KV cache allocation.
        """
        super().__init__(model_params=model_params, generation_params=generation_params)

        if not jax:
            raise RuntimeError(
                "JAX is not installed. Please install JAX with: pip install oumi[jax]"
            )

        self._tensor_parallel_size = tensor_parallel_size
        self._quantization = quantization
        self._max_seq_len = max_seq_len

        # Resolved during _load_model
        self._model_module: Any = None  # The model's Python module (e.g., l3jax)
        self._weights: Any = None  # Model weights (Weights dataclass)
        self._config: Any = None  # Model config (Config dataclass)
        self._architecture: str = ""  # Architecture family name
        self._mesh: Any = None  # JAX device mesh
        self._rng_key: Any = None

        self._setup_devices()
        self._load_model()

    def _setup_devices(self) -> None:
        """Sets up JAX devices and creates the device mesh."""
        assert jax is not None
        assert random is not None
        devices = jax.devices()
        if self._tensor_parallel_size <= 0:
            self._tensor_parallel_size = len(devices)

        logger.info(
            f"JAX devices: {len(devices)} available, "
            f"using {self._tensor_parallel_size} for tensor parallelism."
        )

        # Create mesh following jax-llm-examples pattern: (tp_size, 1, 1)
        # with axis names ("x", "y", "z")
        tp = min(self._tensor_parallel_size, len(devices))
        try:
            self._mesh = jax.make_mesh((tp, 1, 1), ("x", "y", "z"))
        except AttributeError:
            # Fallback for older JAX versions without make_mesh
            from jax.sharding import Mesh

            self._mesh = Mesh(np.array(devices[:tp]).reshape(tp, 1, 1), ("x", "y", "z"))

        self._rng_key = random.PRNGKey(42)

    @staticmethod
    def _resolve_architecture(model_name: str) -> tuple[str, str]:
        """Resolves model name to module path and architecture family.

        Args:
            model_name: The model name or path.

        Returns:
            Tuple of (module_import_path, architecture_family).

        Raises:
            ValueError: If the model architecture cannot be determined.
        """
        name_lower = model_name.lower()
        for keyword, (module_path, arch) in _MODEL_ARCHITECTURES.items():
            if keyword in name_lower:
                return module_path, arch

        supported = sorted({v[1] for v in _MODEL_ARCHITECTURES.values()})
        raise ValueError(
            f"Cannot determine JAX architecture for model: {model_name}. "
            f"Supported architectures: {supported}"
        )

    def _load_model(self) -> None:
        """Loads the JAX model, config, and weights."""
        model_name = self._model_params.model_name
        module_path, self._architecture = self._resolve_architecture(model_name)

        # Build tokenizer
        self._tokenizer = build_tokenizer(self._model_params)

        # Import the model module
        import importlib

        try:
            model_mod = importlib.import_module(f"{module_path}.model")
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import JAX model module '{module_path}.model'. "
                "Ensure oumi[jax] is installed: pip install oumi[jax]"
            ) from e
        self._model_module = model_mod

        # Resolve checkpoint path
        ckpt_path = self._resolve_checkpoint_path(model_name)

        if ckpt_path is not None:
            self._load_from_checkpoint(ckpt_path, model_mod, module_path)
        else:
            self._load_with_random_weights(model_mod)

        logger.info(
            f"Loaded JAX model: {model_name} (architecture={self._architecture})"
        )

    def _resolve_checkpoint_path(self, model_name: str) -> Path | None:
        """Resolves the checkpoint path from the model name."""
        # First check if it's a direct path
        candidate = Path(model_name).expanduser()
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate

        # Check model_params for explicit path
        model_path = getattr(self._model_params, "model_path", None)
        if model_path:
            candidate = Path(model_path).expanduser()
            if candidate.is_dir():
                return candidate

        return None

    def _load_config_from_checkpoint(self, ckpt_path: Path, model_mod: Any) -> Any:
        """Creates a JAX config from checkpoint's config.json."""
        config_data = json.loads((ckpt_path / "config.json").read_text())

        if self._architecture in _LLAMA_CONFIG_MODELS:
            cfg = model_mod.llama_to_jax_config(config_data)
        elif self._architecture in _HF_CONFIG_MODELS:
            # Llama4 nests text config under "text_config"
            if self._architecture == "llama4" and "text_config" in config_data:
                config_data = config_data["text_config"]
            cfg = model_mod.hf_to_jax_config(config_data)
        elif self._architecture in _DEFAULT_CONFIG_MODELS:
            cfg = model_mod.Config()
        else:
            raise ValueError(
                f"Unknown config pattern for architecture: {self._architecture}"
            )

        # Apply mesh and quantization
        replace_kwargs: dict[str, Any] = {"mesh": self._mesh}

        if self._quantization:
            quant = True
            if self._architecture == "llama3":
                replace_kwargs.update(quant_layer=quant, quant_cache=quant)
            elif self._architecture in ("qwen3", "nemotron3"):
                replace_kwargs.update(
                    quant_attn=quant,
                    quant_moe=quant,
                    quant_mlp=quant,
                    quant_cache=quant,
                )
                if self._architecture == "nemotron3":
                    replace_kwargs["quant_mamba"] = quant
            elif self._architecture in ("llama4",):
                replace_kwargs.update(
                    quant_attn=quant, quant_moe=quant, quant_mlp=quant
                )
            elif self._architecture in ("gpt_oss",):
                replace_kwargs.update(quant_moe=quant, quant_cache=quant)

        # Set max_seq_len for models that support it
        if self._architecture != "llama3":
            replace_kwargs["max_seq_len"] = self._max_seq_len

        cfg = dataclasses.replace(cfg, **replace_kwargs)
        return cfg

    def _load_weights(
        self, ckpt_path: Path, model_mod: Any, module_path: str, cfg: Any
    ) -> Any:
        """Loads model weights using the architecture-appropriate method."""
        if self._architecture in _USES_CHKPT_LOAD_MODEL:
            # DeepSeek R1 and Kimi K2 use chkpt_utils.load_model
            import importlib

            try:
                from etils import epath  # type: ignore[import-not-found]

                chkpt_mod = importlib.import_module(f"{module_path}.chkpt_utils")
                return chkpt_mod.load_model(epath.Path(str(ckpt_path)), cfg)
            except ImportError:
                chkpt_mod = importlib.import_module(f"{module_path}.chkpt_utils")
                return chkpt_mod.load_model(ckpt_path, cfg)

        elif self._architecture in _USES_OPTIMAL_FORMATS:
            # GPT-OSS and Nemotron3 use optimal_formats
            weights_formats, cache_formats = model_mod.optimal_formats(cfg)
            weights = model_mod.load_pytree(ckpt_path, weights_formats)
            # Store cache formats for GPT-OSS cache resharding
            if self._architecture == "gpt_oss":
                self._cache_formats = cache_formats
            return weights

        else:
            # Llama3, Llama4, Qwen3 use load_pytree with Weights.shardings
            return model_mod.load_pytree(ckpt_path, model_mod.Weights.shardings(cfg))

    def _load_from_checkpoint(
        self, ckpt_path: Path, model_mod: Any, module_path: str
    ) -> None:
        """Loads model from a checkpoint directory."""
        self._config = self._load_config_from_checkpoint(ckpt_path, model_mod)
        self._weights = self._load_weights(
            ckpt_path, model_mod, module_path, self._config
        )
        logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

    def _load_with_random_weights(self, model_mod: Any) -> None:
        """Initializes model with random weights for testing/development."""
        logger.warning(
            "No checkpoint found. Initializing with random weights "
            "(suitable for testing only)."
        )

        if self._architecture in _DEFAULT_CONFIG_MODELS:
            cfg = model_mod.Config()
        elif self._architecture == "llama3":
            # Create a minimal Llama3 config for testing
            config_dict = {
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "head_dim": 64,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
            }
            cfg = model_mod.llama_to_jax_config(config_dict)
        else:
            raise ValueError(
                f"Cannot initialize {self._architecture} with random weights. "
                "Please provide a checkpoint path."
            )

        cfg = dataclasses.replace(cfg, mesh=self._mesh)
        self._config = cfg
        self._weights = model_mod.Weights.init(self._rng_key, cfg)

    @override
    def get_supported_params(self) -> set[str]:
        """Returns supported generation parameters for JAX engine."""
        return {
            "max_new_tokens",
            "temperature",
            "top_p",
            "do_sample",
            "stop_token_ids",
        }

    @override
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference on conversations.

        Args:
            input: Conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List of conversations with generated assistant responses.
        """
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )

        assert self._tokenizer is not None
        assert jnp is not None

        output_conversations = []
        for conversation in input:
            # Apply chat template
            prompt = self._tokenizer.apply_chat_template(
                conversation.to_dict()["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(prompt, str):
                raise RuntimeError(
                    f"`apply_chat_template` returned a non-string. Type: {type(prompt)}"
                )

            # Tokenize
            input_ids = self._tokenizer.encode(prompt, return_tensors="np")
            if isinstance(input_ids, list):
                input_ids = np.array([input_ids])
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]

            input_jax = jnp.array(input_ids)

            # Generate
            max_new_tokens = generation_params.max_new_tokens
            generated_tokens = self._generate_tokens(input_jax, max_new_tokens)

            # Decode only the new tokens
            generated_text = self._tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Create new Conversation (following NativeTextInferenceEngine pattern)
            messages = [
                *conversation.messages,
                Message(role=Role.ASSISTANT, content=generated_text),
            ]
            new_conversation = Conversation(
                messages=messages,
                metadata=conversation.metadata,
                conversation_id=conversation.conversation_id,
            )
            self._save_conversation_to_scratch(
                new_conversation,
                inference_config.output_path if inference_config else None,
            )
            output_conversations.append(new_conversation)

        return output_conversations

    def _generate_tokens(
        self,
        input_ids: Any,
        max_new_tokens: int,
    ) -> list[int]:
        """Runs the prefill-then-decode generation loop.

        Follows the upstream jax-llm-examples pattern for each model architecture.

        Args:
            input_ids: Input token IDs as JAX array [batch_size, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            List of generated token IDs (new tokens only, no prompt).
        """
        assert jax is not None
        assert jnp is not None
        assert P is not None
        assert self._tokenizer is not None
        _jax = jax  # local ref for lambdas (pyright can't narrow closures)

        mod = self._model_module
        cfg = self._config
        weights = self._weights
        batch_size = input_ids.shape[0]

        # Initialize KV cache
        if self._architecture in _KVCACHE_3_ARGS:
            zero_cache = mod.KVCache.init(self._rng_key, cfg, batch_size)
        else:
            zero_cache = mod.KVCache.init(
                self._rng_key, cfg, batch_size, self._max_seq_len
            )

        # GPT-OSS: reshard cache with optimal formats
        if self._architecture == "gpt_oss" and hasattr(self, "_cache_formats"):
            zero_cache = _jax.tree.map(
                lambda x, sds: _jax.device_put(x, sds.sharding),
                zero_cache,
                self._cache_formats,
            )

        # Optional set_mesh context
        set_mesh_ctx = None
        if self._architecture in _USES_SET_MESH:
            try:
                from jax.sharding import set_mesh
            except ImportError:
                try:
                    from jax.sharding import (
                        use_mesh as set_mesh,  # type: ignore[attr-defined]
                    )
                except ImportError:
                    set_mesh = None

            if set_mesh is not None:
                set_mesh_ctx = set_mesh(cfg.mesh)

        if set_mesh_ctx is not None:
            set_mesh_ctx.__enter__()

        try:
            # Prefill
            next_tokens, logits, cache = mod.prefill(
                input_ids, weights, zero_cache, cfg
            )

            # Extract first generated token
            if self._architecture in _USES_CACHE_ITER:
                curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
                    out_sharding=P(None, None)
                )
            else:
                curr_tokens = next_tokens[:, cache.length - 1 : cache.length]

            # Decode loop
            tokens_list = []
            eos_token_id = self._tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                tokens_list.append(curr_tokens)
                curr_tokens, cache = mod.decode_step(curr_tokens, weights, cache, cfg)

                # Check EOS
                if eos_token_id is not None:
                    token_val = int(curr_tokens[0, 0])
                    if token_val == eos_token_id:
                        break

        finally:
            if set_mesh_ctx is not None:
                set_mesh_ctx.__exit__(None, None, None)

        # Concatenate and convert to Python list
        if tokens_list:
            all_tokens = jnp.concatenate(tokens_list, axis=-1)
            return np.array(all_tokens[0]).tolist()
        return []

    def cleanup(self) -> None:
        """Releases JAX model resources."""
        self._model_module = None
        self._weights = None
        self._config = None
        self._tokenizer = None
        logger.info("JAX inference engine resources cleaned up.")
