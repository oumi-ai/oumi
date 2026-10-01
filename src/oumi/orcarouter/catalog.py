# Copyright 2026 - Oumi
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

"""OrcaRouter model discovery, capability filtering, and verified fallback seed.

``GET {api_base}/v1/models`` is the only source of truth for the model list.
Live discovery is treated as authoritative when it succeeds; a small, verified
fallback seed keeps a fresh installation usable during an outage, and is always
labelled as degraded.

Capabilities are read from catalog metadata, never guessed from a model name.
Model IDs keep their ``vendor/model`` namespace verbatim.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp

from oumi.orcarouter.credentials import (
    OrcaRouterAuthError,
    api_v1_base_url,
    mask_api_key,
)
from oumi.utils.logging import logger

CATALOG_TIMEOUT_SECONDS = 20.0
# Upper bound on a single catalog request.

CATALOG_MAX_BYTES = 512 * 1024
# Upper bound on the catalog response body we are willing to buffer.

CATALOG_MAX_ITEMS = 2000
# Upper bound on the number of catalog records we will consider.

_CATALOG_CHUNK_BYTES = 64 * 1024
# Read granularity for the bounded catalog body.

CHAT_ENDPOINT_TYPES = frozenset({"openai", "openai-response", "anthropic", "gemini"})
# Endpoint types that can serve a text chat completion.

EMBEDDING_ENDPOINT_TYPES = frozenset({"embeddings", "embedding"})
IMAGE_ENDPOINT_TYPES = frozenset({"image-generation"})
VIDEO_ENDPOINT_TYPES = frozenset({"openai-video"})
RERANK_ENDPOINT_TYPES = frozenset({"jina-rerank"})

CAPABILITY_MULTIMODAL_PREFIX = "multimodal"
# Capability keys are ``multimodal:<modality>``, e.g. ``multimodal:image``.

_SUPPORTED_INPUT_MODALITIES = frozenset({"text", "image", "audio", "video"})


class OrcaRouterCatalogError(Exception):
    """Raised when the OrcaRouter model catalog cannot be retrieved."""


class OrcaRouterCapability(str, Enum):
    """A capability an oumi entry point can require from the catalog."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    VIDEO = "video"
    RERANK = "rerank"

    @staticmethod
    def parse(value: str) -> OrcaRouterCapability:
        """Parses a capability string, including ``multimodal:<modality>`` keys.

        Args:
            value: A capability key such as ``chat`` or ``multimodal:image``.

        Returns:
            OrcaRouterCapability: The base capability.

        Raises:
            ValueError: If the value is not a known capability key.
        """
        if value.startswith(f"{CAPABILITY_MULTIMODAL_PREFIX}:"):
            modality = value.split(":", 1)[1]
            if modality not in _SUPPORTED_INPUT_MODALITIES - {"text"}:
                raise ValueError(f"Unknown multimodal capability: {value!r}")
            return OrcaRouterCapability.CHAT
        return OrcaRouterCapability(value)

    @staticmethod
    def required_modality(value: str) -> str | None:
        """Returns the non-text modality required by a ``multimodal:<modality>`` key."""
        if value.startswith(f"{CAPABILITY_MULTIMODAL_PREFIX}:"):
            return value.split(":", 1)[1]
        return None

    def catalog_query(self) -> str:
        """Returns the ``?capability=`` value this capability asks the catalog for."""
        return self.value

    def all_keys(self) -> list[str]:
        """Returns every selector key this capability offers to a model selector."""
        if self is OrcaRouterCapability.CHAT:
            return [
                OrcaRouterCapability.CHAT.value,
                "multimodal:image",
                "multimodal:audio",
                "multimodal:video",
            ]
        return [self.value]


@dataclass
class ModelInfo:
    """One OrcaRouter catalog record, normalized and bounded."""

    id: str
    name: str | None = None
    context_length: int | None = None
    max_completion_tokens: int | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    supported_endpoint_types: tuple[str, ...] = ()
    supports_reasoning: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    owned_by: str | None = None
    description: str | None = None

    @property
    def supports_chat(self) -> bool:
        """True when at least one advertised endpoint type can serve chat."""
        return bool(CHAT_ENDPOINT_TYPES.intersection(self.supported_endpoint_types))

    def supports_input(self, modality: str) -> bool:
        """True when the catalog explicitly declares ``modality`` as an input.

        Undeclared capabilities fail closed: a model with no ``architecture``
        block is not admitted to a multimodal selector.
        """
        return modality in self.input_modalities

    def supports_any_input(self, modalities: tuple[str, ...]) -> bool:
        """True when every modality in ``modalities`` is explicitly declared."""
        return all(self.supports_input(modality) for modality in modalities)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable projection for the CLI."""
        return {
            "id": self.id,
            "name": self.name,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "input_modalities": list(self.input_modalities),
            "output_modalities": list(self.output_modalities),
            "supported_endpoint_types": list(self.supported_endpoint_types),
            "supports_reasoning": self.supports_reasoning,
            "reasoning_efforts": list(self.reasoning_efforts),
        }


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def parse_model(raw: Any) -> ModelInfo | None:
    """Normalizes one catalog record, or returns None when it is unusable.

    Records without a string ``id``, or carrying a capability the client cannot
    speak, are dropped rather than surfaced.
    """
    if not isinstance(raw, dict):
        return None
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None

    architecture = raw.get("architecture")
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    if isinstance(architecture, dict):
        input_modalities = _coerce_str_tuple(architecture.get("input_modalities"))
        output_modalities = _coerce_str_tuple(architecture.get("output_modalities"))

    reasoning = raw.get("reasoning")
    supports_reasoning = False
    reasoning_efforts: tuple[str, ...] = ()
    if isinstance(reasoning, dict):
        supports_reasoning = True
        reasoning_efforts = _coerce_str_tuple(reasoning.get("supported_efforts")) or (
            _coerce_str_tuple(reasoning.get("efforts"))
        )
    elif raw.get("supports_reasoning") is True:
        supports_reasoning = True

    name = raw.get("name")
    owned_by = raw.get("owned_by")
    description = raw.get("description")

    return ModelInfo(
        id=model_id.strip(),
        name=name if isinstance(name, str) else None,
        context_length=_coerce_int(raw.get("context_length")),
        max_completion_tokens=_coerce_int(raw.get("max_completion_tokens")),
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        supported_endpoint_types=_coerce_str_tuple(raw.get("supported_endpoint_types")),
        supports_reasoning=supports_reasoning,
        reasoning_efforts=reasoning_efforts,
        owned_by=owned_by if isinstance(owned_by, str) else None,
        description=description if isinstance(description, str) else None,
    )


def filter_models(
    models: list[ModelInfo],
    capability: OrcaRouterCapability | str | None = OrcaRouterCapability.CHAT,
    *,
    required_input_modalities: tuple[str, ...] = (),
) -> list[ModelInfo]:
    """Filters catalog records down to those that can serve ``capability``.

    Args:
        models: Normalized catalog records.
        capability: The capability the calling entry point needs. ``None``
            returns the unfiltered catalog.
        required_input_modalities: Non-text modalities the caller actually sends.

    Returns:
        list[ModelInfo]: Sorted, filtered records. Capabilities are read from
        catalog metadata only; a model that does not declare a required modality
        is excluded (fail closed).
    """
    if capability is None:
        return sorted(models, key=lambda m: m.id)

    key = (
        capability.value
        if isinstance(capability, OrcaRouterCapability)
        else str(capability)
    )
    base = OrcaRouterCapability.parse(key)
    modality = OrcaRouterCapability.required_modality(key)
    if modality:
        required_input_modalities = (modality,)

    selected: list[ModelInfo] = []
    for model in models:
        endpoints = set(model.supported_endpoint_types)
        if base is OrcaRouterCapability.CHAT:
            if not model.supports_chat:
                continue
            # Chat selectors must not offer a model that is only an image,
            # video, or rerank endpoint.
            if endpoints and endpoints <= (
                IMAGE_ENDPOINT_TYPES | VIDEO_ENDPOINT_TYPES | RERANK_ENDPOINT_TYPES
            ):
                continue
        elif base is OrcaRouterCapability.EMBEDDING:
            if not (endpoints & EMBEDDING_ENDPOINT_TYPES):
                continue
        elif base is OrcaRouterCapability.IMAGE:
            if not (endpoints & IMAGE_ENDPOINT_TYPES):
                continue
        elif base is OrcaRouterCapability.VIDEO:
            if not (endpoints & VIDEO_ENDPOINT_TYPES):
                continue
        elif base is OrcaRouterCapability.RERANK:
            if not (endpoints & RERANK_ENDPOINT_TYPES):
                continue

        if required_input_modalities and not model.supports_any_input(
            required_input_modalities
        ):
            continue
        selected.append(model)

    return sorted(selected, key=lambda m: m.id)


_VERIFIED_SEED: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="openai/gpt-5.5",
        name="OpenAI: GPT-5.5",
        context_length=400000,
        max_completion_tokens=128000,
        input_modalities=("text", "image"),
        output_modalities=("text",),
        supported_endpoint_types=("openai", "openai-response", "anthropic"),
        supports_reasoning=True,
        reasoning_efforts=("low", "medium", "high", "xhigh"),
    ),
    ModelInfo(
        id="anthropic/claude-opus-4.8",
        name="Anthropic: Claude Opus 4.8",
        context_length=200000,
        max_completion_tokens=64000,
        input_modalities=("text", "image"),
        output_modalities=("text",),
        supported_endpoint_types=("openai", "anthropic"),
        supports_reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    ModelInfo(
        id="google/gemini-3.5-flash",
        name="Google: Gemini 3.5 Flash",
        context_length=1000000,
        max_completion_tokens=65536,
        input_modalities=("text", "image", "audio", "video"),
        output_modalities=("text",),
        supported_endpoint_types=("openai", "openai-response", "gemini"),
        supports_reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    ModelInfo(
        id="deepseek/deepseek-v4-pro",
        name="DeepSeek: DeepSeek V4 Pro",
        context_length=1048576,
        max_completion_tokens=384000,
        input_modalities=("text",),
        output_modalities=("text",),
        supported_endpoint_types=("openai", "openai-response"),
        supports_reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    ModelInfo(
        id="orcarouter/auto",
        name="OrcaRouter: Auto",
        context_length=None,
        max_completion_tokens=None,
        input_modalities=("text",),
        output_modalities=("text",),
        supported_endpoint_types=("openai", "openai-response", "anthropic", "gemini"),
        supports_reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
)


def verified_seed_models() -> list[ModelInfo]:
    """Returns the small, verified cold-start catalog used during an outage.

    These entries carry verified context windows, input modalities, and
    reasoning-effort ladders. Live discovery, when it succeeds, replaces them
    entirely -- seed entries are never merged into a live result.
    """
    return [ModelInfo(**vars(model)) for model in _VERIFIED_SEED]


@dataclass
class CatalogResult:
    """The outcome of a catalog lookup, including its provenance."""

    models: list[ModelInfo] = field(default_factory=list)
    source: str = "live"
    """``"live"`` or ``"fallback"``."""

    degraded: bool = False
    """True when the live catalog could not be reached and the seed was used."""

    detail: str | None = None
    """A short, user-facing explanation of a degraded result. Never holds a key."""

    @property
    def catalog_source(self) -> str:
        """Returns a stable identifier of where the models came from."""
        return "https://api.orcarouter.ai/v1/models" if not self.degraded else "seed"

    def validate_selection(self, model_name: str | None) -> bool:
        """Returns whether ``model_name`` is still present in this catalog."""
        if not model_name:
            return False
        return any(model.id == model_name for model in self.models)


def normalize_capability(
    capability: OrcaRouterCapability | str | None,
) -> OrcaRouterCapability | None:
    """Coerces a capability key (including ``multimodal:x``) to the enum."""
    if capability is None or isinstance(capability, OrcaRouterCapability):
        return capability
    return OrcaRouterCapability.parse(str(capability))


def models_endpoint_url(
    api_base_url: str, capability: OrcaRouterCapability | str | None
) -> str:
    """Builds the catalog URL for ``capability`` on the inference origin.

    ``capability=None`` asks for the unfiltered catalog.
    """
    base = f"{api_v1_base_url(api_base_url)}/models"
    normalized = normalize_capability(capability)
    if normalized is None:
        return base
    return f"{base}?capability={normalized.catalog_query()}"


async def fetch_models(
    api_base_url: str,
    api_key: str,
    *,
    capability: OrcaRouterCapability | None = OrcaRouterCapability.CHAT,
    timeout_seconds: float = CATALOG_TIMEOUT_SECONDS,
    max_bytes: int = CATALOG_MAX_BYTES,
    max_items: int = CATALOG_MAX_ITEMS,
    session: aiohttp.ClientSession | None = None,
) -> list[ModelInfo]:
    """Fetches and normalizes the OrcaRouter catalog.

    The request is bounded in time, bytes, and item count so a hostile or broken
    catalog response cannot consume unbounded memory or advertise routes this
    client cannot speak.

    Args:
        api_base_url: The inference origin (``https://api.orcarouter.ai``).
        api_key: A plain OrcaRouter API key. Sent as a Bearer token.
        capability: Which capability to ask the provider to filter on. ``None``
            returns the unfiltered catalog.
        timeout_seconds: Per-request timeout.
        max_bytes: Maximum response body size to buffer.
        max_items: Maximum number of records to consider.
        session: Optional caller-owned aiohttp session.

    Returns:
        list[ModelInfo]: Normalized, filtered catalog records.

    Raises:
        OrcaRouterAuthError: If OrcaRouter rejects the credential.
        OrcaRouterCatalogError: On any other failure.
    """
    url = models_endpoint_url(api_base_url, capability)
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _fetch(active_session: aiohttp.ClientSession) -> list[ModelInfo]:
        async with active_session.get(
            url, headers=headers, timeout=timeout
        ) as response:
            if response.status in (401, 403):
                raise OrcaRouterAuthError(
                    "OrcaRouter rejected the credential while listing models "
                    f"(HTTP {response.status}). Re-run the OrcaRouter login."
                )
            if response.status != 200:
                raise OrcaRouterCatalogError(
                    f"OrcaRouter model catalog request failed with HTTP "
                    f"{response.status}."
                )
            if response.content_length is not None and (
                response.content_length > max_bytes
            ):
                raise OrcaRouterCatalogError(
                    "OrcaRouter model catalog response exceeded the size limit."
                )
            # Read in bounded chunks: a single read() returns whatever arrived
            # first, not the whole body, and the accumulated size is what must
            # stay bounded.
            chunks: list[bytes] = []
            received = 0
            async for chunk in response.content.iter_chunked(_CATALOG_CHUNK_BYTES):
                received += len(chunk)
                if received > max_bytes:
                    raise OrcaRouterCatalogError(
                        "OrcaRouter model catalog response exceeded the size limit."
                    )
                chunks.append(chunk)
            body = b"".join(chunks)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise OrcaRouterCatalogError(
                "OrcaRouter model catalog returned a body that is not JSON."
            ) from e

        records = payload.get("data") if isinstance(payload, dict) else payload
        if records is None:
            records = []
        if not isinstance(records, list):
            raise OrcaRouterCatalogError(
                "OrcaRouter model catalog returned an unexpected shape."
            )

        models = [
            model
            for model in (parse_model(raw) for raw in records[:max_items])
            if model is not None
        ]
        return filter_models(models, capability)

    if session is not None:
        return await _fetch(session)

    async with aiohttp.ClientSession() as own_session:
        return await _fetch(own_session)


def _run_coroutine(coroutine: Any) -> Any:
    """Runs ``coroutine`` to completion, even from inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def discover_models(
    api_base_url: str,
    api_key: str | None,
    *,
    capability: OrcaRouterCapability | str | None = OrcaRouterCapability.CHAT,
    required_input_modalities: tuple[str, ...] = (),
    allow_fallback: bool = True,
    timeout_seconds: float = CATALOG_TIMEOUT_SECONDS,
) -> CatalogResult:
    """Resolves the catalog for ``capability``, falling back to the verified seed.

    A successful live discovery is authoritative and is returned as-is. When it
    fails, the verified seed (filtered by the same capability rules) is returned
    and clearly marked degraded; the caller must not present it as live.

    Args:
        api_base_url: The inference origin.
        api_key: A plain OrcaRouter API key, or None when unauthenticated.
        capability: Which capability the caller needs. A ``multimodal:<modality>``
            key both narrows the wire query to chat and requires the modality to
            be declared in the catalog.
        required_input_modalities: Non-text modalities the caller actually sends.
        allow_fallback: Whether to fall back to the seed on failure.
        timeout_seconds: Per-request timeout.

    Returns:
        CatalogResult: The models plus provenance.
    """
    raw_capability = capability
    capability = normalize_capability(raw_capability)
    # A "multimodal:<modality>" key both narrows the wire query to chat and
    # requires the modality to be declared in architecture.input_modalities.
    if required_input_modalities:
        modality_requirement = required_input_modalities
    else:
        modality = (
            OrcaRouterCapability.required_modality(raw_capability)
            if isinstance(raw_capability, str)
            else None
        )
        modality_requirement = (modality,) if modality else ()
    if not api_key:
        if not allow_fallback:
            raise OrcaRouterCatalogError(
                "No OrcaRouter API key is available, so the live model catalog "
                "cannot be read."
            )
        return CatalogResult(
            models=filter_models(
                verified_seed_models(),
                capability,
                required_input_modalities=modality_requirement,
            ),
            source="fallback",
            degraded=True,
            detail="no OrcaRouter API key is configured",
        )

    try:
        models = _run_coroutine(
            fetch_models(
                api_base_url,
                api_key,
                capability=capability,
                timeout_seconds=timeout_seconds,
            )
        )
    except OrcaRouterAuthError:
        # A rejected credential is terminal, not an outage: propagate so the
        # caller can mark the exact generation as needing reauthentication.
        raise
    except (
        OrcaRouterCatalogError,
        aiohttp.ClientError,
        asyncio.TimeoutError,
        OSError,
    ) as e:
        if not allow_fallback:
            raise
        logger.warning(
            f"OrcaRouter model catalog unavailable ({type(e).__name__}); using the "
            "verified fallback catalog."
        )
        return CatalogResult(
            models=filter_models(
                verified_seed_models(),
                capability,
                required_input_modalities=modality_requirement,
            ),
            source="fallback",
            degraded=True,
            detail=f"live catalog request failed ({type(e).__name__})",
        )

    return CatalogResult(
        models=filter_models(
            models,
            capability,
            required_input_modalities=modality_requirement,
        ),
        source="live",
        degraded=False,
    )


def describe_credential(api_key: str | None) -> str:
    """Returns a log-safe description of the credential backing a discovery call."""
    return mask_api_key(api_key)


@dataclass
class ModelSelector:
    """The option list a model control offers, plus the surviving selection.

    ``options`` is the capability-filtered list that must be handed to the model
    selector. ``selected`` is ``current_model`` when it is still present in that
    list, and None otherwise -- a stale or newly incompatible value is cleared
    rather than silently kept.
    """

    options: list[ModelInfo]
    selected: str | None
    catalog: CatalogResult

    @property
    def degraded(self) -> bool:
        """True when the options came from the verified fallback catalog."""
        return self.catalog.degraded

    @property
    def source(self) -> str:
        """Returns ``"live"`` or ``"fallback"``."""
        return self.catalog.source


def resolve_model_selector(
    api_base_url: str,
    api_key: str | None,
    *,
    capability: OrcaRouterCapability | str = OrcaRouterCapability.CHAT,
    current_model: str | None = None,
    required_input_modalities: tuple[str, ...] = (),
    allow_fallback: bool = True,
) -> ModelSelector:
    """Builds the filtered model selector for one capability.

    Callers re-run this whenever the provider, the attachment type, or the task
    capability changes, so the options always match the request being assembled.

    Args:
        api_base_url: The OrcaRouter inference origin.
        api_key: A plain OrcaRouter API key, or None.
        capability: The capability the target entry point needs.
        current_model: The model currently selected, if any.
        required_input_modalities: Non-text modalities the caller will send.
        allow_fallback: Whether the verified seed may stand in during an outage.

    Returns:
        ModelSelector: Filtered options plus the validated selection.
    """
    catalog = discover_models(
        api_base_url,
        api_key,
        capability=capability,
        required_input_modalities=required_input_modalities,
        allow_fallback=allow_fallback,
    )
    selected = current_model if catalog.validate_selection(current_model) else None
    return ModelSelector(options=catalog.models, selected=selected, catalog=catalog)
