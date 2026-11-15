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

"""Adapt Oumi tool environments to TRL's ``environment_factory`` protocol.

TRL drives multi-turn tool use itself: it calls ``reset()`` per rollout, then
invokes one method per tool and feeds the return value back as an observation.
Oumi environments expose the opposite shape -- a single ``step(calls)`` entry
point that takes a batch of calls. This module bridges the two, so the rollout
loop lives in TRL instead of in Oumi.
"""

import inspect
import json
from collections.abc import Callable
from typing import Any

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.tool_call import ToolResult


class OumiEnvironmentAdapter:
    """One TRL environment instance backed by a per-rollout ``ToolRouter``.

    TRL requires an environment object whose public methods are the tools and
    whose ``reset()`` clears per-rollout state. Oumi tools are routed through
    ``ToolRouter``, so each tool id becomes a method that dispatches a single
    call. Instantiate through :func:`build_environment_factory`, which gives
    TRL a factory rather than one shared instance.

    Tools are installed on a per-config subclass rather than on the instance:
    TRL discovers them with ``inspect.ismethod``, which only matches bound
    methods, and a plain function attached to an instance would be silently
    ignored. The methods dispatch through ``self._router``, so every rollout
    still reaches its own environment state.

    TRL builds the environments once, in ``__init__``, and reuses each instance
    across generation steps, calling ``reset()`` before every rollout. That is
    why this class holds two routers: ``_template`` stays untouched as the
    blueprint, and ``_router`` is a fresh per-rollout clone that ``reset()``
    replaces. Cloning is not optional -- ``ToolRouter.from_environment_config``
    closes every env that ``requires_isolation()`` (databases, stateful
    simulators), because it expects callers to rebuild them through
    ``for_sample()``. Dispatching on the template would hit a closed resource.
    """

    def __init__(self, router: ToolRouter) -> None:
        """Wrap one rollout's router; build instances via the factory."""
        self._template = router
        self._router = router.for_sample()

    def reset(self, **kwargs: Any) -> None:
        """Start a new rollout with freshly built environment state.

        TRL passes the dataset row's columns as keyword arguments, so this must
        tolerate arbitrary keys. State is reset by rebuilding the router rather
        than by clearing fields: envs that require isolation are rebuilt from
        their config, and the previous rollout's envs are closed so their
        sessions do not outlive the rollout.
        """
        previous = self._router
        self._router = self._template.for_sample()
        previous.close()


def _render_result(result: ToolResult) -> str:
    """Flatten a ``ToolResult`` into the string TRL appends as an observation."""
    output = result.output
    return output if isinstance(output, str) else json.dumps(output)


def _make_tool(tool: Any) -> Callable[..., str]:
    """Build the method TRL exposes to the model for one Oumi tool.

    TRL renders the tool schema from the callable: ``get_type_hints`` supplies
    the argument types, ``inspect.signature`` supplies which are required, and
    a Google-style docstring supplies the descriptions. ``__signature__`` alone
    is not enough because the types come from ``__annotations__``, so both are
    set here. The tool's own JSON schema stays authoritative for validation --
    ``ToolRouter.route_batch`` re-validates every call.

    The method dispatches through ``self._router`` rather than closing over one
    router, because it is installed on the class and shared by every instance.
    """
    tool_id = tool.id
    parameters = tool.parameters or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])

    def call_tool(self: OumiEnvironmentAdapter, **kwargs: Any) -> str:
        results = self._router.route_batch([(tool_id, kwargs)])
        return _render_result(results[0])

    call_tool.__name__ = tool_id
    call_tool.__annotations__ = {name: str for name in properties}
    call_tool.__annotations__["return"] = str
    call_tool.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *[
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=inspect.Parameter.empty if name in required else None,
                    annotation=str,
                )
                for name in properties
            ],
        ],
        return_annotation=str,
    )
    call_tool.__doc__ = _google_docstring(tool.description, properties)
    return call_tool


def _google_docstring(description: str | None, properties: dict[str, Any]) -> str:
    """Render a tool's description and arguments as a Google-style docstring.

    TRL refuses to build a schema when an annotated argument has no docstring
    entry, so every property needs a line even if the schema omits a
    description for it.
    """
    lines = [description or "Call the tool.", ""]
    if properties:
        lines.append("Args:")
        for name, schema in properties.items():
            detail = (schema or {}).get("description") or f"The {name} argument."
            lines.append(f"    {name}: {detail}")
    return "\n".join(lines)


def _build_adapter(router: ToolRouter) -> OumiEnvironmentAdapter:
    """Build an adapter whose class exposes this router's tools as methods."""
    namespace: dict[str, Any] = {
        spec.function.name: _make_tool(router.tools_by_id[spec.function.name])
        for spec in router.tool_specs
    }
    adapter_cls = type("OumiEnvironmentAdapter", (OumiEnvironmentAdapter,), namespace)
    return adapter_cls(router)


def build_environment_factory(config_path: str) -> Callable[[], OumiEnvironmentAdapter]:
    """Build the ``environment_factory`` TRL expects from an Oumi env config.

    TRL calls the returned factory once per generation-batch slot and reuses
    each adapter across steps, so an adapter keeps its own template router and
    rebuilds rollout state from it in ``reset()``.
    """

    def factory() -> OumiEnvironmentAdapter:
        env_config = EnvironmentConfig.from_yaml(config_path)
        env_config.finalize_and_validate()
        return _build_adapter(ToolRouter.from_environment_config(env_config))

    return factory
