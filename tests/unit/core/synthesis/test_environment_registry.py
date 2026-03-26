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

"""Tests for EnvironmentRegistry."""

from typing import Any

from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.environment_registry import (
    _pluralize,
    build_dependency_graph,
    derive_schema_from_tools,
    sort_into_waves,
)


class TestPluralize:
    def test_regular_noun(self):
        assert _pluralize("tenant") == "tenants"

    def test_already_plural(self):
        assert _pluralize("tenants") == "tenants"

    def test_noun_ending_in_s(self):
        assert _pluralize("status") == "statuses"

    def test_noun_ending_in_y(self):
        assert _pluralize("category") == "categories"

    def test_noun_ending_in_e(self):
        assert _pluralize("file") == "files"


def _make_tool(**overrides: Any) -> ToolAttribute:
    defaults: dict[str, Any] = dict(
        id="get_tenant",
        name="GetTenant",
        description="Look up a tenant",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
        parameters={
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
        },
    )
    defaults.update(overrides)
    return ToolAttribute(**defaults)


class TestDeriveSchemaFromTools:
    def test_single_entity_from_output_schema(self):
        """A tool returning tenant fields creates a tenants collection."""
        tool = _make_tool(
            output_schema={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            }
        )
        schema = derive_schema_from_tools([tool])
        assert schema["type"] == "object"
        assert "tenants" in schema["properties"]
        tenants_schema = schema["properties"]["tenants"]
        assert tenants_schema["type"] == "object"
        record_props = tenants_schema["additionalProperties"]["properties"]
        assert "name" in record_props
        assert "email" in record_props

    def test_multiple_entities_from_multiple_tools(self):
        """Tools referencing different _id fields create separate collections."""
        tenant_tool = _make_tool(
            id="get_tenant",
            name="GetTenant",
            parameters={
                "type": "object",
                "properties": {"tenant_id": {"type": "string"}},
                "required": ["tenant_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            },
        )
        unit_tool = _make_tool(
            id="get_unit",
            name="GetUnit",
            parameters={
                "type": "object",
                "properties": {"unit_id": {"type": "string"}},
                "required": ["unit_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "unit_number": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        )
        schema = derive_schema_from_tools([tenant_tool, unit_tool])
        assert "tenants" in schema["properties"]
        assert "units" in schema["properties"]

    def test_merges_fields_across_tools(self):
        """Two tools referencing the same entity merge their fields."""
        get_tool = _make_tool(
            id="get_tenant",
            name="GetTenant",
            output_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )
        create_tool = _make_tool(
            id="create_tenant",
            name="CreateTenant",
            parameters={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                },
                "required": ["tenant_id", "name"],
            },
            output_schema=None,
        )
        schema = derive_schema_from_tools([get_tool, create_tool])
        record_props = (
            schema["properties"]["tenants"]["additionalProperties"]["properties"]
        )
        assert "name" in record_props
        assert "email" in record_props
        assert "phone" in record_props

    def test_preserves_enum_constraints(self):
        """Enum values from parameters are preserved in the schema."""
        tool = _make_tool(
            id="update_unit",
            name="UpdateUnit",
            parameters={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["occupied", "vacant"],
                    },
                },
                "required": ["unit_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        )
        schema = derive_schema_from_tools([tool])
        record_props = (
            schema["properties"]["units"]["additionalProperties"]["properties"]
        )
        assert record_props["status"].get("enum") == ["occupied", "vacant"]

    def test_empty_tools_returns_empty_schema(self):
        """No tools produces an empty schema (triggers LLM fallback)."""
        schema = derive_schema_from_tools([])
        assert schema["type"] == "object"
        assert len(schema.get("properties", {})) == 0

    def test_id_fields_excluded_from_record_properties(self):
        """The entity's own _id field is not duplicated inside the record."""
        tool = _make_tool(
            output_schema={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        )
        schema = derive_schema_from_tools([tool])
        record_props = (
            schema["properties"]["tenants"]["additionalProperties"]["properties"]
        )
        assert "tenant_id" not in record_props
        assert "name" in record_props


class TestBuildDependencyGraph:
    def test_no_foreign_keys(self):
        """Collections with no _id references have empty dependency sets."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "units": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"number": {"type": "string"}},
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph == {"tenants": set(), "units": set()}

    def test_foreign_key_creates_dependency(self):
        """A lease with tenant_id depends on tenants."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "leases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "start_date": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["leases"] == {"tenants"}
        assert graph["tenants"] == set()

    def test_multiple_foreign_keys(self):
        """A lease referencing tenant_id and unit_id depends on both."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "units": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"number": {"type": "string"}},
                    },
                },
                "leases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "unit_id": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["leases"] == {"tenants", "units"}

    def test_self_referencing_id_ignored(self):
        """A tenant_id inside the tenants collection is not a dependency."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["tenants"] == set()


class TestSortIntoWaves:
    def test_independent_collections_in_one_wave(self):
        graph = {"tenants": set(), "units": set()}
        waves = sort_into_waves(graph)
        assert len(waves) == 1
        assert set(waves[0]) == {"tenants", "units"}

    def test_dependent_collection_in_later_wave(self):
        graph = {
            "tenants": set(),
            "units": set(),
            "leases": {"tenants", "units"},
        }
        waves = sort_into_waves(graph)
        assert len(waves) == 2
        assert set(waves[0]) == {"tenants", "units"}
        assert waves[1] == ["leases"]

    def test_three_wave_chain(self):
        graph = {
            "tenants": set(),
            "units": set(),
            "leases": {"tenants", "units"},
            "payments": {"leases"},
        }
        waves = sort_into_waves(graph)
        assert len(waves) == 3
        assert set(waves[0]) == {"tenants", "units"}
        assert waves[1] == ["leases"]
        assert waves[2] == ["payments"]

    def test_empty_graph(self):
        waves = sort_into_waves({})
        assert waves == []

    def test_cycle_broken(self):
        """Cycles are broken — all collections still appear in output."""
        graph = {
            "tenants": {"leases"},
            "leases": {"tenants"},
        }
        waves = sort_into_waves(graph)
        all_collections = [c for wave in waves for c in wave]
        assert set(all_collections) == {"tenants", "leases"}
