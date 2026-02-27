#!/usr/bin/env python3
"""Fetch Fireworks API specs from docs and merge into a single OpenAPI 3.1.0 document.

Each Fireworks API reference page at docs.fireworks.ai/api-reference/{name}.md
embeds a self-contained OpenAPI 3.1.0 YAML spec in a fenced code block.
This script downloads the pages relevant to the oumi deploy client, extracts
the YAML fragments, and merges paths + schemas into one unified spec.

Usage:
    python scripts/fireworks/fetch_openapi.py [--output fireworks.openapi.yaml]

The output file can then drive:
  - Request/response validation (openapi-core, jsonschema)
  - Mock server generation (prism, connexion)
  - Test fixture generation
"""

import argparse
import re
import sys
from collections import OrderedDict

import requests
import yaml

DOCS_BASE = "https://docs.fireworks.ai/api-reference"

# Endpoint pages used by FireworksDeploymentClient (src/oumi/deploy/fireworks_client.py)
ENDPOINT_PAGES = [
    # Models
    "create-model",
    "get-model",
    "delete-model",
    "list-models",
    "get-model-upload-endpoint",
    "validate-model-upload",
    "prepare-model",
    "get-model-download-endpoint",
    "update-model",
    # Deployments
    "create-deployment",
    "get-deployment",
    "delete-deployment",
    "list-deployments",
    "scale-deployment",
    "update-deployment",
    "undelete-deployment",
    # Deployed models (LoRA)
    "create-deployed-model",
    "get-deployed-model",
    "delete-deployed-model",
    "list-deployed-models",
    "update-deployed-model",
]

YAML_BLOCK_RE = re.compile(
    r"````yaml\s+.*?\n(.*?)````",
    re.DOTALL,
)


def fetch_page(page_name: str) -> str:
    """Fetch markdown content for a single API reference page from docs.fireworks.ai."""
    url = f"{DOCS_BASE}/{page_name}.md"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_openapi_yaml(markdown: str) -> dict | None:
    """Extract the OpenAPI YAML block from a Fireworks docs markdown page."""
    match = YAML_BLOCK_RE.search(markdown)
    if not match:
        return None
    return yaml.safe_load(match.group(1))


def merge_specs(fragments: list[dict]) -> dict:
    """Merge multiple per-endpoint OpenAPI fragments into a single spec."""
    merged: dict = {
        "openapi": "3.1.0",
        "info": {
            "title": "Fireworks Gateway REST API (merged)",
            "version": "4.27.5",
            "description": (
                "Merged OpenAPI spec covering the Fireworks REST API endpoints "
                "used by the oumi deploy client.  Auto-generated from "
                "docs.fireworks.ai/api-reference pages."
            ),
        },
        "servers": [{"url": "https://api.fireworks.ai"}],
        "security": [{"BearerAuth": []}],
        "tags": [{"name": "Gateway"}],
        "paths": {},
        "components": {"schemas": {}, "securitySchemes": {}},
    }

    for frag in fragments:
        # Merge paths
        for path, methods in frag.get("paths", {}).items():
            if path not in merged["paths"]:
                merged["paths"][path] = {}
            merged["paths"][path].update(methods)

        # Merge component schemas (last-write wins; they are identical across pages)
        for schema_name, schema_def in (
            frag.get("components", {}).get("schemas", {}).items()
        ):
            merged["components"]["schemas"][schema_name] = schema_def

        # Merge security schemes
        for scheme_name, scheme_def in (
            frag.get("components", {}).get("securitySchemes", {}).items()
        ):
            merged["components"]["securitySchemes"][scheme_name] = scheme_def

    # Sort paths for readability
    merged["paths"] = OrderedDict(sorted(merged["paths"].items()))

    return merged


class FlowStyleDumper(yaml.SafeDumper):
    """YAML dumper that keeps enum lists compact."""

    pass


def str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Represent strings with literal block style when they contain newlines."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


FlowStyleDumper.add_representer(str, str_representer)
FlowStyleDumper.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)


def main() -> None:
    """Parse args, fetch docs pages, merge OpenAPI fragments, write output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="fireworks.openapi.yaml",
        help="Output file path (default: fireworks.openapi.yaml)",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        default=None,
        help="Specific page slugs to fetch (default: all deploy-related endpoints)",
    )
    args = parser.parse_args()

    pages = args.pages or ENDPOINT_PAGES
    fragments: list[dict] = []
    errors: list[str] = []

    for page in pages:
        print(f"  Fetching {DOCS_BASE}/{page}.md ... ", end="", flush=True)
        try:
            markdown = fetch_page(page)
            spec = extract_openapi_yaml(markdown)
            if spec:
                fragments.append(spec)
                n_paths = len(spec.get("paths", {}))
                n_schemas = len(spec.get("components", {}).get("schemas", {}))
                print(f"OK ({n_paths} path(s), {n_schemas} schema(s))")
            else:
                print("WARN: no OpenAPI YAML block found")
                errors.append(f"{page}: no YAML block")
        except requests.HTTPError as exc:
            print(f"ERROR: {exc}")
            errors.append(f"{page}: {exc}")

    if not fragments:
        print("\nNo specs extracted. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nMerging {len(fragments)} fragments...")
    merged = merge_specs(fragments)

    n_paths = sum(len(methods) for methods in merged["paths"].values())
    n_schemas = len(merged["components"]["schemas"])
    print(
        f"  Result: {n_paths} operations across {len(merged['paths'])} paths, "
        f"{n_schemas} schemas"
    )

    with open(args.output, "w") as f:
        yaml.dump(
            merged,
            f,
            Dumper=FlowStyleDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )
    print(f"\nWritten to {args.output}")

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
