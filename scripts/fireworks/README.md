# Fireworks API Tooling

Scripts for extracting, maintaining, and using the Fireworks REST API OpenAPI specification.

## Background

Fireworks does not publish a standalone OpenAPI spec, but each API reference page at
`https://docs.fireworks.ai/api-reference/{endpoint-name}.md` embeds a **self-contained
OpenAPI 3.1.0 YAML fragment** (protobuf-generated from their gateway service).
The `fetch_openapi.py` script scrapes these pages and merges the fragments into a
single unified spec.

## Files

| File | Purpose |
|------|---------|
| `fetch_openapi.py` | Scraper that fetches Fireworks doc pages and merges OpenAPI fragments |
| `../../fireworks.openapi.yaml` | Generated spec (project root) — **do not edit manually** |

---

## Generating / Refreshing the OpenAPI Spec

```bash
# From project root:
python scripts/fireworks/fetch_openapi.py

# Custom output path:
python scripts/fireworks/fetch_openapi.py --output path/to/spec.yaml

# Fetch only specific endpoints:
python scripts/fireworks/fetch_openapi.py --pages create-model get-model create-deployment
```

The script fetches all endpoints used by `FireworksDeploymentClient`:

- **Models**: create, get, delete, list, upload-endpoint, validate-upload, prepare, download-endpoint, update
- **Deployments**: create, get, delete, list, scale, update, undelete
- **Deployed Models (LoRA)**: create (load), get, delete (unload), list, update

### When to re-run

- After a Fireworks API version bump (check `info.version` in the generated spec)
- When adding new Fireworks endpoints to the deploy client
- Periodically (e.g., monthly) to catch schema drift

### Dependencies

Only `requests` and `pyyaml` — both already in the project's dependency tree.

---

## Using the Spec for Testing

### 1. Request/Response Validation with `jsonschema`

Validate that `FireworksDeploymentClient` sends payloads conforming to the spec:

```python
import yaml
import jsonschema

with open("fireworks.openapi.yaml") as f:
    spec = yaml.safe_load(f)

schemas = spec["components"]["schemas"]

# Validate a deployment creation payload
payload = {
    "baseModel": "accounts/fireworks/models/qwen3-4b",
    "acceleratorType": "NVIDIA_H100_80GB",
    "minReplicaCount": 0,
    "maxReplicaCount": 1,
}
jsonschema.validate(payload, schemas["gatewayDeployment"])
```

### 2. Schema-Validating HTTP Transport

Intercept outgoing `httpx` requests and validate them against the spec
before they leave the process. See TestPlan.md section 24.4 for the
full `SchemaValidatingTransport` implementation.

```python
# In tests:
transport = SchemaValidatingTransport(responses={...})
client._client = httpx.AsyncClient(base_url="https://api.fireworks.ai", transport=transport)
await client.create_endpoint(...)  # Raises jsonschema.ValidationError on schema mismatch
```

### 3. Mock Server with Prism (Stoplight)

Spin up a local mock server that returns spec-conforming responses:

```bash
npm install -g @stoplight/prism-cli
prism mock fireworks.openapi.yaml --port 4010

# Point tests at the mock:
export FIREWORKS_API_BASE_URL=http://localhost:4010
pytest tests/unit/deploy/
```

### 4. Enum Drift Detection (no HTTP)

Catch cases where the local accelerator mapping falls behind the API:

```python
def test_accelerator_enum_coverage(openapi_spec):
    valid = openapi_spec["components"]["schemas"]["gatewayAcceleratorType"]["enum"]
    for oumi_name, fw_name in FIREWORKS_ACCELERATORS.items():
        assert fw_name in valid, f"{fw_name} not in spec enum"
```

### 5. Client Code Generation

Generate a typed Python client from the spec and compare it against
the hand-written `FireworksDeploymentClient`:

```bash
# Using openapi-python-client:
pip install openapi-python-client
openapi-python-client generate --path fireworks.openapi.yaml --output-path /tmp/fw-client

# Using datamodel-code-generator (Pydantic models only):
pip install datamodel-code-generator
datamodel-codegen --input fireworks.openapi.yaml --output /tmp/fw_models.py

# Using openapi-generator (multi-language):
docker run --rm -v $(pwd):/local openapitools/openapi-generator-cli generate \
    -i /local/fireworks.openapi.yaml \
    -g python \
    -o /local/generated-client
```

**How to use the generated client for validation:**

| Approach | What it catches | Effort |
|----------|----------------|--------|
| Generate Pydantic models, use them as fixtures | Response parsing mismatches, missing fields | Low |
| Generate full client, run side-by-side tests | URL construction, HTTP method, payload shape | Medium |
| Diff generated models vs hand-written dataclasses | Field name / type drift | Low |
| Import generated enums in tests | Enum value coverage | Minimal |

Example: generate Pydantic models and use them to validate responses in tests:

```python
# After running datamodel-codegen:
from fw_models import GatewayDeployment, GatewayModel

# In tests — validate that _parse_deployment produces valid data:
raw = {"name": "accounts/a/deployments/d", "state": "CREATING", "baseModel": "..."}
parsed = GatewayDeployment.model_validate(raw)  # Raises ValidationError on mismatch
```

### 6. Integration with `validateOnly=true`

For live API tests that should not create resources, pass `validateOnly=true`
as a query parameter to `CreateDeployment`. The server validates the full
payload and returns the would-be response without creating anything.

---

## Spec Coverage vs Client Methods

| Client Method | Spec Operation | operationId |
|--------------|---------------|-------------|
| `upload_model()` → `_create_model_resource()` | `POST /models` | `Gateway_CreateModel` |
| `upload_model()` → `_upload_model_files()` | `POST /models/{id}:getUploadEndpoint` | `Gateway_GetModelUploadEndpoint` |
| `upload_model()` → `_wait_and_validate()` | `GET /models/{id}:validateUpload` | `Gateway_ValidateModelUpload` |
| `prepare_model()` | `POST /models/{id}:prepare` | `Gateway_PrepareModel` |
| `get_model_status()` | `GET /models/{id}` | `Gateway_GetModel` |
| `list_models()` | `GET /models` | `Gateway_ListModels` |
| `delete_model()` | `DELETE /models/{id}` | `Gateway_DeleteModel` |
| `create_endpoint()` | `POST /deployments` | `Gateway_CreateDeployment` |
| `get_endpoint()` | `GET /deployments/{id}` | `Gateway_GetDeployment` |
| `update_endpoint()` | `PATCH /deployments/{id}:scale` | `Gateway_ScaleDeployment` |
| `delete_endpoint()` | `DELETE /deployments/{id}` | `Gateway_DeleteDeployment` |
| `list_endpoints()` | `GET /deployments` | `Gateway_ListDeployments` |

## Known Discrepancies

### Scale Deployment payload shape

The official spec defines `GatewayScaleDeploymentBody` as:

```yaml
properties:
  replicaCount:
    type: integer
```

But `FireworksDeploymentClient.update_endpoint()` sends:

```python
{"config": {"minReplicas": ..., "maxReplicas": ..., "acceleratorType": ...}}
```

This uses an undocumented `config` wrapper. Needs investigation: the client
may be calling the wrong endpoint (should use `PATCH /deployments/{id}`
i.e. `UpdateDeployment` instead of `:scale`) or the docs may be incomplete.
