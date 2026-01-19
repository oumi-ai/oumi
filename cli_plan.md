# Model Deployment Feature Gap Analysis: Together.ai & Fireworks.ai

## Executive Summary

This document identifies missing **model deployment-related features** in our current Together.ai and Fireworks.ai clients compared to their full API capabilities as of January 2026.

**Scope**: Model deployment lifecycle only

- Model upload, preparation, optimization
- Endpoint creation, configuration, lifecycle management
- Hardware selection and deployment monitoring
- Cost optimization for deployed endpoints

**Out of Scope**: Fine-tuning, batch inference, evaluation jobs, dataset management

**Current Implementation Status**: ✅ Core deployment workflow fully functional

- Model upload, endpoint creation, status monitoring, deletion
- Basic hardware discovery and model listing
- LoRA adapter support for both providers

**Key Gaps**:

1. **Endpoint Lifecycle** - No start/stop operations for cost savings
2. **Model Optimization** - Limited preparation/quantization options
3. **Hardware Discovery** - Fireworks uses hardcoded list
4. **Deployment Testing** - Basic CLI test not in client API

---

## Part 1: Current Implementation Coverage

### ✅ What We Have Implemented

#### Together.ai Coverage

| Feature | Status | Implementation |
|---------|--------|----------------|
| Model Upload | ✅ Full | `upload_model()` with HF/S3/GCS support |
| Job Status Tracking | ✅ Full | Custom `get_job_status()` method |
| Model Listing | ✅ Full | Upload jobs + fine-tune jobs + public catalog |
| Endpoint Create | ✅ Full | Hardware + autoscaling config |
| Endpoint Update | ✅ Full | Scale replicas and hardware |
| Endpoint Delete | ✅ Full | Clean deletion |
| Endpoint List | ✅ Full | Filter by user with `mine=true` |
| Endpoint Status | ✅ Full | Get detailed endpoint info |
| Hardware List | ✅ Full | Query `/hardware` endpoint |
| Model Delete | ❌ N/A | Provider doesn't support via API |

#### Fireworks.ai Coverage

| Feature | Status | Implementation |
|---------|--------|----------------|
| Model Upload | ✅ Full | Multi-step presigned URL flow |
| Model Preparation | ✅ Partial | `prepare_model()` exists but limited |
| Model Validation | ✅ Full | Automatic after upload |
| Model Listing | ✅ Full | User + public models with pagination |
| Model Delete | ✅ Full | Full path and short ID support |
| Deployment Create | ✅ Full | Hardware + autoscaling config |
| Deployment Scale | ✅ Full | Update hardware/replicas via `:scale` |
| Deployment Delete | ✅ Full | Clean deletion |
| Deployment List | ✅ Full | Paginated listing |
| Hardware List | ⚠️ Hardcoded | Static list of 4 GPU types |

---

## Part 2: Missing Deployment Features

### 🔴 Priority 1: Critical for Production

#### 1. Endpoint Start/Stop Operations (Together.ai)

**Current**: Can only create or delete endpoints (wastes cost when idle)
**Missing**:

- `PATCH /v1/endpoints/{id}` with `{"min_replicas": 0}` to stop
- Start command to resume from stopped state

**Impact**: Cannot pause endpoints to save costs without full deletion
**API Reference**: [Update Endpoint](https://docs.together.ai/reference/updateendpoint)

**Implementation Needed**:

```python
async def stop_endpoint(self, endpoint_id: str) -> Endpoint:
    """Stop an endpoint by scaling to 0 replicas."""
    return await self.update_endpoint(
        endpoint_id,
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=0)
    )

async def start_endpoint(self, endpoint_id: str, min_replicas: int = 1) -> Endpoint:
    """Start a stopped endpoint."""
    endpoint = await self.get_endpoint(endpoint_id)
    return await self.update_endpoint(
        endpoint_id,
        autoscaling=AutoscalingConfig(
            min_replicas=min_replicas,
            max_replicas=endpoint.autoscaling.max_replicas
        )
    )
```

**CLI Impact**:

```bash
# Cost savings by pausing unused endpoints
oumi deploy stop --endpoint-id ep-123 --provider together
oumi deploy start --endpoint-id ep-123 --provider together --min-replicas 2
```

---

#### 2. Dynamic Hardware Discovery (Fireworks.ai)

**Current**: Uses hardcoded list of 4 GPU types
**Missing**: Query actual hardware availability from API

**Impact**:

- Cannot discover new GPU types as Fireworks adds them
- No availability checking before deployment
- No hardware filtering by model compatibility

**Implementation Needed**:

```python
async def list_hardware(self, model_id: Optional[str] = None) -> list[HardwareConfig]:
    """Query available hardware from Fireworks API."""
    # Check if Fireworks has /hardware or /deployment-shapes endpoint
    # Fall back to hardcoded list if not available
    pass
```

**Investigation Required**: Check if Fireworks exposes hardware discovery API

---

### 🟡 Priority 2: Important for Developer Experience

#### 3. Endpoint Testing API

**Current**: CLI has basic `test` command via external HTTP call
**Missing**: Built-in client method for endpoint testing

**Impact**: Users must switch to OpenAI SDK or manual HTTP for testing
**Current Workaround**: `oumi deploy test` uses separate HTTP client

**Implementation Needed**:

```python
async def test_endpoint(
    self,
    endpoint_url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> dict:
    """Send test prompt to deployed endpoint."""
    # Use OpenAI-compatible /v1/chat/completions format
    pass
```

**Benefits**:

- Unified interface for testing
- No need to install separate SDK
- Can validate deployment immediately after creation

---

#### 4. Model Optimization Options (Fireworks.ai)

**Current**: Basic `prepare_model()` exists but unexposed
**Missing**: Full quantization and optimization configuration

**Available Options** (from Fireworks API):

- Precision selection: FP32, FP16, BF16, INT8, INT4
- Memory optimization strategies
- Batch size tuning
- KV cache configuration

**Implementation Enhancement**:

```python
@dataclass
class ModelPreparationConfig:
    """Configuration for model optimization."""
    precision: str = "fp16"  # fp32, fp16, bf16, int8, int4
    max_batch_size: Optional[int] = None
    kv_cache_dtype: Optional[str] = None

async def prepare_model(
    self,
    model_id: str,
    config: ModelPreparationConfig
) -> dict:
    """Optimize model for deployment."""
    pass
```

**CLI Impact**:

```bash
oumi deploy prepare-model \
  --model-id my-model \
  --precision int8 \
  --max-batch-size 32 \
  --provider fireworks
```

---

#### 5. Deployment Health Monitoring

**Missing Across Both**: Real-time deployment health metrics

**Potential Data** (if exposed by providers):

- Current request rate
- Average latency
- Error rate
- Active replica count
- Token throughput
- GPU utilization

**Implementation**:

```python
@dataclass
class DeploymentMetrics:
    """Real-time deployment metrics."""
    requests_per_second: float
    avg_latency_ms: float
    error_rate: float
    active_replicas: int
    timestamp: datetime

async def get_deployment_metrics(
    self,
    endpoint_id: str,
    window: str = "1h"  # 1h, 24h, 7d
) -> DeploymentMetrics:
    """Fetch deployment health metrics."""
    pass
```

**Status**: Needs investigation - unclear if providers expose these via API

---

### 🟢 Priority 3: Nice-to-Have Features

#### 6. Model Deployment Presets (Fireworks.ai)

**Current**: Manual hardware + autoscaling configuration
**Missing**: Deployment Shapes API for predefined configs

**Benefits**:

- Quick deployment with optimized settings
- Consistency across deployments
- Learn from Fireworks best practices

**API**: `/v1/deployment-shapes` (needs investigation)

---

#### 7. Cost Estimation

**Missing**: Pre-deployment cost calculation

**Desired Features**:

- Estimate monthly cost based on hardware + replica config
- Compare costs across different hardware options
- Show cost per token/request estimates

**Implementation**:

```python
@dataclass
class CostEstimate:
    """Estimated deployment costs."""
    hourly_cost: float
    monthly_cost: float  # Assuming 24/7
    cost_per_1k_tokens: Optional[float]
    currency: str = "USD"

async def estimate_deployment_cost(
    self,
    hardware: HardwareConfig,
    autoscaling: AutoscalingConfig
) -> CostEstimate:
    """Calculate estimated costs."""
    # Based on provider pricing tables
    pass
```

**Status**: Requires provider pricing data (likely not in API, need to hardcode)

---

#### 8. Multi-Region Support (Together.ai)

**Current**: No region selection
**Missing**: Regions API for geographic placement

**Together.ai Features**:

- List available regions
- Deploy to specific region
- Multi-region failover

**Impact**: Limited for users needing low-latency in specific geographies
**API Reference**: Together Regions API (low priority)

---

## Part 3: Out of Scope Features

The following features are available in provider APIs but are NOT part of model deployment:

**Training & Data Management**:

- ❌ Fine-tuning job creation/management
- ❌ Dataset uploads and validation
- ❌ File management for training data
- ❌ GPU cluster provisioning

**Inference Operations**:

- ❌ Batch inference jobs
- ❌ Evaluation job execution
- ❌ Embeddings/reranking endpoints
- ❌ Audio/video generation
- ❌ Code interpreter

**Administration**:

- ❌ Account/user management
- ❌ API key rotation
- ❌ Secrets management
- ❌ Billing/quota configuration

**Rationale**: These are separate concerns that should use dedicated tools or provider SDKs

---

## Part 4: Implementation Recommendations

### Phase 1: Essential Deployment Features (2-3 days)

**Goal**: Enable cost management and endpoint lifecycle operations

1. **Endpoint Start/Stop** (Together.ai)
   - Add `start_endpoint()` and `stop_endpoint()` methods to client
   - Update CLI with `oumi deploy start` and `oumi deploy stop` commands
   - Update worker activities: `start_deployment_workflow` and `stop_deployment_workflow`
   - Add `STOPPED` state to `DeploymentStatus` enum

2. **Hardware Discovery Fix** (Fireworks.ai)
   - Investigate if Fireworks has hardware/deployment-shapes API
   - If yes: Query from API instead of hardcoded list
   - If no: Document hardcoded list with update instructions
   - Add timestamp/version tracking for hardcoded data

3. **Endpoint Testing API**
   - Add `test_endpoint()` method to base client
   - Implement OpenAI-compatible chat completion requests
   - Support both sync and streaming responses
   - Update CLI `test` command to use client method

**Critical Files**:

- `oumi/src/oumi/deploy/base_client.py` - Add start/stop/test methods
- `oumi/src/oumi/deploy/together_client.py` - Implement start/stop
- `oumi/src/oumi/deploy/fireworks_client.py` - Fix hardware discovery
- `oumi/src/oumi/cli/deploy.py` - Add start/stop commands
- `worker/src/worker/workflows/deploy_model_workflow.py` - Start/stop workflows
- `shared/src/shared/models.py` - Add STOPPED status

**Estimated Effort**: 12-16 hours

---

### Phase 2: Model Optimization & Monitoring (3-5 days)

**Goal**: Enable advanced deployment configuration and monitoring

1. **Model Optimization** (Fireworks.ai)
   - Expose and enhance `prepare_model()` method
   - Add `ModelPreparationConfig` dataclass for options
   - Support precision selection (FP16, INT8, INT4)
   - Add CLI: `oumi deploy prepare-model` command
   - Document optimization trade-offs (speed vs accuracy vs memory)

2. **Deployment Health Monitoring**
   - Investigate if providers expose metrics APIs
   - If yes: Implement `get_deployment_metrics()` method
   - If no: Document limitation and alternative (web dashboards)
   - Add CLI: `oumi deploy metrics --endpoint-id ep-123`

3. **Cost Estimation** (Hardcoded pricing data)
   - Research provider pricing models
   - Implement `estimate_deployment_cost()` helper
   - Add to `oumi deploy up` output (show estimate before deploy)
   - Update documentation with cost optimization tips

**Critical Files**:

- `oumi/src/oumi/deploy/fireworks_client.py` - Enhance prepare_model()
- `oumi/src/oumi/deploy/base_client.py` - Add metrics/cost methods
- `oumi/src/oumi/cli/deploy.py` - Add prepare-model, metrics commands
- `oumi/configs/examples/deploy/README.md` - Cost optimization guide

**Estimated Effort**: 20-24 hours

---

### Phase 3: Nice-to-Have Features (Optional)

**Goal**: Polish and advanced capabilities

1. **Deployment Presets** (Fireworks.ai)
   - Investigate deployment shapes API
   - If available, add preset selection to CLI
   - Document recommended configs for common use cases

2. **Multi-Region Support** (Together.ai)
   - Add region parameter to create_endpoint()
   - Query available regions from API
   - Update CLI with --region flag

3. **Enhanced Documentation**
   - Deployment best practices guide
   - Cost optimization strategies
   - Troubleshooting common issues
   - Performance benchmarking guide

**Estimated Effort**: 16-20 hours (if pursued)

---

## Part 5: Design Considerations

### 1. API Client Architecture

**Current Structure**:

```
BaseDeploymentClient (abstract)
├── TogetherDeploymentClient
└── FireworksDeploymentClient
```

**Proposed Changes**:

- Add methods to `BaseDeploymentClient` for start/stop/test operations
- Mark as optional with `NotImplementedError` for unsupported providers
- Keep single-file client design (no composition needed)

**New Methods in Base**:

```python
async def start_endpoint(self, endpoint_id: str, min_replicas: int = 1) -> Endpoint
async def stop_endpoint(self, endpoint_id: str) -> Endpoint
async def test_endpoint(self, endpoint_url: str, prompt: str, **kwargs) -> dict
async def get_deployment_metrics(self, endpoint_id: str, window: str) -> DeploymentMetrics
async def estimate_deployment_cost(self, hardware, autoscaling) -> CostEstimate
```

**Rationale**: Keep it simple - all deployment features in one client

---

### 2. CLI Organization

**Current**: `oumi deploy <command>`

**Proposed Additions**:

```bash
# Existing commands (keep as-is)
oumi deploy upload
oumi deploy create-endpoint
oumi deploy list
oumi deploy list-models
oumi deploy status
oumi deploy delete
oumi deploy delete-model
oumi deploy list-hardware
oumi deploy test
oumi deploy up

# New commands (Phase 1)
oumi deploy start --endpoint-id <id> --provider <provider>
oumi deploy stop --endpoint-id <id> --provider <provider>

# New commands (Phase 2)
oumi deploy prepare-model --model-id <id> --precision int8 --provider fireworks
oumi deploy metrics --endpoint-id <id> --provider <provider>
```

**No Breaking Changes**: All existing commands remain unchanged

---

### 3. Backward Compatibility Strategy

**Requirements**:

- Existing `BaseDeploymentClient` interface must not break
- All current methods remain unchanged
- New methods are additive only
- CLI commands remain stable (only additions)
- Worker activities backward compatible

**Implementation**:

```python
# New methods in BaseDeploymentClient with default NotImplementedError
@abstractmethod
async def start_endpoint(self, endpoint_id: str, min_replicas: int = 1) -> Endpoint:
    """Start a stopped endpoint. Optional - may raise NotImplementedError."""
    raise NotImplementedError(
        f"{self.provider} does not support endpoint start/stop operations"
    )
```

**Migration Path**:

1. Add new abstract methods with NotImplementedError defaults
2. Implement in providers that support them (Together for start/stop)
3. Existing code continues to work without changes

---

## Part 6: Testing Requirements

### Unit Tests Needed

**Phase 1 Tests**:

- Mock `PATCH /v1/endpoints/{id}` for start/stop operations
- Test `start_endpoint()` with various min_replicas values
- Test `stop_endpoint()` setting replicas to 0
- Test `test_endpoint()` with OpenAI-compatible payloads
- NotImplementedError handling for unsupported providers
- Hardware discovery with mocked/hardcoded data

**Phase 2 Tests**:

- Mock model preparation API responses (Fireworks)
- Test `ModelPreparationConfig` validation
- Cost estimation calculations with various hardware configs
- Metrics parsing (if API available)

**Test Files**:

- `oumi/tests/deploy/test_together_client.py` - Add start/stop tests
- `oumi/tests/deploy/test_fireworks_client.py` - Add prepare_model tests
- `oumi/tests/deploy/test_base_client.py` - Test NotImplementedError defaults

---

### Integration Tests Needed

**End-to-End Workflows**:

1. Deploy model → Stop endpoint → Verify stopped → Start endpoint → Verify running
2. Upload model → Prepare with INT8 → Deploy → Test inference quality
3. Create deployment → Get metrics → Compare with expected ranges
4. Estimate cost → Deploy → Verify actual cost aligns with estimate

**Manual Testing Checklist** (Phase 1):

- [ ] Deploy endpoint to Together.ai
- [ ] Stop endpoint and verify billing stops
- [ ] Start endpoint and verify it resumes
- [ ] Test endpoint with sample prompts
- [ ] Verify start/stop reflects in status command
- [ ] Delete endpoint completely

**Manual Testing Checklist** (Phase 2):

- [ ] Upload model to Fireworks
- [ ] Prepare model with INT8 quantization
- [ ] Deploy optimized model
- [ ] Compare latency/throughput vs FP16
- [ ] Get deployment metrics (if available)
- [ ] Verify cost estimation accuracy

---

## Part 7: Documentation Updates Needed

### API Documentation

- Add docstrings for new methods (start/stop/test/metrics/estimate_cost)
- Document NotImplementedError patterns
- Update README with new capabilities
- Add cost optimization section

### CLI Documentation

**Update `oumi/configs/examples/deploy/README.md`**:

```markdown
## Cost Optimization

### Pause Unused Endpoints (Together.ai)
```bash
# Stop to save costs when not in use
oumi deploy stop --endpoint-id ep-123 --provider together

# Restart when needed
oumi deploy start --endpoint-id ep-123 --provider together
```

### Model Optimization (Fireworks.ai)

```bash
# Quantize to INT8 for 2-3x speedup
oumi deploy prepare-model \
  --model-id my-model \
  --precision int8 \
  --provider fireworks
```

### Monitor Deployment Health

```bash
oumi deploy metrics --endpoint-id ep-123 --provider together
```

```

### Example Workflows
- Deploy → Test → Monitor → Optimize → Redeploy
- Cost-conscious deployment with start/stop scheduling
- Quantization trade-offs (accuracy vs speed vs cost)

---

## Part 8: Success Metrics

### Phase 1 Success Criteria
- [ ] Start/stop operations work on Together.ai
- [ ] Endpoints can be paused to save costs
- [ ] Test endpoint works for both providers
- [ ] Hardware discovery fixed or documented for Fireworks
- [ ] Zero breaking changes to existing API
- [ ] All new methods have unit tests

### Phase 2 Success Criteria
- [ ] Model preparation exposed with quantization options
- [ ] Cost estimation helps users make informed decisions
- [ ] Metrics available (or limitation documented)
- [ ] Documentation updated with optimization guide

### User Impact Goals
- [ ] **Cost Savings**: Users can pause endpoints during off-hours
- [ ] **Performance**: Users can optimize models for speed/cost
- [ ] **Visibility**: Users understand deployment health and costs
- [ ] **Self-Service**: Reduced need to use provider web UIs

---

## Summary Table: Deployment Features Priority Matrix

| Feature | Together | Fireworks | Priority | Effort | Impact | Notes |
|---------|----------|-----------|----------|--------|--------|-------|
| Start/Stop Endpoints | ✅ API | N/A | P1 | Small | High | Cost savings |
| Endpoint Testing API | ✅ | ✅ | P1 | Small | Medium | Client integration |
| Hardware Discovery | ✅ API | ⚠️ Hardcoded | P1 | Small | Medium | Fix Fireworks |
| Model Optimization | N/A | ✅ API | P2 | Medium | High | Quantization |
| Deployment Metrics | ❓ | ❓ | P2 | Medium | Medium | Need investigation |
| Cost Estimation | Manual | Manual | P2 | Medium | High | Hardcoded pricing |
| Deployment Presets | N/A | ⚠️ Shapes API | P3 | Small | Low | Nice-to-have |
| Multi-Region | ✅ API | N/A | P3 | Medium | Low | Geographic placement |

**Legend**:
- ✅ = Provider supports via API
- ⚠️ = Partially supported or needs investigation
- ❓ = Unknown if API exists
- N/A = Not applicable to this provider
- Manual = Requires hardcoded data

---

## Sources & References

### Together.ai Documentation
- [API Reference](https://docs.together.ai/reference/endpoints-1)
- [Endpoints Management](https://docs.together.ai/reference/endpoints-1)
- [Files API](https://docs.together.ai/reference/files)
- [Fine-tuning](https://docs.together.ai/reference/finetune)
- [Batches](https://docs.together.ai/reference/batch-create)
- [Evaluations](https://docs.together.ai/reference/list-evaluations)

### Fireworks.ai Documentation
- [API Reference](https://docs.fireworks.ai/api-reference/introduction)
- [Deployments](https://docs.fireworks.ai/api-reference/create-deployment)
- [Fine-tuning Jobs](https://docs.fireworks.ai/api-reference/create-supervised-fine-tuning-job)
- [Datasets](https://docs.fireworks.ai/api-reference/create-dataset)
- [Evaluation Jobs](https://docs.fireworks.ai/api-reference/create-evaluation-job)
- [Models](https://docs.fireworks.ai/api-reference/delete-model)

### Implementation Files
- Current: `/Users/oussama/src/main/apiserver/oumi/src/oumi/deploy/`
- Tests: `/Users/oussama/src/main/apiserver/oumi/tests/deploy/`
- CLI: `/Users/oussama/src/main/apiserver/oumi/src/oumi/cli/deploy.py`
- Activities: `/Users/oussama/src/main/apiserver/worker/src/worker/activities/deploy_model_activities.py`
