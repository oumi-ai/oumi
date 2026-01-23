---
name: Typed Analyzer Architecture
overview: Replace DataFrame-centric analyze approach with typed Conversation objects and Pydantic results. Start with conversation + preference support, cleanly separate metrics from tests.
todos:
  - id: base-classes
    content: Create base analyzer classes (MessageAnalyzer, ConversationAnalyzer, DatasetAnalyzer, PreferenceAnalyzer)
    status: completed
  - id: result-models
    content: Create Pydantic result models (LengthMetrics as first example)
    status: completed
  - id: length-analyzer
    content: Implement LengthAnalyzer with new typed architecture
    status: completed
  - id: pipeline
    content: Create AnalysisPipeline to orchestrate analyzers
    status: completed
  - id: dataframe-bridge
    content: Implement to_analysis_dataframe() utility
    status: completed
  - id: test-engine
    content: Create TestEngine for pure validation on typed results
    status: completed
  - id: config-integration
    content: Update AnalyzeConfig to work with new architecture
    status: completed
  - id: cli-update
    content: Update oumi analyze CLI command
    status: completed
  - id: custom-metrics
    content: Add custom_metrics support for inline Python metrics
    status: completed
---

# Typed Analyzer Architecture

## Goals

1. Replace DataFrame-centric API with typed Conversation objects and Pydantic results
2. Clear analyzer hierarchy by scope (Message, Conversation, Dataset, Preference)
3. Separate metrics (computed, cached) from tests (pure validation)
4. Start with conversations + preferences, extensible for other types later

## Architecture Overview

```mermaid
flowchart TB
    subgraph input [Input - Conversations]
        D[Dataset]
    end

    subgraph analyzers [Analyzer Hierarchy]
        MA[MessageAnalyzer]
        CA[ConversationAnalyzer]
        DA[DatasetAnalyzer - cross-sample]
        PA[PreferenceAnalyzer - DPO]
    end

    subgraph results [Typed Results - Pydantic]
        R[LengthMetrics / QualityMetrics / etc.]
    end

    subgraph pipeline [AnalysisPipeline]
        P[Orchestrate + Cache]
    end

    subgraph validation [Tests - Pure Validation]
        T[TestEngine]
    end

    subgraph output [Output]
        O1[Typed Results]
        O2[DataFrame via bridge]
        O3[Artifacts]
    end

    D --> P
    P --> MA
    P --> CA
    P --> DA
    P --> PA
    MA --> R
    CA --> R
    DA --> R
    PA --> R
    R --> T
    R --> O1
    O1 --> O2
    O1 --> O3
```

## Base Analyzer Classes

**File:** `src/oumi/analyze/base.py`

```python
class MessageAnalyzer(ABC, Generic[TResult]):
    def analyze(self, message: Message) -> TResult: ...
    def analyze_batch(self, messages: list[Message]) -> list[TResult]: ...

class ConversationAnalyzer(ABC, Generic[TResult]):
    def analyze(self, conversation: Conversation) -> TResult: ...
    def analyze_batch(self, conversations: list[Conversation]) -> list[TResult]: ...

class DatasetAnalyzer(ABC, Generic[TResult]):
    """Cross-sample operations (deduplication, clustering)."""
    def analyze(self, conversations: list[Conversation]) -> TResult: ...

class PreferenceAnalyzer(ABC, Generic[TResult]):
    """For DPO preference pairs."""
    def analyze(self, chosen: Conversation, rejected: Conversation) -> TResult: ...
```

## Typed Result Models

**File:** `src/oumi/analyze/results/length.py`

```python
class LengthMetrics(BaseModel):
    total_chars: int
    total_words: int
    total_tokens: int | None = None
    avg_message_length: float
    message_lengths: list[int]
```

## Example Implementation

**File:** `src/oumi/analyze/analyzers/length.py`

```python
class LengthAnalyzer(ConversationAnalyzer[LengthMetrics]):
    def analyze(self, conversation: Conversation) -> LengthMetrics:
        lengths = [len(m.content.split()) for m in conversation.messages]
        return LengthMetrics(
            total_chars=sum(len(m.content) for m in conversation.messages),
            total_words=sum(lengths),
            avg_message_length=sum(lengths) / len(lengths) if lengths else 0,
            message_lengths=lengths,
        )
```

## AnalysisPipeline

**File:** `src/oumi/analyze/pipeline.py`

```python
class AnalysisPipeline:
    def __init__(self, analyzers: list, cache_dir: Path | None = None): ...
    def run(self, conversations: list[Conversation]) -> dict[str, list[BaseModel]]: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

## DataFrame Bridge

**File:** `src/oumi/analyze/utils/dataframe.py`

```python
def to_analysis_dataframe(
    conversations: list[Conversation],
    results: dict[str, list[BaseModel]],
) -> pd.DataFrame:
    """Convert typed results to DataFrame when needed for analysis/viz."""
```

## Tests (Pure Validation)

Tests operate on typed results - no computation:

```yaml
tests:
  - id: no_pii
    metric: QualityMetrics.has_pii  # Access typed field
    condition: "== False"
    max_percentage: 1.0
```

## File Structure

```
src/oumi/analyze/
├── __init__.py           # Public API
├── base.py               # Base analyzer classes
├── pipeline.py           # AnalysisPipeline
├── results/              # Pydantic models
│   ├── length.py
│   └── quality.py
├── analyzers/            # Implementations
│   ├── length.py
│   └── quality.py
├── testing/              # Test engine
│   └── engine.py
└── utils/
    └── dataframe.py      # to_analysis_dataframe()
```

## Extensibility

When other dataset types are needed later:

- Add `TextAnalyzer` base class for pretraining
- Add `KTOAnalyzer` for KTO datasets
- Pipeline auto-detects and routes to appropriate analyzers

## Migration

1. Only `LengthAnalyzer` exists in main - refactor as proof of concept
2. Build new architecture in `src/oumi/analyze/`
3. Update CLI to use new pipeline
4. Deprecate old `SampleAnalyzer` approach
