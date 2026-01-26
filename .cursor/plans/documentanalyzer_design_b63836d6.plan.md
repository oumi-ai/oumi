---
name: Unified Analysis Framework
overview: Design a unified analysis framework that handles both conversations and documents with the same config format, metrics system, and user experience.
todos:
  - id: phase1-unified-result
    content: Create AnalysisResult with configurable output_fields (score, description, explanation, category, etc.)
    status: pending
  - id: phase1-output-fields
    content: Add output_fields parameter to existing LLMAnalyzer (backward compatible)
    status: pending
  - id: phase2-document-base
    content: Create DocumentAnalyzer base class (parallel to ConversationAnalyzer)
    status: pending
  - id: phase2-document-llm
    content: Create DocumentLLMAnalyzer with same params as LLMAnalyzer
    status: pending
  - id: phase3-document-pipeline
    content: Create DocumentAnalysisPipeline with same interface as AnalysisPipeline
    status: pending
  - id: phase3-document-config
    content: Add documents config section parsing and CLI auto-detection
    status: pending
  - id: phase4-chunking
    content: Add chunking utilities with early stopping for large documents
    status: pending
  - id: phase4-file-extraction
    content: Add file type support with PDF, DOCX, HTML text extraction
    status: pending
isProject: false
---

# Unified Analysis Framework Design

## Overview

Create a **single unified analysis framework** that handles both conversations and documents with:

- Same YAML config format
- Same metrics/output fields
- Same test/assertion system
- Same CLI experience

## Design Principle: Configurable Output Schema

Instead of separate `LLMJudgmentMetrics` and `DocumentAnalysisResult`, use a **single configurable result model** where the user specifies which fields they want.

## Unified Result Model

```python
class AnalysisResult(BaseModel):
    """Unified result model with configurable fields."""
    
    # All possible output fields (optional by default)
    score: int | None = None           # 0-100 score
    reasoning: str | None = None       # LLM explanation
    description: str | None = None     # What the content is
    explanation: str | None = None     # Why it's useful
    category: str | None = None        # Categorical label
    judgment: bool | None = None       # Pass/fail boolean
    confidence: float | None = None    # LLM confidence
    
    # Derived fields (auto-computed)
    label: str | None = None           # poor/fair/good/excellent from score
    passed: bool | None = None         # score >= threshold
    
    # Metadata
    criteria: str                      # What was evaluated
    raw_response: str | None = None
    error: str | None = None
    
    @classmethod
    def from_llm_response(
        cls,
        parsed: dict,
        output_fields: list[str],
        criteria: str,
        **kwargs
    ) -> "AnalysisResult":
        """Create result with only the requested fields populated."""
```

## Analyzer Classes

### ConversationLLMAnalyzer (existing, enhanced)

```python
class LLMAnalyzer(ConversationAnalyzer[AnalysisResult]):
    """LLM analyzer for conversations (existing, add output_fields)."""
    
    def __init__(
        self,
        target_scope: TargetScope = TargetScope.CONVERSATION,
        output_fields: list[str] = ["score", "reasoning"],  # NEW
        category_values: list[str] | None = None,
        criteria: str | None = None,
        prompt_template: str | None = None,
        model_name: str = "gpt-4o-mini",
        ...
    ): ...
    
    def analyze(self, conversation: Conversation) -> AnalysisResult:
        """Analyze a conversation."""
```

### DocumentLLMAnalyzer (new, parallel structure)

```python
class DocumentLLMAnalyzer(DocumentAnalyzer[AnalysisResult]):
    """LLM analyzer for documents (new, same params as conversation)."""
    
    def __init__(
        self,
        output_fields: list[str] = ["description", "explanation", "category"],
        category_values: list[str] | None = None,
        criteria: str | None = None,
        prompt_template: str | None = None,
        model_name: str = "gpt-4o-mini",
        # Document-specific
        enable_chunking: bool = False,
        chunk_size: int = 50000,
        ...
    ): ...
    
    def analyze(self, document: str | Path) -> AnalysisResult:
        """Analyze a document."""
```

**Same parameters** (except document-specific chunking), **same result type**.

## Parallel Config Formats

Conversation and document configs are **separate files** but share the **same structure**.

### Conversation Analysis Config (existing)

```yaml
# conversation_analysis.yaml
dataset:
  name: OpenAssistant/oasst2
  split: train
  sample_count: 100

analyzers:
  - id: usefulness
    params:
      target_scope: last_turn
      model_name: gpt-4o-mini
      
  - id: llm
    params:
      criteria_name: quality_assessment
      output_fields: [score, description, category]
      category_values: [excellent, good, fair, poor]

tests:
  - id: high_usefulness
    metric: usefulness.passed
    condition: "== True"
    min_percentage: 80.0
    
  - id: quality_scores
    metric: quality_assessment.score
    operator: ">="
    value: 70
    min_percentage: 80.0
```

### Document Analysis Config (new, same structure!)

```yaml
# document_analysis.yaml
documents:
  path: /path/to/files/
  file_pattern: "*.pdf"  # or specific files
  # OR
  files:
    - /path/to/doc1.pdf
    - /path/to/doc2.txt

analyzers:
  - id: file_usefulness  # preset
    params:
      model_name: gpt-4o-mini
      
  - id: llm
    params:
      criteria_name: content_quality
      output_fields: [score, description, category]
      category_values: [excellent, good, fair, poor]
      enable_chunking: true
      chunk_size: 50000

tests:
  - id: valid_categories
    metric: file_usefulness.category
    condition: "in [full_dataset, reference_dataset]"
    min_percentage: 50.0
    
  - id: quality_scores
    metric: content_quality.score
    operator: ">="
    value: 70
    min_percentage: 80.0
```

**Key point**: Same `analyzers` and `tests` structure, just different `dataset` vs `documents` input source.

## Document Content Extraction

```python
class DocumentExtractor:
    """Extracts text content from various file types."""
    
    def extract(self, source: str | Path) -> str:
        """Extract text from file or return raw string."""
        if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
            return self._extract_from_file(Path(source))
        return source  # Already text
        
    def _extract_from_file(self, path: Path) -> str:
        """Extract text based on file type."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix in [".docx", ".doc"]:
            return self._extract_docx(path)
        elif suffix in [".html", ".htm"]:
            return self._extract_html(path)
        else:
            return path.read_text()  # Plain text
```

## Parallel Pipelines

Separate pipelines for conversations vs documents, but same interface and result format.

```python
# Conversation pipeline (existing)
class ConversationAnalysisPipeline:
    def run(self, conversations: list[Conversation]) -> dict[str, list[AnalysisResult]]:
        ...

# Document pipeline (new, same interface pattern)
class DocumentAnalysisPipeline:
    def run(self, documents: list[str | Path]) -> dict[str, list[AnalysisResult]]:
        ...
```

Both return `dict[str, list[AnalysisResult]]` so tests and output formats are identical.

## Unified Presets

Merge conversation and document presets:

```python
UNIFIED_PRESETS = {
    # Conversation quality (existing)
    "usefulness": {...},
    "safety": {...},
    "coherence": {...},
    
    # Document characterization (new)
    "file_usefulness": {
        "prompt": "Assess this document for LLM training use...",
        "output_fields": ["description", "explanation", "category"],
        "category_values": ["assistant_reference", "synthesis_reference", 
                          "full_dataset", "reference_dataset", "unknown"],
        "input_type": "document",
    },
    "document_quality": {
        "prompt": "Evaluate the quality of this document...",
        "output_fields": ["score", "description", "category"],
        "category_values": ["excellent", "good", "fair", "poor"],
        "input_type": "document",
    },
}
```

## CLI Experience

Same command for everything:

```bash
# Analyze conversations (current)
oumi analyze --config conversation_analysis.yaml --typed

# Analyze documents (new, same command!)
oumi analyze --config document_analysis.yaml --typed

# Mixed analysis
oumi analyze --config mixed_analysis.yaml --typed

# List all available presets (conversations + documents)
oumi analyze --list-metrics
```

## User Experience Comparison

| Aspect | Conversation Config | Document Config |

|--------|---------------------|-----------------|

| Input source | `dataset:` (name/path) | `documents:` (path/files) |

| Analyzers section | Same structure | Same structure |

| Tests section | Same structure | Same structure |

| Output format | `AnalysisResult` | `AnalysisResult` |

| CLI command | `oumi analyze --typed` | `oumi analyze --typed` |

**Same learning curve** - learn one config structure, use for both.

## Test Results Format (Identical)

Both conversation and document analysis produce the same output structure:

### analysis_results.json

```json
{
  "usefulness": [
    {"score": 75, "reasoning": "...", "category": "good", "passed": true, ...}
  ],
  "file_usefulness": [
    {"score": 80, "description": "...", "category": "full_dataset", "passed": true, ...}
  ]
}
```

### test_results.json

```json
{
  "results": [
    {
      "test_id": "high_usefulness",
      "passed": true,
      "affected_percentage": 85.0,
      "threshold": 80.0,
      ...
    }
  ],
  "total_tests": 3,
  "passed_tests": 2,
  "pass_rate": 66.7
}
```

### summary.json

```json
{
  "total_items": 100,
  "analyzers_run": ["usefulness", "quality"],
  "tests": {"total": 3, "passed": 2, "pass_rate": 66.7}
}
```

Same structure whether analyzing conversations or documents.

## Migration Path

### From Current LLMAnalyzer

```yaml
# Before
- id: llm
  params:
    criteria: usefulness
    judgment_type: score
    
# After (backward compatible)
- id: llm
  params:
    criteria: usefulness
    # output_fields defaults to [score, reasoning] for preset criteria
```

### From API analyze_file_activity

```yaml
# New config replaces custom Python code
- id: llm
  input_type: document
  params:
    criteria: file_usefulness
    enable_chunking: true
```

## Implementation Phases

### Phase 1: Unified Result Model

- Create `AnalysisResult` with all optional fields (score, description, explanation, category, etc.)
- Add `output_fields` parameter to existing `LLMAnalyzer`
- Backward compatible: default `output_fields=["score", "reasoning"]` matches current behavior

### Phase 2: DocumentLLMAnalyzer

- Create `DocumentAnalyzer` base class (parallel to `ConversationAnalyzer`)
- Create `DocumentLLMAnalyzer` with same params as `LLMAnalyzer`
- Both return `AnalysisResult`

### Phase 3: Document Pipeline and CLI

- Create `DocumentAnalysisPipeline` (same interface as `AnalysisPipeline`)
- Add `documents:` config section parsing
- Same CLI: `oumi analyze --typed` auto-detects config type
- Same test engine works for both

### Phase 4: Chunking and File Extraction

- Add chunking utilities for large documents
- PDF/DOCX/HTML text extraction
- Document-specific presets (file_usefulness, etc.)
