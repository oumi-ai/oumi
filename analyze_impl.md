# Oumi Analyze Feature - Production Readiness Implementation

## Overview
Implementation plan to bring the `oumi analyze` feature to production readiness.

## Priority 1: Critical (Blocking) - COMPLETED

### 1.1 Add CLI Command
- [x] Create `src/oumi/cli/analyze.py` with typer command
- [x] Register in `src/oumi/cli/main.py`
- [x] Support YAML config loading
- [x] Add progress display via Rich console status

### 1.2 Fix Dataset-Agnostic Summaries
- [x] Refactor `_get_message_level_summary()` to detect column structure dynamically
- [x] Refactor `_get_conversation_level_summary()` similarly
- [x] Refactor `_get_conversation_turns_summary()` to handle non-conversation datasets
- [x] Remove hardcoded metric suffix list - now uses registered analyzer IDs

### 1.3 Add Result Export
- [x] Implement export in CLI (`_export_results()` function)
- [x] Support CSV, JSON, Parquet formats
- [x] Use `output_path` from config (can be overridden via `--output`)
- [x] Export message analysis, conversation analysis, and JSON summary

## Priority 2: Significant Gaps - COMPLETED

### 2.1 HuggingFace Hub Loading
- [x] Implement direct HF dataset loading in `load_dataset_from_config()`
- [x] Created `HuggingFaceDatasetWrapper` class for compatibility
- [x] Handle dataset schemas dynamically

### 2.2 Fix Iterable Dataset Filtering
- [x] Raise explicit `NotImplementedError` instead of silently returning original
- [x] Clear error message explaining limitation and alternatives

### 2.3 Memory Optimization
- [x] Replace `copy.deepcopy()` with `copy.copy()` + DataFrame slice
- [ ] Add chunked processing option for large datasets (future)

## Priority 3: Additional Analyzers - TODO (Future)

### 3.1 Language Detector Analyzer
- [ ] Detect language of text content
- [ ] Use langdetect or fasttext

### 3.2 Quality/Toxicity Analyzer
- [ ] Integrate with existing quality scoring

### 3.3 Statistics Analyzer
- [ ] Compute distribution statistics
- [ ] Detect outliers

## Priority 4: Documentation - COMPLETED

### 4.1 User Guide
- [x] Create docs/user_guides/analyze/analyze.md - comprehensive guide
- [x] Create docs/user_guides/analyze/analyze_config.md - config reference
- [x] Add to docs/index.md toctree
- [x] Add YAML config examples in configs/examples/analyze/
- [x] Add Python API examples in documentation

---

## Implementation Progress

### Completed
- CLI command `oumi analyze` with full config support
- Result export to CSV/JSON/Parquet formats
- Dataset-agnostic summary generation
- HuggingFace Hub dataset loading
- Explicit error for iterable dataset filtering
- Memory-efficient filtering using shallow copy
- All 61 existing tests pass

### Files Modified
- `src/oumi/cli/analyze.py` (NEW) - CLI command implementation
- `src/oumi/cli/main.py` - Register analyze command
- `src/oumi/cli/alias.py` - Add ANALYZE alias type
- `src/oumi/core/analyze/dataset_analyzer.py` - Refactor summary methods, fix filtering
- `src/oumi/utils/analysis_utils.py` - Add HuggingFace Hub loading

### Documentation Added
- `docs/user_guides/analyze/analyze.md` (NEW) - Comprehensive user guide
- `docs/user_guides/analyze/analyze_config.md` (NEW) - Configuration reference
- `docs/index.md` - Added analyze to user guides toctree
- `configs/examples/analyze/basic_analyze.yaml` (NEW) - Basic example config (runnable)
- `configs/examples/analyze/analyze_with_tokens.yaml` (NEW) - Token counting example (runnable)
- `configs/examples/analyze/analyze_local_dataset.yaml` (NEW) - Alpaca format example (runnable)

All example configs use local data files from `data/dataset_examples/` and are directly runnable.

### Usage
```bash
# Analyze a dataset with YAML config
oumi analyze --config path/to/analyze_config.yaml

# Export to specific format and directory
oumi analyze --config config.yaml --output ./results --format parquet

# With verbose logging
oumi analyze --config config.yaml --verbose
```

### Example Config (analyze_config.yaml)
```yaml
dataset_source: config
dataset_name: my_hf_dataset  # HuggingFace Hub dataset
split: train
sample_count: 1000
output_path: ./analysis_output
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: false
```

### Blocked
- (none)
