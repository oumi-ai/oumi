# Dataset Analysis Configuration Examples

This directory contains example configurations for analyzing dataset quality using Oumi's dataset analyzer.

## Available Configurations

### Basic Analysis

- **`analyze.yaml`** - Simple length-based analysis (character, word, sentence, token counts)
- **`quality_basic.yaml`** - Essential quality checks for quick dataset validation

### Comprehensive Analysis

- **`quality_comprehensive.yaml`** - All Phase 1-3 quality analyzers for thorough assessment
- **`quality_deduplication.yaml`** - Focus on detecting duplicate and near-duplicate content
- **`quality_request_types.yaml`** - Analyze distribution of request types and content patterns
- **`quality_conversation.yaml`** - Specialized for multi-turn conversation datasets (Oumi format)

## Quick Start

### Basic Quality Check

```bash
oumi analyze --config configs/examples/analyze/quality_basic.yaml
```

This will:
- Detect exact duplicates
- Find empty or invalid content
- Validate format and schema
- Check for encoding issues
- Compute basic length statistics

**Output files:**
- `quality_analysis_output/message_analysis.csv` - Per-sample metrics
- `quality_analysis_output/analysis_summary.json` - Statistical summary

### Comprehensive Quality Assessment

```bash
oumi analyze --config configs/examples/analyze/quality_comprehensive.yaml
```

This applies all Phase 1-3 analyzers:
- **Phase 1**: Duplicates, format validation, encoding issues, empty content
- **Phase 2**: Statistical outliers, n-gram patterns, repetition, vocabulary
- **Phase 3**: Request type classification, readability scores, conversation structure

Recommended before major training runs.

### Deduplication Analysis

```bash
oumi analyze --config configs/examples/analyze/quality_deduplication.yaml
```

Focuses on finding and removing duplicate content:
- Exact duplicates (hash-based)
- Overrepresented n-gram patterns
- Repetitive content within samples
- Statistical clustering of similar content

### Request Type Distribution

```bash
oumi analyze --config configs/examples/analyze/quality_request_types.yaml
```

Analyzes the composition of your dataset:
- Classifies requests into 15+ categories
- Identifies underrepresented request types
- Analyzes common phrases and patterns
- Computes readability metrics

## Customizing Configurations

### Using Your Own Dataset

Replace the `dataset_path` with your local file:

```yaml
dataset_path: path/to/your/dataset.jsonl
```

Or use a HuggingFace dataset:

```yaml
dataset_name: username/dataset-name
split: train
```

### Limiting Sample Count

For testing on large datasets:

```yaml
sample_count: 1000  # Analyze first 1000 samples
```

### Specifying a Tokenizer

For accurate token counting:

```yaml
tokenizer_name: meta-llama/Llama-2-7b-hf
# or
tokenizer_name: openai-community/gpt2
```

### Adjusting Analyzer Parameters

Each analyzer accepts parameters. Example:

```yaml
analyzers:
  - id: duplicate
    params:
      normalize_whitespace: true  # Treat multiple spaces as one
      case_sensitive: false       # Ignore case differences

  - id: ngram
    params:
      n: 4                        # Use 4-grams instead of 3-grams
      min_document_frequency: 0.1 # Flag n-grams in >10% of samples
      top_k: 100                  # Track top 100 n-grams
```

## Available Analyzers

### Phase 1: Core Deterministic (Tier 1)

| Analyzer ID | Description | Key Metrics |
|-------------|-------------|-------------|
| `duplicate` | Exact duplicate detection | `*_is_duplicate`, `*_duplicate_count`, `*_hash` |
| `empty_content` | Empty/whitespace content | `*_is_empty`, `*_has_content`, `*_stripped_length` |
| `format_validation` | Schema validation | `format_missing_fields`, `format_empty_fields`, `format_is_valid` |
| `encoding` | Text encoding issues | `*_has_replacement_chars`, `*_control_char_count` |
| `role_sequence` | Role alternation (conversations) | `role_sequence_valid`, `consecutive_same_role` |

### Phase 2: Statistical Analysis

| Analyzer ID | Description | Key Metrics |
|-------------|-------------|-------------|
| `length` | Length statistics | `*_char_count`, `*_word_count`, `*_token_count` |
| `statistical_outlier` | Outlier detection | `*_zscore`, `*_percentile`, `*_is_outlier_iqr` |
| `ngram` | N-gram frequency | `contains_overrepresented`, `sample_ngram_uniqueness` |
| `repetition` | Repetitive patterns | `char_repetition_ratio`, `word_repetition_ratio` |
| `vocabulary` | Vocabulary diversity | `vocabulary_size`, `type_token_ratio`, `hapax_ratio` |

### Phase 3: Pattern & Classification

| Analyzer ID | Description | Key Metrics |
|-------------|-------------|-------------|
| `request_type` | Request classification | `request_type`, `request_type_is_unknown` |
| `category_distribution` | Category balance | `category_percentage`, `category_is_underrepresented` |
| `readability` | Readability scores | `flesch_reading_ease`, `flesch_kincaid_grade` |
| `conversation_structure` | Conversation metrics | `conv_turn_count`, `conv_user_assistant_ratio` |

## Output Files

After running analysis, you'll get:

```
output_path/
├── message_analysis.csv         # Per-sample/message metrics
├── conversation_analysis.csv    # Per-conversation metrics (if applicable)
├── analysis_summary.json        # Statistical summary
└── schema.json                  # Column schema information
```

### Understanding the Output

**message_analysis.csv** - One row per sample/message:
```csv
message_index,role,text_content,text_content_char_count,text_content_is_duplicate,...
0,user,"Explain...",234,False,...
1,assistant,"Sure...",456,False,...
```

**analysis_summary.json** - Statistical summary:
```json
{
  "text_content_char_count": {
    "mean": 234.5,
    "std": 123.4,
    "min": 10,
    "max": 1000,
    "median": 200
  },
  ...
}
```

## Dataset Format Support

### Alpaca Format
```json
{"instruction": "...", "input": "...", "output": "..."}
```

Use: `quality_basic.yaml`, `quality_comprehensive.yaml`, `quality_request_types.yaml`

### Oumi Conversation Format
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Use: `quality_conversation.yaml`

### Custom Formats

Specify required columns in `format_validation`:
```yaml
- id: format_validation
  params:
    required_columns: ["your_field_1", "your_field_2"]
    non_empty_columns: ["your_field_1"]
```

## Interpreting Results

### Common Issues and Actions

| Issue | Indicator | Action |
|-------|-----------|--------|
| **Exact duplicates** | `*_is_duplicate = True` | Remove duplicate samples |
| **Empty content** | `*_has_content = False` | Remove or investigate |
| **Format errors** | `format_is_valid = False` | Fix format or remove |
| **Encoding issues** | `*_has_encoding_issues = True` | Clean text encoding |
| **Outliers** | `*_is_outlier_zscore = True` | Review for quality |
| **Overused phrases** | `contains_overrepresented = True` | Consider removing templates |
| **Low diversity** | `type_token_ratio < 0.5` | Add more varied content |
| **Imbalanced types** | `request_type_is_unknown = True` | Check classification or add diversity |

### Next Steps

1. **Review flagged samples**: Sort by quality metrics, inspect outliers
2. **Remove problematic samples**: Filter out duplicates, empty content, format errors
3. **Balance dataset**: Add underrepresented request types
4. **Re-analyze**: Run analysis again after cleaning

## Advanced Usage

### Analyzing Specific Columns

For datasets with multiple text fields:

```yaml
analyzers:
  - id: length
    params:
      columns: ["instruction", "output"]  # Only analyze these
```

### Chain Multiple Analyses

Run different configs sequentially:

```bash
# Quick check first
oumi analyze --config configs/examples/analyze/quality_basic.yaml

# Then comprehensive if basic passes
oumi analyze --config configs/examples/analyze/quality_comprehensive.yaml
```

### Custom Output Paths

```yaml
output_path: ./analysis_results/alpaca_2024_01_15
```

## Troubleshooting

### "No text fields found"

Ensure your dataset has text fields and schema is properly detected. You may need to specify:

```yaml
analyzers:
  - id: duplicate
    # Analyzer will auto-detect text fields from schema
```

### "Tokenizer not found"

Install required model or use a different tokenizer:

```yaml
tokenizer_name: openai-community/gpt2  # Smaller, faster
# tokenizer_name: meta-llama/Llama-2-7b-hf  # Requires access
```

### Memory Issues

Reduce sample count:

```yaml
sample_count: 10000  # Analyze subset first
```

Or use fewer analyzers in a single pass.

## References

- [Dataset Quality Roadmap](../../../docs/development/dataset_quality_roadmap.md)
- [Oumi Documentation](https://docs.oumi.ai)
- [Analysis API Reference](https://docs.oumi.ai/api/analyze)
