# Oumi Analyze: Analyzer Documentation

## Overview

The `oumi analyze` system provides a modular framework for analyzing datasets before training. It includes 6 core analyzers that can be combined via configuration.

---

## Core Analyzers

### 1. Length Analyzer (`length`)

**Purpose**: Measures text size and structure metrics.

**Implementation**:
- Registered via `@register_sample_analyzer("length")`
- Iterates over text columns identified by schema
- Uses simple string operations for most metrics

**Metrics Produced**:
| Metric | Description | Method |
|--------|-------------|--------|
| `char_count` | Character count | `len(text)` |
| `word_count` | Word count | Whitespace split |
| `sentence_count` | Sentence count | Regex split on `[.!?]+` |
| `token_count` | Token count | HuggingFace tokenizer (optional) |

**Limitations**:
1. **Sentence splitting is naive** - Only splits on `.!?`, doesn't handle abbreviations ("Dr.", "U.S."), decimal numbers ("3.14"), or ellipses
2. **No multi-byte character handling** - Character count doesn't distinguish between ASCII and Unicode characters (emojis count as 1)
3. **Token counting requires explicit tokenizer** - Must pass tokenizer in config, no fallback
4. **No paragraph detection** - Could be useful for document-level analysis

---

### 2. Diversity Analyzer (`diversity`)

**Purpose**: Measures vocabulary richness and linguistic diversity.

**Implementation**:
- Uses `collections.Counter` for word frequency tracking
- Whitespace-based tokenization with optional lowercasing
- Computes ratio-based metrics to normalize for length

**Metrics Produced**:
| Metric | Description | Formula |
|--------|-------------|---------|
| `unique_words_ratio` | Lexical diversity | unique_words / total_words |
| `type_token_ratio` | Same as above (linguistic term) | types / tokens |
| `vocabulary_richness` | Length-normalized TTR | unique_words / sqrt(total_words) |
| `hapax_legomena_ratio` | Single-use words ratio | words_appearing_once / unique_words |

**Limitations**:
1. **No punctuation handling** - "word." and "word" treated as different tokens
2. **No stemming/lemmatization** - "run", "runs", "running" counted separately
3. **No stopword filtering** - Common words inflate diversity scores
4. **TTR still somewhat length-dependent** - Root TTR helps but doesn't fully solve
5. **No n-gram diversity** - Only unigrams considered
6. **No cross-sample diversity** - Only within-sample metrics

---

### 3. Format Analyzer (`format`)

**Purpose**: Detects structural formatting patterns in text.

**Implementation**:
- Pre-compiled regex patterns (class-level constants)
- Boolean detection with count metrics
- JSON validation via `json.loads()` with try/except
- Weighted complexity scoring

**Metrics Produced**:
| Metric | Description |
|--------|-------------|
| `has_markdown` | Headers, lists, bold, italic, links, images |
| `has_json` | Fenced JSON blocks or inline JSON objects |
| `has_code_blocks` | Triple-backtick code blocks |
| `code_block_count` | Number of code blocks |
| `code_block_languages` | Comma-separated detected languages |
| `has_urls` | URL pattern detection |
| `has_emails` | Email pattern detection (disabled by default) |
| `format_complexity_score` | Weighted 0-1 score |

**Limitations**:
1. **No table detection** - Markdown/HTML tables not detected
2. **No LaTeX/math detection** - Common in technical datasets
3. **No HTML/XML detection** - Only markdown-style formatting
4. **No YAML/TOML detection** - Config formats not identified
5. **Hardcoded complexity weights** - Not configurable
6. **Basic language detection** - Only extracts from ``` prefix
7. **URL pattern simplistic** - May have false positives/negatives
8. **No nested structure analysis** - Doesn't track nesting depth

---

### 4. Quality Analyzer (`quality`)

**Purpose**: Detects data quality and safety issues.

**Implementation**:
- Regex-based PII detection with multiple patterns
- Mojibake pattern detection for encoding issues
- N-gram based repetition detection
- Configurable special token patterns
- Composite quality scoring with penalties

**Metrics Produced**:
| Metric | Description |
|--------|-------------|
| `has_pii` | Any PII detected |
| `pii_types` | Types found (email, phone, ssn, credit_card, ip, api_key) |
| `pii_count` | Total PII instances |
| `detected_language` | ISO 639-1 code (requires langdetect) |
| `language_confidence` | Detection confidence 0-1 |
| `has_encoding_issues` | Mojibake/control characters |
| `has_special_tokens` | Leaked LLM tokens |
| `repetition_ratio` | Repeated n-gram ratio |
| `has_high_repetition` | Above threshold |
| `quality_score` | Composite 0-1 score |

**Limitations**:
1. **PII detection is regex-only** - No NER for names, addresses, etc.
2. **Phone number false positives** - Pattern may match non-phone numbers
3. **No Luhn validation** - Credit card patterns not validated
4. **IP address ranges unchecked** - Doesn't validate 0-255 ranges
5. **SSN detection too broad** - May match valid non-SSN numbers
6. **API key detection heuristic** - High false positive rate
7. **Language detection optional** - Requires langdetect package
8. **Linear quality scoring** - Simple penalty deduction, not sophisticated
9. **No toxicity detection** - Would require ML model
10. **No factual accuracy checking** - Out of scope for regex

---

### 5. Embedding Analyzer (`embedding`)

**Purpose**: Semantic analysis using dense vector embeddings.

**Implementation**:
- Lazy-loads sentence-transformers model
- Batch processing with configurable batch size
- Cosine similarity for duplicate detection
- DBSCAN or K-means clustering
- Chunk-based processing for memory management

**Metrics Produced**:
| Metric | Description |
|--------|-------------|
| `duplicate_group` | Group ID for semantic duplicates |
| `has_semantic_duplicate` | Boolean duplicate flag |
| `cluster` | Cluster label (if clustering enabled) |
| `embedding` | Raw embedding vector (if store_embeddings=True) |

**Limitations**:
1. **Memory intensive** - Full similarity matrix problematic for large datasets
2. **No approximate nearest neighbor** - Should use FAISS/Annoy for scale
3. **Hardcoded chunk size** - 1000 samples, not configurable
4. **No dimensionality reduction** - No t-SNE/UMAP for visualization
5. **Embedding storage inefficient** - Lists per row, should use numpy arrays
6. **Slow for >10k samples** - Warning logged but no optimization
7. **Single threshold for duplicates** - No soft duplicate detection
8. **No cross-batch duplicate detection** - Chunks processed independently
9. **Model downloading required** - No offline support
10. **GPU utilization unclear** - Device parameter exists but not optimized

---

### 6. LLM Judge Analyzer (`llm_judge`)

**Purpose**: LLM-based evaluation using natural language prompts.

**Implementation**:
- Uses Oumi inference engine (supports OpenAI, Anthropic, vLLM, etc.)
- Configurable prompt templates
- JSON response parsing with fallbacks
- Response caching via text hash
- Batch processing with cache reuse

**Metrics Produced**:
| Metric | Description |
|--------|-------------|
| `score` | Numeric score (parsed from JSON, default 0-10) |
| `label` | Category assigned by LLM |
| `reasoning` | LLM explanation |
| `raw_response` | Full unprocessed response |

**Limitations**:
1. **API costs** - Expensive at scale with remote APIs
2. **Non-deterministic** - LLM responses vary even at low temperature
3. **High latency** - Network calls to remote APIs
4. **Text truncation** - 4000 char limit loses context
5. **JSON parsing fragile** - Regex fallbacks may fail
6. **Hash collision risk** - Python hash() not collision-free
7. **No retry logic** - Single attempt per sample
8. **No rate limiting** - May hit API limits on large batches
9. **Score range unvalidated** - Could be >10 or <0
10. **No multi-aspect evaluation** - Single prompt, single score
11. **No calibration** - Scores not normalized across models
12. **Silent failure** - Returns defaults on LLM errors

---

## Supporting Components

### Presets (`presets.py`)

Available preset: `sft_quality` - combines Length, Diversity, Format, Quality analyzers.

**Limitations**:
- Only one preset available
- No domain-specific presets (code, medical, legal)
- Token counting disabled by default

### Health Score (`health_score.py`)

Computes 0-100 dataset health score with A-F grades.

**Dimensions**: Diversity (20%), Balance (15%), Quality (25%), Consistency (20%), Length Distribution (20%)

**Limitations**:
- Hardcoded baseline scores
- Column detection by naming convention
- No customizable dimension weights
- Grade boundaries not configurable

### Recommendations (`recommendations.py`)

Generates actionable suggestions with severity levels.

**Limitations**:
- Hardcoded thresholds
- No parallel processing
- Limited to 20 sample indices per recommendation
- No machine learning-based anomaly detection

### Report Generator (`report_generator.py`)

Creates interactive HTML reports with Plotly visualizations.

**Limitations**:
- Requires Plotly CDN (no offline support)
- Limited to 10 charts by default
- Column detection by naming convention
- No PDF export option

---

## Key Improvement Opportunities

### High Priority
1. **Approximate nearest neighbor for embeddings** - Use FAISS for scale
2. **Named entity recognition for PII** - Detect names, addresses
3. **Better sentence tokenization** - Use NLTK or spaCy
4. **Offline report generation** - Bundle Plotly assets

### Medium Priority
5. **More presets** - Code, multilingual, scientific domains
6. **Configurable health score weights** - Per-use-case tuning
7. **Toxicity detection** - Integrate classifier model
8. **Table/LaTeX detection** - Expand format analyzer
9. **Cross-sample diversity** - Dataset-level uniqueness

### Lower Priority
10. **Retry logic for LLM judge** - Exponential backoff
11. **Rate limiting** - Respect API quotas
12. **PDF report export** - Alternative to HTML
13. **Streaming analysis** - For very large datasets
