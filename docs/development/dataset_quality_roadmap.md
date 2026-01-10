# Dataset Quality Validation Roadmap

> **Goal**: Provide comprehensive tooling to flag bad training samples and identify dataset completeness issues that affect LLM finetuning quality.

## Executive Summary

This roadmap outlines the implementation of dataset quality analyzers for the Oumi platform. All analyzers extend the existing `SampleAnalyzer` plugin architecture and integrate with `DatasetAnalyzer`.

### Key Principles

1. **Robustness over heuristics**: Prefer deterministic, mathematically well-defined metrics
2. **Explicit configuration**: When model/threshold choices are needed, make them explicit user configuration
3. **Sample-level + Dataset-level**: Support both individual sample flagging and aggregate distribution analysis
4. **Actionable output**: Every metric should lead to a clear action (remove, review, augment)

---

## Tier Classification

| Tier | Description | External Dependencies | Reproducibility |
|------|-------------|----------------------|-----------------|
| **Tier 1** | Fully deterministic, pure computation | None | 100% reproducible |
| **Tier 2** | Deterministic given fixed configuration | Minimal (regex, hash functions) | 100% given config |
| **Tier 3** | Exact metrics, configurable thresholds | Optional (textstat, nltk) | Metrics exact, flags configurable |
| **Tier 4** | Model-dependent but deterministic | Embedding models, LMs | 100% given fixed model + seed |
| **Tier 5** | Inherently approximate/subjective | LLM-as-judge | Not recommended for production |

---

## Phase 1: Core Deterministic Analyzers

**Priority**: Highest - These catch the most common issues with zero approximation.

### 1.1 ExactDuplicateAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Sample + Dataset

| Metric | Type | Description |
|--------|------|-------------|
| `content_hash` | string | SHA256 hash of normalized content |
| `is_exact_duplicate` | bool | Whether this sample has duplicates |
| `duplicate_count` | int | Number of times this exact content appears |
| `duplicate_indices` | list[int] | Indices of duplicate samples |
| `first_occurrence` | bool | Whether this is the first occurrence |

**Implementation Details**:
```python
- Normalize text (lowercase, whitespace normalization, optional unicode normalization)
- Hash using SHA256 (deterministic, collision-resistant)
- Build hash → indices mapping across dataset
- Flag all but first occurrence
```

**Estimated LOC**: ~120
**Dependencies**: None (hashlib is stdlib)

---

### 1.2 FormatValidationAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `schema_valid` | bool | Whether sample matches expected schema |
| `missing_fields` | list[str] | Required fields that are missing |
| `empty_fields` | list[str] | Fields that are empty/null |
| `type_errors` | list[str] | Fields with wrong data type |
| `validation_errors` | list[str] | All validation error messages |

**Implementation Details**:
```python
- Define schema per dataset format (oumi, alpaca, dpo, kto, pretraining)
- Validate required fields presence
- Validate field types
- Validate field constraints (non-empty, valid values)
```

**Estimated LOC**: ~150
**Dependencies**: None

---

### 1.3 EmptyContentAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `is_empty` | bool | Content is empty string |
| `is_whitespace_only` | bool | Content contains only whitespace |
| `has_meaningful_content` | bool | Has non-whitespace content |
| `empty_messages` | list[int] | Indices of empty messages (for conversations) |
| `empty_message_count` | int | Count of empty messages |

**Estimated LOC**: ~60
**Dependencies**: None

---

### 1.4 RoleSequenceAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Sample (Conversation datasets only)

| Metric | Type | Description |
|--------|------|-------------|
| `valid_role_sequence` | bool | Roles follow expected pattern |
| `starts_with_system_or_user` | bool | First message is system or user |
| `ends_with_assistant` | bool | Last message is from assistant |
| `has_alternating_roles` | bool | User/assistant alternate properly |
| `consecutive_same_role` | list[tuple] | Indices where same role repeats |
| `role_sequence` | str | Encoded sequence (e.g., "SUAUA") |
| `unknown_roles` | list[str] | Roles not in {system, user, assistant} |

**Estimated LOC**: ~100
**Dependencies**: None

---

### 1.5 EncodingAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `valid_utf8` | bool | Content is valid UTF-8 |
| `has_replacement_chars` | bool | Contains U+FFFD replacement characters |
| `has_null_bytes` | bool | Contains null bytes |
| `control_char_count` | int | Count of control characters |
| `has_bom` | bool | Contains byte order mark |
| `encoding_issues` | list[str] | Description of encoding problems |

**Estimated LOC**: ~80
**Dependencies**: None

---

### 1.6 StatisticalOutlierAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Sample + Dataset

Operates on existing numeric columns from other analyzers (e.g., LengthAnalyzer).

| Metric | Type | Description |
|--------|------|-------------|
| `{column}_zscore` | float | Z-score for each numeric column |
| `{column}_percentile` | float | Percentile rank (0-100) |
| `{column}_iqr_outlier` | bool | Outside 1.5*IQR range |
| `{column}_extreme_outlier` | bool | Outside 3*IQR range |
| `outlier_flags` | list[str] | List of columns where sample is outlier |
| `outlier_score` | float | Aggregate outlier score |

**Estimated LOC**: ~150
**Dependencies**: None (numpy already in deps)

---

### 1.7 DatasetNgramAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Dataset + Sample

| Metric | Type | Description |
|--------|------|-------------|
| **Dataset-level** | | |
| `unique_ngram_count` | int | Total unique n-grams in dataset |
| `ngram_entropy` | float | Shannon entropy of n-gram distribution |
| `ngram_gini` | float | Gini coefficient (0=uniform, 1=concentrated) |
| `top_k_ngrams` | list[dict] | Most frequent n-grams with counts |
| `overrepresented_ngrams` | list[dict] | N-grams above doc frequency threshold |
| **Sample-level** | | |
| `contains_overrepresented` | bool | Sample contains flagged n-grams |
| `overrepresented_count` | int | Count of flagged n-grams in sample |
| `overrepresented_matches` | list[str] | Which flagged n-grams appear |
| `sample_ngram_uniqueness` | float | % of sample's n-grams that are rare |

**Configuration**:
```python
DatasetNgramAnalyzerParams(
    n_values: list[int] = [2, 3, 4],  # Which n-gram sizes to analyze
    min_document_frequency: float = 0.05,  # Flag if in >5% of samples
    top_k: int = 100,  # Report top K n-grams
    analyze_by_role: bool = True,  # Separate analysis for user/assistant
    exclude_stopwords: bool = True,  # Exclude common stopwords
    case_sensitive: bool = False,  # Case-insensitive by default
)
```

**Estimated LOC**: ~250
**Dependencies**: None (collections.Counter)

---

### 1.8 CategoryDistributionAnalyzer

**Tier**: 1 (Fully Deterministic)
**Level**: Dataset

Requires explicit category/label column in dataset.

| Metric | Type | Description |
|--------|------|-------------|
| `category_counts` | dict[str, int] | Count per category |
| `category_percentages` | dict[str, float] | Percentage per category |
| `category_entropy` | float | Distribution evenness |
| `category_gini` | float | Concentration measure |
| `num_categories` | int | Total unique categories |
| `underrepresented` | list[str] | Categories below threshold |
| `overrepresented` | list[str] | Categories above threshold |
| `expected_missing` | list[str] | Expected categories with 0 samples |

**Configuration**:
```python
CategoryDistributionAnalyzerParams(
    category_column: str,  # Column containing category labels
    expected_categories: list[str] = None,  # Optional: expected categories
    min_percentage: float = 0.01,  # Flag if <1%
    max_percentage: float = 0.50,  # Flag if >50%
)
```

**Estimated LOC**: ~100
**Dependencies**: None

---

## Phase 2: Pattern-Based Analyzers

**Priority**: High - Deterministic given explicit configuration.

### 2.1 RepetitionAnalyzer

**Tier**: 2 (Deterministic given n)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `char_repetition_ratio` | float | Ratio of repeated character sequences |
| `word_repetition_ratio` | float | Ratio of repeated words |
| `ngram_repetition_ratio` | dict[int, float] | Repetition ratio per n-gram size |
| `longest_repeated_sequence` | str | Longest repeated substring |
| `longest_repeated_length` | int | Length of longest repeat |
| `unique_word_ratio` | float | Unique words / total words |
| `has_excessive_repetition` | bool | Exceeds configured threshold |

**Estimated LOC**: ~120
**Dependencies**: None

---

### 2.2 NearDuplicateAnalyzer

**Tier**: 2 (Deterministic given hash functions)
**Level**: Sample + Dataset

Uses MinHash/LSH for efficient near-duplicate detection.

| Metric | Type | Description |
|--------|------|-------------|
| `minhash_signature` | bytes | MinHash signature for sample |
| `near_duplicate_cluster_id` | int | Cluster ID for near-duplicates |
| `jaccard_similarity_max` | float | Max Jaccard similarity to any other sample |
| `near_duplicate_indices` | list[int] | Indices of near-duplicate samples |
| `is_near_duplicate` | bool | Has near-duplicates above threshold |
| `cluster_size` | int | Size of near-duplicate cluster |

**Configuration**:
```python
NearDuplicateAnalyzerParams(
    num_perm: int = 128,  # Number of permutations for MinHash
    similarity_threshold: float = 0.8,  # Jaccard threshold
    ngram_size: int = 5,  # Shingle size
)
```

**Estimated LOC**: ~200
**Dependencies**: `datasketch` (MIT license, well-maintained)

---

### 2.3 RequestTypePatternAnalyzer

**Tier**: 2 (Deterministic given patterns)
**Level**: Sample + Dataset

Classifies requests based on user-defined keyword patterns.

| Metric | Type | Description |
|--------|------|-------------|
| **Sample-level** | | |
| `request_type` | str | Classified request type |
| `request_type_confidence` | str | "exact" or "partial" match |
| `matched_patterns` | list[str] | Which patterns matched |
| **Dataset-level** | | |
| `type_distribution` | dict[str, int] | Count per request type |
| `type_percentages` | dict[str, float] | Percentage per type |
| `type_entropy` | float | Distribution evenness |
| `missing_types` | list[str] | Defined types with 0 matches |
| `underrepresented_types` | list[str] | Types below threshold |
| `unknown_count` | int | Samples matching no pattern |
| `unknown_percentage` | float | % not matching any pattern |

**Configuration**:
```python
RequestTypePatternAnalyzerParams(
    patterns: dict[str, list[str]] = {
        "explanation": ["explain", "what is", "how does", "why is", "describe"],
        "code_generation": ["write code", "implement", "create a function", "code for"],
        "summarization": ["summarize", "summary", "tldr", "brief", "shorten"],
        "debugging": ["fix", "debug", "error", "bug", "not working", "issue"],
        "translation": ["translate", "convert to", "in spanish", "in french"],
        "creative_writing": ["write a story", "poem", "creative", "imagine"],
        "analysis": ["analyze", "review", "evaluate", "assess", "critique"],
        "comparison": ["compare", "difference between", "vs", "versus"],
        "how_to": ["how to", "how do i", "how can i", "steps to"],
        "opinion": ["what do you think", "your opinion", "recommend"],
    },
    case_sensitive: bool = False,
    match_mode: str = "first",  # "first", "all", or "weighted"
    min_type_percentage: float = 0.01,  # Flag types below 1%
    apply_to_column: str = "text_content",  # Or "prompt" for DPO
    apply_to_role: str = "user",  # Only classify user messages
)
```

**Estimated LOC**: ~180
**Dependencies**: None (regex)

---

### 2.4 PerplexityAnalyzer

**Tier**: 2 (Deterministic given fixed model)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `perplexity` | float | Perplexity score |
| `log_likelihood` | float | Log likelihood of text |
| `tokens_evaluated` | int | Number of tokens scored |
| `perplexity_percentile` | float | Percentile within dataset |
| `is_high_perplexity` | bool | Above threshold (potential noise) |
| `is_low_perplexity` | bool | Below threshold (potential duplicate/template) |

**Configuration**:
```python
PerplexityAnalyzerParams(
    model_name: str = "gpt2",  # Reference model
    batch_size: int = 8,
    max_length: int = 512,  # Truncate for efficiency
    high_percentile: float = 95.0,  # Flag above 95th percentile
    low_percentile: float = 5.0,  # Flag below 5th percentile
    device: str = "auto",
)
```

**Estimated LOC**: ~180
**Dependencies**: `transformers` (already in deps)

---

### 2.5 VocabularyAnalyzer

**Tier**: 2 (Deterministic)
**Level**: Sample + Dataset

| Metric | Type | Description |
|--------|------|-------------|
| **Sample-level** | | |
| `vocabulary_size` | int | Unique words in sample |
| `type_token_ratio` | float | Unique words / total words |
| `hapax_legomena_count` | int | Words appearing exactly once |
| `hapax_ratio` | float | Hapax / total words |
| **Dataset-level** | | |
| `dataset_vocabulary_size` | int | Unique words across dataset |
| `dataset_ttr` | float | Dataset-level type-token ratio |
| `vocabulary_growth_rate` | float | Rate of new words per sample |
| `rare_words` | list[str] | Words appearing in <1% of samples |
| `common_words` | list[str] | Words appearing in >50% of samples |

**Estimated LOC**: ~130
**Dependencies**: None

---

### 2.6 ReadabilityAnalyzer

**Tier**: 2 (Deterministic - standard formulas)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `flesch_reading_ease` | float | Flesch Reading Ease (0-100) |
| `flesch_kincaid_grade` | float | Flesch-Kincaid Grade Level |
| `gunning_fog_index` | float | Gunning Fog Index |
| `smog_index` | float | SMOG Index |
| `automated_readability_index` | float | ARI |
| `avg_sentence_length` | float | Words per sentence |
| `avg_syllables_per_word` | float | Syllables per word |
| `complex_word_ratio` | float | % words with 3+ syllables |

**Estimated LOC**: ~150
**Dependencies**: Optional `textstat` (or implement formulas directly)

---

### 2.7 ActionVerbAnalyzer

**Tier**: 2 (Deterministic given POS tagger)
**Level**: Sample + Dataset

Extracts and analyzes imperative verbs from user requests.

| Metric | Type | Description |
|--------|------|-------------|
| **Sample-level** | | |
| `primary_action_verb` | str | Main imperative verb |
| `all_action_verbs` | list[str] | All verbs in request |
| `verb_count` | int | Number of action verbs |
| **Dataset-level** | | |
| `verb_distribution` | dict[str, int] | Count per verb |
| `verb_entropy` | float | Diversity of actions |
| `top_k_verbs` | list[tuple] | Most common verbs |
| `rare_verbs` | list[str] | Verbs in <1% of samples |
| `missing_common_verbs` | list[str] | Expected verbs with 0 occurrences |

**Configuration**:
```python
ActionVerbAnalyzerParams(
    pos_tagger: str = "nltk",  # or "spacy"
    expected_verbs: list[str] = [
        "explain", "write", "create", "fix", "debug", "summarize",
        "translate", "analyze", "compare", "describe", "implement",
        "generate", "find", "solve", "help", "show", "tell", "list"
    ],
    apply_to_role: str = "user",
)
```

**Estimated LOC**: ~200
**Dependencies**: `nltk` (already in evaluation deps) or `spacy`

---

## Phase 3: Threshold-Based Quality Metrics

**Priority**: Medium - Metrics are exact, flagging requires user-defined thresholds.

### 3.1 ConversationStructureAnalyzer

**Tier**: 3 (Metrics exact, thresholds configurable)
**Level**: Sample (Conversation datasets)

| Metric | Type | Description |
|--------|------|-------------|
| `turn_count` | int | Number of conversation turns |
| `user_turn_count` | int | Number of user messages |
| `assistant_turn_count` | int | Number of assistant messages |
| `system_message_count` | int | Number of system messages |
| `avg_turn_length_chars` | float | Average characters per turn |
| `avg_turn_length_words` | float | Average words per turn |
| `turn_length_variance` | float | Variance in turn lengths |
| `user_assistant_ratio` | float | User turns / assistant turns |
| `longest_turn_length` | int | Maximum turn length |
| `shortest_turn_length` | int | Minimum turn length |
| `is_single_turn` | bool | Only one exchange |
| `is_too_short` | bool | Below minimum turn threshold |
| `is_too_long` | bool | Above maximum turn threshold |

**Estimated LOC**: ~120
**Dependencies**: None

---

### 3.2 TokenBudgetAnalyzer

**Tier**: 3 (Metrics exact, thresholds configurable)
**Level**: Sample

Analyzes token usage relative to model context limits.

| Metric | Type | Description |
|--------|------|-------------|
| `total_tokens` | int | Total tokens in sample |
| `input_tokens` | int | Tokens in user/system messages |
| `output_tokens` | int | Tokens in assistant messages |
| `input_output_ratio` | float | Input / output token ratio |
| `context_utilization` | float | % of max context used |
| `exceeds_context` | bool | Exceeds model context length |
| `truncation_required` | bool | Would require truncation |
| `tokens_over_limit` | int | How many tokens over limit |

**Configuration**:
```python
TokenBudgetAnalyzerParams(
    max_context_length: int = 4096,
    target_input_ratio: float = 0.3,  # Ideal input/total ratio
    warn_threshold: float = 0.9,  # Warn if >90% of context
)
```

**Estimated LOC**: ~100
**Dependencies**: Tokenizer (already available)

---

## Phase 4: Preference Data Analyzers (DPO/RLHF)

**Priority**: High for preference tuning workflows.

### 4.1 DPOContrastivenessAnalyzer

**Tier**: 2 (Deterministic) for text metrics, Tier 4 for embedding-based
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| **Text-based (Tier 2)** | | |
| `length_difference` | int | \|len(chosen) - len(rejected)\| |
| `word_overlap_ratio` | float | Jaccard similarity of word sets |
| `edit_distance` | int | Levenshtein distance |
| `edit_distance_normalized` | float | Edit distance / max length |
| `chosen_length` | int | Length of chosen response |
| `rejected_length` | int | Length of rejected response |
| **Embedding-based (Tier 4)** | | |
| `embedding_cosine_similarity` | float | Cosine similarity of embeddings |
| `embedding_euclidean_distance` | float | Euclidean distance |
| **Flags** | | |
| `low_contrastiveness` | bool | Pair too similar |
| `length_only_difference` | bool | Only difference is length |
| `potential_label_flip` | bool | Rejected might be better |

**Configuration**:
```python
DPOContrastivenessAnalyzerParams(
    min_edit_distance: int = 10,
    max_word_overlap: float = 0.9,
    use_embeddings: bool = False,  # Enable Tier 4 metrics
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
)
```

**Estimated LOC**: ~200 (text), ~300 (with embeddings)
**Dependencies**: Optional `sentence-transformers`

---

### 4.2 DPOFormatValidator

**Tier**: 1 (Fully Deterministic)
**Level**: Sample

| Metric | Type | Description |
|--------|------|-------------|
| `has_prompt` | bool | Prompt field exists and non-empty |
| `has_chosen` | bool | Chosen field exists and non-empty |
| `has_rejected` | bool | Rejected field exists and non-empty |
| `prompt_in_chosen` | bool | Prompt text appears in chosen |
| `prompt_in_rejected` | bool | Prompt text appears in rejected |
| `chosen_equals_rejected` | bool | Chosen and rejected are identical |
| `all_fields_valid` | bool | All validation checks pass |

**Estimated LOC**: ~80
**Dependencies**: None

---

## Phase 5: Embedding-Based Analyzers

**Priority**: Lower - Requires embedding model, but adds semantic analysis.

### 5.1 EmbeddingOutlierAnalyzer

**Tier**: 4 (Model-dependent)
**Level**: Sample + Dataset

| Metric | Type | Description |
|--------|------|-------------|
| `knn_distance_mean` | float | Mean distance to K nearest neighbors |
| `knn_distance_max` | float | Max distance to K nearest neighbors |
| `local_outlier_factor` | float | LOF score |
| `isolation_score` | float | Isolation Forest score |
| `distance_to_centroid` | float | Distance to dataset centroid |
| `is_outlier` | bool | Flagged as outlier by any method |
| `outlier_methods` | list[str] | Which methods flagged it |

**Configuration**:
```python
EmbeddingOutlierAnalyzerParams(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k_neighbors: int = 10,
    outlier_methods: list[str] = ["knn", "lof", "isolation_forest"],
    contamination: float = 0.05,  # Expected outlier fraction
    batch_size: int = 32,
)
```

**Estimated LOC**: ~350
**Dependencies**: `sentence-transformers`, `scikit-learn`

---

### 5.2 SemanticDuplicateAnalyzer

**Tier**: 4 (Model-dependent)
**Level**: Sample + Dataset

| Metric | Type | Description |
|--------|------|-------------|
| `semantic_cluster_id` | int | Cluster assignment |
| `cluster_size` | int | Number of samples in cluster |
| `similarity_to_nearest` | float | Cosine similarity to most similar sample |
| `nearest_neighbor_index` | int | Index of most similar sample |
| `is_semantic_duplicate` | bool | Above similarity threshold |
| `semantic_duplicate_count` | int | Number of semantic duplicates |

**Configuration**:
```python
SemanticDuplicateAnalyzerParams(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.95,
    clustering_algorithm: str = "agglomerative",  # or "hdbscan"
    batch_size: int = 32,
)
```

**Estimated LOC**: ~300
**Dependencies**: `sentence-transformers`, `scikit-learn`

---

### 5.3 RequestClusteringAnalyzer

**Tier**: 4 (Model-dependent)
**Level**: Dataset

Discovers request types via clustering (when explicit labels unavailable).

| Metric | Type | Description |
|--------|------|-------------|
| `cluster_id` | int | Assigned cluster |
| `cluster_size` | int | Samples in cluster |
| `cluster_label` | str | Auto-generated cluster label (top keywords) |
| `distance_to_cluster_center` | float | Distance from centroid |
| `cluster_distribution` | dict[int, int] | Size of each cluster |
| `cluster_entropy` | float | Evenness of cluster sizes |
| `num_clusters` | int | Total clusters discovered |
| `small_clusters` | list[int] | Clusters with <1% of samples |
| `large_clusters` | list[int] | Clusters with >20% of samples |

**Configuration**:
```python
RequestClusteringAnalyzerParams(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    clustering_algorithm: str = "hdbscan",  # or "kmeans"
    min_cluster_size: int = 10,
    num_clusters: int = None,  # For kmeans, auto for hdbscan
    generate_labels: bool = True,  # Extract keywords per cluster
    random_seed: int = 42,  # For reproducibility
)
```

**Estimated LOC**: ~400
**Dependencies**: `sentence-transformers`, `scikit-learn`, `hdbscan`

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

| Analyzer | Tier | Priority | LOC |
|----------|------|----------|-----|
| ExactDuplicateAnalyzer | 1 | P0 | 120 |
| FormatValidationAnalyzer | 1 | P0 | 150 |
| EmptyContentAnalyzer | 1 | P0 | 60 |
| RoleSequenceAnalyzer | 1 | P0 | 100 |
| EncodingAnalyzer | 1 | P0 | 80 |

**Total**: ~510 LOC
**Dependencies**: None
**Deliverable**: Core validation catching structural issues

---

### Phase 2: Statistical Analysis (Weeks 3-4)

| Analyzer | Tier | Priority | LOC |
|----------|------|----------|-----|
| StatisticalOutlierAnalyzer | 1 | P0 | 150 |
| DatasetNgramAnalyzer | 1 | P0 | 250 |
| RepetitionAnalyzer | 2 | P1 | 120 |
| VocabularyAnalyzer | 2 | P1 | 130 |

**Total**: ~650 LOC
**Dependencies**: None
**Deliverable**: Distribution analysis, n-gram frequency detection

---

### Phase 3: Pattern & Classification (Weeks 5-6)

| Analyzer | Tier | Priority | LOC |
|----------|------|----------|-----|
| RequestTypePatternAnalyzer | 2 | P0 | 180 |
| CategoryDistributionAnalyzer | 1 | P1 | 100 |
| ActionVerbAnalyzer | 2 | P2 | 200 |
| ReadabilityAnalyzer | 2 | P2 | 150 |

**Total**: ~630 LOC
**Dependencies**: Optional `nltk`, `textstat`
**Deliverable**: Request type classification, coverage analysis

---

### Phase 4: Advanced Text Analysis (Weeks 7-8)

| Analyzer | Tier | Priority | LOC |
|----------|------|----------|-----|
| NearDuplicateAnalyzer | 2 | P0 | 200 |
| PerplexityAnalyzer | 2 | P1 | 180 |
| ConversationStructureAnalyzer | 3 | P1 | 120 |
| TokenBudgetAnalyzer | 3 | P1 | 100 |

**Total**: ~600 LOC
**Dependencies**: `datasketch`, `transformers`
**Deliverable**: Near-duplicate detection, perplexity filtering

---

### Phase 5: DPO/Preference Analysis (Weeks 9-10)

| Analyzer | Tier | Priority | LOC |
|----------|------|----------|-----|
| DPOFormatValidator | 1 | P0 | 80 |
| DPOContrastivenessAnalyzer | 2/4 | P0 | 200-300 |

**Total**: ~280-380 LOC
**Dependencies**: Optional `sentence-transformers`
**Deliverable**: Preference data quality validation

---

### Phase 6: Embedding-Based (Weeks 11-12)

| Analyzer | Tier | Priority | LOC |
|----------|------|----------|-----|
| EmbeddingOutlierAnalyzer | 4 | P2 | 350 |
| SemanticDuplicateAnalyzer | 4 | P2 | 300 |
| RequestClusteringAnalyzer | 4 | P2 | 400 |

**Total**: ~1050 LOC
**Dependencies**: `sentence-transformers`, `scikit-learn`, `hdbscan`
**Deliverable**: Semantic analysis capabilities

---

## Quality Report Output

### Sample-Level Report

```python
@dataclass
class SampleQualityReport:
    sample_index: int
    flags: list[str]  # List of issue flags
    severity: str  # "critical", "warning", "info"
    metrics: dict[str, Any]  # All computed metrics
    recommended_action: str  # "remove", "review", "keep"
```

### Dataset-Level Report

```python
@dataclass
class DatasetQualityReport:
    # Overview
    total_samples: int
    analyzed_samples: int

    # Issue Summary
    critical_issues: int
    warning_issues: int
    samples_flagged: int
    flagged_percentage: float

    # By Issue Type
    issues_by_type: dict[str, int]
    # e.g., {"exact_duplicate": 1200, "empty_content": 45, ...}

    # Distribution Analysis
    ngram_analysis: NgramAnalysisResult
    request_type_distribution: dict[str, float]
    category_distribution: dict[str, float]

    # Recommendations
    recommendations: list[str]
    # e.g., ["Remove 1,200 exact duplicates",
    #        "Add more 'debugging' request types (currently 2.1%)",
    #        "Review 450 samples with overrepresented n-grams"]

    # Detailed Flags
    flagged_samples: dict[str, list[int]]
    # e.g., {"exact_duplicate": [12, 45, 89, ...], ...}
```

---

## Dependencies Summary

| Dependency | License | Required For | Phase |
|------------|---------|--------------|-------|
| None (stdlib) | - | Phase 1, 2 core | 1-2 |
| `datasketch` | MIT | NearDuplicateAnalyzer | 4 |
| `nltk` | Apache 2.0 | ActionVerbAnalyzer | 3 |
| `textstat` | MIT | ReadabilityAnalyzer | 3 |
| `sentence-transformers` | Apache 2.0 | Embedding analyzers | 5, 6 |
| `scikit-learn` | BSD | Outlier/clustering | 6 |
| `hdbscan` | BSD | RequestClusteringAnalyzer | 6 |

---

## Success Metrics

1. **Coverage**: All Tier 1 analyzers implemented with 100% test coverage
2. **Performance**: Analyze 100K samples in <5 minutes (non-embedding analyzers)
3. **Reproducibility**: Same input always produces same output (given config)
4. **Actionability**: Every flag includes recommended action
5. **Integration**: Seamless integration with existing DatasetAnalyzer

---

## References

- [Cleanlab - Data Quality for ML](https://github.com/cleanlab/cleanlab)
- [Perplexity-Based Data Pruning](https://arxiv.org/html/2405.20541v1)
- [D4: Document De-Duplication and Diversification](https://arxiv.org/pdf/2308.12284)
- [What Matters in Data for DPO](https://arxiv.org/html/2508.18312v1)
- [Data Quality in NLP: Metrics and Taxonomy](https://link.springer.com/chapter/10.1007/978-3-031-58547-0_18)
