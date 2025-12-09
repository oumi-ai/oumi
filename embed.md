# Oumi Analyze: Duplicate Detection & Clustering Investigation

## Overview

The `oumi analyze` command provides three main features for data quality analysis:
1. **Semantic Duplicate Detection** (embedding-based, cosine similarity)
2. **Fuzzy Duplicate Detection** (MinHash LSH, Jaccard similarity)
3. **Clustering** (DBSCAN or K-Means)

All three features are implemented in the `EmbeddingAnalyzer` class.

---

## 1. Semantic Duplicate Detection (Embedding-based)

**File:** `src/oumi/core/analyze/embedding_analyzer.py`

### Configuration Options

```yaml
detect_duplicates: true          # Enable semantic duplicate detection
duplicate_threshold: 0.95        # Cosine similarity threshold (0.0-1.0)
store_embeddings: true           # Store embedding vectors in output
```

### How It Works

**Step 1: Generate Embeddings**
- Uses `sentence-transformers` library (default model: `all-MiniLM-L6-v2`)
- Converts each text into a 384-dimensional vector
- Processes in batches (configurable `batch_size`, default 32)

**Step 2: Compute Cosine Similarity**
- Uses `sklearn.metrics.pairwise.cosine_similarity()`
- Computes similarity matrix between all pairs of embeddings
- Memory-efficient: processes in chunks of 1000 samples to avoid OOM
- Cosine similarity range: 0.0 (completely different) to 1.0 (identical)

**Step 3: Identify Duplicates**
```python
# Find samples with similarity >= threshold
duplicate_indices = np.where(similarity_row >= duplicate_threshold)[0]

# Exclude self (self-similarity is always 1.0)
duplicate_indices = duplicate_indices[duplicate_indices != current_index]
```

**Step 4: Group Duplicates**
- Uses union-find style grouping
- Transitive grouping: if A~B and B~C, then A, B, C get the same group ID
- All connected duplicates assigned minimum group ID

### Output Columns

For each text column analyzed, two columns are added:
- `{column}_embedding_duplicate_group` (int): Group ID for duplicate cluster
- `{column}_embedding_has_semantic_duplicate` (bool): Whether sample has duplicates

### Key Characteristics

| Property | Value |
|----------|-------|
| **Time Complexity** | O(n²) - Quadratic |
| **Use Case** | Detects paraphrases, same meaning with different wording |
| **Threshold Range** | 0.90 (aggressive) to 0.98 (conservative), 0.95 recommended |
| **Speed** | Slower, especially for large datasets |
| **Behavior** | Marks duplicates, does NOT remove them |

### When to Use
- Finding semantically similar content with different wording
- Quality control for training datasets
- Identifying paraphrased or conceptually duplicate samples

---

## 2. Fuzzy Duplicate Detection (MinHash LSH)

**File:** `src/oumi/core/analyze/embedding_analyzer.py`

### Configuration Options

```yaml
detect_fuzzy_duplicates: true    # Enable fuzzy duplicate detection
fuzzy_threshold: 0.8             # Jaccard similarity threshold (0.0-1.0)
fuzzy_ngram_size: 3              # Character n-gram size
fuzzy_num_perm: 128              # MinHash permutations (accuracy vs speed)
```

### How It Works

**Step 1: Extract Character N-grams**
```python
# For text "hello" with ngram_size=3:
# Returns: {"hel", "ell", "llo"}
text = text.lower().strip()
ngrams = {text[i:i+ngram_size] for i in range(len(text)-ngram_size+1)}
```

**Step 2: Create MinHash Signatures**
- Uses `datasketch` library
- Creates MinHash with `fuzzy_num_perm` permutations (default 128)
- Hashes each n-gram into the MinHash signature
- Result: Fixed-size signature representing the text's n-gram set

**Step 3: Build LSH Index**
- Creates `MinHashLSH` index with similarity threshold
- Inserts each MinHash signature
- LSH automatically buckets similar items together for fast retrieval

**Step 4: Find Duplicates**
```python
# Query LSH for candidates similar to each sample
candidates = lsh.query(minhash)  # Fast O(1) lookup

# Compute actual Jaccard similarity with candidates
for candidate in candidates:
    jaccard = minhash1.jaccard(minhash2)
    if jaccard >= fuzzy_threshold:
        # Mark as duplicate
```

**Step 5: Group Duplicates**
- Similar to semantic detection: union-find style grouping
- Assigns minimum group ID to connected components

### Output Columns

For each text column analyzed, three columns are added:
- `{column}_embedding_fuzzy_duplicate_group` (int): Group ID for fuzzy duplicate cluster
- `{column}_embedding_has_fuzzy_duplicate` (bool): Whether sample has fuzzy duplicates
- `{column}_embedding_fuzzy_jaccard_score` (float): Max Jaccard similarity to nearest duplicate

### Key Characteristics

| Property | Value |
|----------|-------|
| **Time Complexity** | O(n) - Linear |
| **Use Case** | Detects near-exact copies, minor edits, copy-paste variants |
| **Threshold Range** | 0.8 is typical, higher = more conservative |
| **Speed** | Very fast, even for 100k+ samples |
| **Sensitivity** | High sensitivity to character-level changes |

### Jaccard Similarity Explained
- Jaccard(A, B) = |A ∩ B| / |A ∪ B| (intersection over union of n-grams)
- Measures character-level similarity, not semantic similarity
- Example: "hello world" vs "helo world" → high Jaccard (one character difference)
- Example: "hello world" vs "greetings earth" → low Jaccard (semantically similar but different words)

### When to Use
- Deduplicating raw datasets with copy-paste duplicates
- Finding variants with typos or minor edits
- Fast preprocessing for large-scale datasets
- Complementary to semantic deduplication

---

## 3. Clustering

**File:** `src/oumi/core/analyze/embedding_analyzer.py`

### Configuration Options

```yaml
cluster_samples: true            # Enable clustering
clustering_method: dbscan        # "dbscan" or "kmeans"

# DBSCAN parameters:
eps: 0.5                         # Maximum distance between samples in cluster
min_samples: 2                   # Minimum samples to form a core point

# K-Means parameters:
# n_clusters: 5                  # Required for kmeans
```

### How It Works

**Step 1: Use Embeddings from Semantic Analyzer**
- Clustering uses the same embeddings generated for semantic duplicate detection
- 384-dimensional vectors (all-MiniLM-L6-v2 model)
- No additional embedding computation needed

**Step 2: Apply Clustering Algorithm**

#### DBSCAN (Density-Based Spatial Clustering)
```python
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=eps, min_samples=min_samples)
labels = clustering.fit_predict(embeddings)
# Returns: array of cluster IDs, where -1 = noise/outliers
```

**DBSCAN Parameters:**
- `eps`: Controls cluster density
  - Lower values → more clusters, require higher similarity
  - For embeddings: eps=0.5 ≈ 60% cosine similarity
  - For questions: eps=0.15 ≈ 99% cosine similarity (very strict)
- `min_samples`: Minimum points to form a core point
  - Default 2: even 2 similar samples form a cluster
  - Higher values require denser clusters

**DBSCAN Noise Points:**
- Samples labeled `-1` are "noise" (don't fit into any cluster)
- In QuestionDiversityAnalyzer: treated as unique/diverse samples
- Contributes positively to diversity metrics

#### K-Means (Centroid-Based Clustering)
```python
from sklearn.cluster import KMeans

clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = clustering.fit_predict(embeddings)
# Returns: array of cluster IDs (0 to n_clusters-1)
```

**K-Means Parameters:**
- `n_clusters`: **Required** - must be specified in config
- `random_state=42`: Ensures reproducible results
- `n_init=10`: Runs algorithm 10 times with different initializations

**DBSCAN vs K-Means:**
| Aspect | DBSCAN | K-Means |
|--------|--------|---------|
| **n_clusters** | Not required (auto-detected) | Required (must specify) |
| **Shape** | Any shape | Spherical clusters |
| **Outliers** | Labels as noise (-1) | Forces into nearest cluster |
| **Best For** | Finding natural groupings | Known number of topics |

### Output Columns

For each text column analyzed, one column is added:
- `{column}_embedding_cluster` (int): Cluster ID assigned to sample

For QuestionDiversityAnalyzer (specialized analyzer for questions):
- `{column}_question_diversity_cluster_id` (int): Cluster ID
- `{column}_question_diversity_cluster_size` (int): Size of cluster
- `{column}_question_diversity_is_concentrated` (bool): Whether cluster is oversized

### Dataset-Level Metrics (QuestionDiversityAnalyzer)

The QuestionDiversityAnalyzer computes additional metrics:

```python
{
    "num_question_clusters": int,        # Total number of clusters
    "num_noise_samples": int,            # DBSCAN noise points
    "diversity_ratio": float,             # noise_count / total_questions
    "question_entropy": float,            # Shannon entropy (higher = more diverse)
    "question_gini": float,               # Gini coefficient (0 = uniform, 1 = concentrated)
    "largest_cluster_ratio": float,      # % in largest cluster
    "diversity_rating": str,              # "low", "medium", or "high"
    "cluster_distribution": dict          # cluster_id -> count
}
```

**Shannon Entropy:**
- Formula: H = -sum(p * log2(p)) where p = cluster_size / total
- Higher entropy = more diverse distribution
- Maximum when all clusters have equal size

**Gini Coefficient:**
- Measures concentration inequality
- 0 = perfectly uniform distribution
- 1 = maximum inequality (all samples in one cluster)

### When to Use
- Grouping samples by topic or theme
- Analyzing dataset diversity
- Identifying overrepresented content clusters
- Understanding data distribution before training

---

## Comparison: Semantic vs Fuzzy vs Clustering

| Feature | Semantic Duplicates | Fuzzy Duplicates | Clustering |
|---------|--------------------|--------------------|-----------|
| **Method** | Cosine similarity of embeddings | Jaccard similarity of n-grams | DBSCAN or K-Means on embeddings |
| **Time** | O(n²) | O(n) | O(n log n) for DBSCAN |
| **Detects** | Paraphrases, semantic equivalence | Near-exact copies, typos | Topical groupings |
| **Output** | Duplicate groups + boolean flags | Duplicate groups + Jaccard scores | Cluster IDs |
| **Removes?** | No, marks only | No, marks only | No, assigns IDs only |
| **Use Case** | Same meaning, different words | Copy-paste with minor edits | Topic/theme grouping |

### Recommended Workflow

1. **Fast preprocessing:** Run fuzzy dedup to remove obvious copies
2. **Semantic quality:** Run semantic dedup to find paraphrased duplicates
3. **Analysis:** Run clustering to understand content distribution

Can run all three together in a single `oumi analyze` command!

---

## Example Configurations

### Embedding-based Deduplication Only
**File:** `configs/examples/analyze/analyze_dedup_emb.yaml`

```yaml
analyzers:
  - id: embedding
    params:
      model_name: all-MiniLM-L6-v2
      batch_size: 64

      # Semantic duplicates
      detect_duplicates: true
      duplicate_threshold: 0.95

      # Disable other features
      detect_fuzzy_duplicates: false
      cluster_samples: false

      store_embeddings: true
```

### Fuzzy Deduplication Only (Fast)
**File:** `configs/examples/analyze/analyze_dedup_lsh.yaml`

```yaml
analyzers:
  - id: embedding
    params:
      # Fuzzy duplicates (O(n) complexity)
      detect_fuzzy_duplicates: true
      fuzzy_threshold: 0.8
      fuzzy_ngram_size: 3
      fuzzy_num_perm: 128

      # Clustering
      cluster_samples: true
      clustering_method: dbscan
      eps: 0.5
      min_samples: 2

      store_embeddings: true
```

### All Features Combined
```yaml
analyzers:
  - id: embedding
    params:
      # Semantic duplicates
      detect_duplicates: true
      duplicate_threshold: 0.95

      # Fuzzy duplicates
      detect_fuzzy_duplicates: true
      fuzzy_threshold: 0.8
      fuzzy_ngram_size: 3
      fuzzy_num_perm: 128

      # Clustering
      cluster_samples: true
      clustering_method: dbscan
      eps: 0.5
      min_samples: 2

      store_embeddings: false  # Save memory for large datasets
```

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `src/oumi/core/analyze/embedding_analyzer.py` | Main implementation (lines 43-667) |
| `src/oumi/core/analyze/question_diversity_analyzer.py` | Specialized analyzer for question diversity |
| `src/oumi/core/configs/analyze_config.py` | Configuration dataclasses |
| `src/oumi/core/analyze/dataset_analyzer.py` | Orchestrates all analyzers |
| `configs/examples/analyze/analyze_dedup_emb.yaml` | Example: semantic dedup |
| `configs/examples/analyze/analyze_dedup_lsh.yaml` | Example: fuzzy dedup |
| `configs/examples/analyze/analyze.yaml` | Example: full analysis |

---

## Performance Considerations

### Memory Usage
- **Semantic dedup:** High memory (similarity matrix, chunked processing)
- **Fuzzy dedup:** Low memory (LSH index, linear space)
- **Clustering:** Moderate memory (scikit-learn algorithms)
- **store_embeddings:** Can significantly increase memory usage

### Speed
- **Semantic dedup:** Slowest (O(n²)), may need GPU for large datasets
- **Fuzzy dedup:** Fastest (O(n)), even for 100k+ samples
- **Clustering:** Fast (O(n log n) for DBSCAN)

### Recommendations
- Use fuzzy dedup for initial fast deduplication
- Use semantic dedup for quality control on smaller datasets
- Set `store_embeddings: false` for large datasets
- Increase `batch_size` if you have sufficient GPU memory

---

## Dependencies

```bash
# Required for all features
pip install 'oumi[analyze_advanced]'

# Installs:
# - sentence-transformers (embeddings)
# - scikit-learn (clustering, cosine similarity)
# - datasketch (MinHash LSH)
```

---

## Summary

The `oumi analyze` command provides three complementary approaches to dataset quality analysis:

1. **Semantic Duplicate Detection**: Catches paraphrases and semantically similar content
2. **Fuzzy Duplicate Detection**: Fast detection of near-exact copies and typos
3. **Clustering**: Groups samples by topic for diversity analysis

All three can be run together to get comprehensive insights into your dataset quality and distribution. The output is a DataFrame with additional columns marking duplicates, assigning clusters, and computing diversity metrics.