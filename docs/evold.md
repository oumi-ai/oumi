# DEITA-Style Data Quality Analysis in Oumi

This document describes the DEITA (Data-Efficient Instruction Tuning for Alignment) methods for data quality analysis and how they are implemented in the Oumi analyze framework.

## Background

### The Problem

Traditional instruction tuning uses large datasets (100K+ samples), but research shows that data quality matters more than quantity. The challenge is: **how do we automatically identify high-quality samples?**

Simple heuristics (like text length or perplexity) fail to capture semantic quality. Direct LLM scoring ("rate this 1-10") produces coarse, poorly-calibrated scores.

### The DEITA Solution

The DEITA paper ("What Makes Good Data for Alignment?", Liu et al., 2023) demonstrates that **6,000 carefully selected samples can match or exceed models trained on 100K+ samples**. They achieve this through three complementary quality dimensions:

| Dimension | What it Measures | Why it Matters |
|-----------|------------------|----------------|
| **Complexity** | Instruction sophistication | Ensures model learns challenging tasks |
| **Quality** | Response helpfulness/accuracy | Prevents learning from poor examples |
| **Diversity** | Sample uniqueness | Prevents overfitting to narrow domains |

---

## Method 1: Evol Complexity Scoring

### Concept

Instruction complexity measures how sophisticated/challenging a task is. A complex instruction requires:

- Multi-step reasoning
- Handling constraints and edge cases
- Domain-specific knowledge
- Precise, unambiguous specification

### The Evol-Instruct Approach

**Key Insight**: Humans and LLMs are better at *comparative judgment* than *absolute scoring*.

Instead of asking "rate this instruction's complexity 1-10" (which produces coarse scores), DEITA:

1. **Evolves** the instruction into progressively more complex versions
2. **Ranks** the original among the evolved variants
3. **Normalizes** the rank position to a 0-1 score

```
Original: "Write a function to sort a list"

Evolved variants:
1. "Write a function to sort a list of integers in ascending order"
2. "Write a function to sort a list using quicksort, handling duplicates efficiently"
3. "Write a function to sort a list using a stable sorting algorithm with O(n log n)
    time complexity, handling edge cases like empty lists and single elements"

Ranking: [Original=1, V1=2, V2=3, V3=4]
Complexity Score: 0.0 (original is simplest)
```

### Evolution Operators

The following operators increase instruction complexity:

| Operator | Description | Example |
|----------|-------------|---------|
| `add_constraints` | Add requirements/restrictions | "...without using built-in sort" |
| `require_reasoning` | Require multi-step explanation | "...and explain your approach" |
| `increase_depth` | Request more detailed analysis | "...with time/space complexity analysis" |
| `add_edge_cases` | Handle exceptional scenarios | "...handling empty input and duplicates" |
| `require_specificity` | Make task more precise | "...using merge sort algorithm" |
| `add_domain_knowledge` | Require expertise | "...following SOLID principles" |

### Our Implementation

```python
from oumi.core.analyze import EvolComplexityAnalyzer

analyzer = EvolComplexityAnalyzer(
    # Model configuration
    model_type="api",
    api_provider="anthropic",
    api_model="claude-4-5-haiku",

    # Evolution configuration
    num_evolutions=3,  # Generate 3 more complex variants
    evolution_operators=["add_constraints", "require_reasoning", "increase_depth"],

    # Which messages to analyze
    analyze_role="user",  # Only analyze user instructions
)

# Output columns:
# - {col}_evol_complexity_score: 0-1 normalized score
# - {col}_evol_complexity_rank: Raw rank (1=simplest)
# - {col}_evol_complexity_headroom: Potential for more complexity (1-score)
```

**Interpretation**:

- Score ≈ 0: Simple instruction (lots of room for complexity evolution)
- Score ≈ 1: Already very complex (comparable to evolved versions)

---

## Method 2: Evol Quality Scoring

### Concept

Response quality measures how helpful, accurate, and complete an answer is. A high-quality response:

- Directly addresses the question
- Provides accurate information
- Has appropriate depth and structure
- Is clear and well-organized

### The Evol-Quality Approach

Similar to complexity scoring, but evolves *responses* to be progressively better:

1. **Evolves** the response into improved versions
2. **Ranks** the original among the improved variants
3. **Normalizes** the rank position to a 0-1 score

```
Instruction: "Explain recursion"

Original Response: "Recursion is when a function calls itself."

Evolved variants:
1. "Recursion is when a function calls itself. It needs a base case to stop."
2. "Recursion is a programming technique where a function calls itself to solve
    smaller instances of a problem. Every recursive function needs: (1) a base
    case that stops recursion, (2) a recursive case that breaks down the problem."
3. [Even more detailed with examples, time complexity, common pitfalls...]

Ranking: [Original=1, V1=2, V2=3, V3=4]
Quality Score: 0.0 (original is lowest quality)
```

### Quality Aspects

The following aspects are improved during evolution:

| Aspect | Description |
|--------|-------------|
| `helpfulness` | How useful and actionable the response is |
| `depth` | Level of detail and thoroughness |
| `accuracy` | Correctness of information |
| `structure` | Organization and formatting |
| `clarity` | How easy to understand |
| `completeness` | Coverage of all relevant aspects |

### Our Implementation

```python
from oumi.core.analyze import EvolQualityAnalyzer

analyzer = EvolQualityAnalyzer(
    # Model configuration
    model_type="api",
    api_provider="anthropic",
    api_model="claude-4-5-haiku",

    # Evolution configuration
    num_evolutions=3,
    quality_aspects=["helpfulness", "depth", "accuracy", "structure"],

    # Context configuration
    instruction_column="instruction",  # Column with the question being answered
    use_conversation_context=True,     # Auto-detect instruction from conversation

    # Which messages to analyze
    analyze_role="assistant",  # Only analyze assistant responses
)

# Output columns:
# - {col}_evol_quality_score: 0-1 normalized score
# - {col}_evol_quality_rank: Raw rank (1=lowest quality)
# - {col}_evol_quality_improvement_potential: How much better it could be (1-score)
```

**Interpretation**:

- Score ≈ 0: Low quality response (much room for improvement)
- Score ≈ 1: High quality response (comparable to best evolved versions)

---

## Method 3: Representation Diversity Scoring

### Concept

Diversity ensures the dataset covers a broad range of topics and problem types. Redundant samples (semantically similar to others) add little value and can cause overfitting.

### The Repr Filter Approach

DEITA uses embedding-based nearest-neighbor distances to measure diversity:

1. **Embed** all samples using a language model
2. **Compute** distance to K nearest neighbors for each sample
3. **Score** samples by their mean neighbor distance

```
Sample A: "Write a Python function to sort a list"
Sample B: "Create a Python method for sorting arrays"
Sample C: "Explain quantum computing basics"

Embeddings → Cosine distances:
- A's nearest neighbor: B (distance=0.05)  → Low diversity
- B's nearest neighbor: A (distance=0.05)  → Low diversity
- C's nearest neighbor: A (distance=0.85)  → High diversity
```

Samples with **larger distances** to their neighbors are more unique and contribute more diversity.

### Difference from Duplicate Detection

| EmbeddingAnalyzer | ReprDiversityAnalyzer |
|-------------------|----------------------|
| Finds duplicate pairs | Scores individual sample uniqueness |
| Binary: is/isn't duplicate | Continuous diversity score |
| "Are A and B similar?" | "How unique is A in this dataset?" |

### Our Implementation

```python
from oumi.core.analyze import ReprDiversityAnalyzer

analyzer = ReprDiversityAnalyzer(
    # Embedding configuration
    model_name="sentence-transformers/all-MiniLM-L6-v2",

    # Diversity scoring
    k_neighbors=5,           # Consider 5 nearest neighbors
    diversity_threshold=0.3, # Flag samples below this as redundant

    # What to embed
    embed_field="all",  # "all", "user", or "assistant"
)

# Output columns:
# - {col}_repr_diversity_nn_distance: Distance to nearest neighbor
# - {col}_repr_diversity_score: Mean distance to K nearest neighbors
# - {col}_repr_diversity_is_redundant: True if score < threshold
# - {col}_repr_diversity_percentile: Percentile rank in dataset
```

**Interpretation**:

- Score ≈ 0: Highly similar to other samples (redundant)
- Score ≈ 1: Unique, unlike other samples (diverse)

---

## Combined DEITA Score

The DEITA paper combines complexity and quality through multiplication:

```
DEITA_score = complexity_score × quality_score
```

**Why multiplication?**

- Ensures both dimensions are reasonably high
- A brilliant but low-quality response (0.9 × 0.1 = 0.09) ranks below a decent, quality response (0.5 × 0.5 = 0.25)
- Prevents gaming by optimizing only one dimension

### Score-First, Diversity-Aware Selection

For data selection, DEITA uses a greedy algorithm:

```python
# Pseudocode for DEITA selection
def select_samples(candidates, target_count, diversity_threshold):
    # Sort by combined score (descending)
    sorted_candidates = sort_by(complexity * quality, descending=True)

    selected = []
    for candidate in sorted_candidates:
        # Check diversity against already-selected samples
        if min_distance(candidate.embedding, selected) > diversity_threshold:
            selected.append(candidate)

        if len(selected) >= target_count:
            break

    return selected
```

This ensures high-scoring samples are selected while maintaining diversity.

---

## Usage Example

### Configuration File

```yaml
# configs/examples/analyze/analyze_deita.yaml
dataset_name: "tatsu-lab/alpaca"
split: "train"
sample_count: 1000

analyzers:
  # Diversity scoring
  - id: "repr_diversity"
    params:
      model_name: "sentence-transformers/all-MiniLM-L6-v2"
      k_neighbors: 5
      diversity_threshold: 0.3

  # Complexity scoring
  - id: "evol_complexity"
    params:
      model_type: "api"
      api_provider: "anthropic"
      api_model: "claude-4-5-haiku"
      num_evolutions: 3
      analyze_role: "user"

  # Quality scoring
  - id: "evol_quality"
    params:
      model_type: "api"
      api_provider: "anthropic"
      api_model: "claude-4-5-haiku"
      num_evolutions: 3
      analyze_role: "assistant"

output_path: "./analysis_output/deita"
generate_report: true
```

### Running Analysis

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Run analysis
oumi analyze -c configs/examples/analyze/analyze_deita.yaml
```

### Programmatic Usage

```python
from oumi.core.analyze import (
    DatasetAnalyzer,
    ReprDiversityAnalyzer,
    EvolComplexityAnalyzer,
    EvolQualityAnalyzer,
)

# Create analyzers
analyzers = [
    ReprDiversityAnalyzer(k_neighbors=5, diversity_threshold=0.3),
    EvolComplexityAnalyzer(num_evolutions=3, analyze_role="user"),
    EvolQualityAnalyzer(num_evolutions=3, analyze_role="assistant"),
]

# Run analysis
dataset_analyzer = DatasetAnalyzer(
    dataset_name="tatsu-lab/alpaca",
    analyzers=analyzers,
)
results = dataset_analyzer.analyze()

# Access scores
df = results.message_df
complexity_scores = df["text_content_evol_complexity_score"]
quality_scores = df["text_content_evol_quality_score"]
diversity_scores = df["text_content_repr_diversity_score"]

# Compute combined DEITA score
df["deita_score"] = complexity_scores * quality_scores

# Filter high-quality, diverse samples
high_quality = df[
    (df["deita_score"] > 0.5) &
    (df["text_content_repr_diversity_is_redundant"] == False)
]
```

---

## Architecture

### Class Hierarchy

```
SampleAnalyzer (base class)
├── ReprDiversityAnalyzer
│   └── Uses sentence-transformers for embedding
│   └── Computes KNN distances via sklearn
│
└── EvolBaseAnalyzer (abstract base)
    ├── EvolComplexityAnalyzer
    │   └── Evolves instructions to be more complex
    │   └── Ranks original among evolved variants
    │
    └── EvolQualityAnalyzer
        └── Evolves responses to be higher quality
        └── Ranks original among improved variants
```

### File Structure

```
src/oumi/core/analyze/
├── repr_diversity_analyzer.py   # Embedding-based diversity scoring
├── evol_base.py                 # Shared Evol infrastructure
├── evol_complexity_analyzer.py  # Instruction complexity scoring
├── evol_quality_analyzer.py     # Response quality scoring
└── __init__.py                  # Exports all analyzers

configs/examples/analyze/
└── analyze_deita.yaml           # Example configuration

tests/unit/core/analyze/
├── test_repr_diversity_analyzer.py
├── test_evol_complexity_analyzer.py
└── test_evol_quality_analyzer.py
```

### Model Support

The Evol analyzers support multiple inference backends:

| Backend | Configuration | Use Case |
|---------|---------------|----------|
| Anthropic API | `api_provider="anthropic"` | Default, cost-effective |
| OpenAI API | `api_provider="openai"` | Alternative API |
| Local (vLLM) | `model_type="local", engine="vllm"` | Self-hosted, no API costs |
| Local (native) | `model_type="local", engine="native"` | Simple local inference |

---

## Performance Considerations

### Cost

Each sample requires 2 LLM calls per Evol analyzer:

1. Generate evolved variants
2. Rank original among variants

For 1000 samples with both complexity and quality analysis:

- ~4000 LLM calls total
- With Claude 3.5 Haiku: ~$0.50-1.00 (depending on text length)

### Optimization Tips

1. **Start small**: Test on 100 samples before scaling
2. **Use caching**: `cache_responses=True` avoids redundant API calls
3. **Reduce evolutions**: `num_evolutions=2` cuts cost by ~33%
4. **Use Haiku**: Claude 3.5 Haiku is 10x cheaper than Sonnet
5. **Local models**: For large-scale analysis, use local inference

### Batch Processing

```python
# Process in batches to manage memory
analyzer = EvolComplexityAnalyzer(
    batch_size=8,        # Process 8 samples per batch
    max_retries=2,       # Retry failed API calls
    cache_responses=True # Cache to avoid re-computation
)
```

---

## References

- **DEITA Paper**: "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning" (Liu et al., 2023)
  - arXiv: https://arxiv.org/abs/2312.15685
  - GitHub: https://github.com/hkust-nlp/deita

- **Evol-Instruct**: "WizardLM: Empowering Large Language Models to Follow Complex Instructions" (Xu et al., 2023)
  - The original evolution approach for instruction complexity

- **Sentence Transformers**: https://www.sbert.net/
  - Used for embedding-based diversity analysis
