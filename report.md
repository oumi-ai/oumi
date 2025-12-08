# SFT Dataset Quality Analysis: Research Report

## Executive Summary

This report analyzes the current state of SFT (Supervised Fine-Tuning) dataset quality metrics, comparing what's implemented in `oumi analyze` against techniques used by frontier AI labs (Meta, DeepSeek, Alibaba, Microsoft, Allen AI, Zhipu AI) and recent academic research (2024-2025).

**Key Finding:** While oumi analyze covers basic quality checks (length, diversity, format, PII), it lacks several high-impact techniques universally adopted by frontier labs, particularly **rejection sampling support**, **LLM-as-judge scoring**, and **quality × difficulty metrics**.

---

## Part 1: Current oumi analyze Capabilities

### Implemented Analyzers

| Analyzer | Metrics | Status |
|----------|---------|--------|
| **Length** | char_count, word_count, sentence_count, token_count (tiktoken) | ✅ Complete |
| **Diversity** | unique_words_ratio, type_token_ratio, vocabulary_richness, hapax_legomena_ratio | ✅ Complete |
| **Format** | markdown, JSON, code blocks, URLs, emails, complexity score | ✅ Complete |
| **Quality** | PII detection, encoding issues, special token leakage, repetition ratio | ✅ Complete |
| **Embedding** | semantic duplicates, clustering (requires sentence-transformers) | ✅ Complete |

### Implemented Recommendations Engine

The `RecommendationsEngine` (in `src/oumi/core/analyze/recommendations.py`) checks for:

- Outliers (values > N standard deviations from mean)
- Exact duplicates (excluding system prompts)
- Empty/near-empty content (≤5 characters)
- Short content (<10 words)
- Role distribution imbalance (>80% single role)
- Token lengths exceeding context windows (4k, 8k, 16k)
- Conversation length distribution (single-turn, very long)
- Language consistency (<90% dominant language)
- Special token leakage
- Instruction format inconsistency (mixing Alpaca, Vicuna, ChatML, etc.)
- PII detection
- Quality scores below threshold
- Encoding issues (mojibake)
- High repetition

### Health Score Components

The `HealthScoreCalculator` (in `src/oumi/core/analyze/health_score.py`) computes scores across:

- Diversity
- Balance
- Quality
- Consistency

---

## Part 2: Common SFT Dataset Problems

Based on research and the current implementation, these are the key problems that can degrade SFT effectiveness:

### Size & Length Issues

| Problem | Impact | Detection |
|---------|--------|-----------|
| Dataset too small | Insufficient learning signal | ⚠️ Not detected |
| Samples too long | Truncation, context overflow | ✅ Detected |
| Empty/near-empty messages | No learning signal | ✅ Detected |
| Very short content | Limited learning signal | ✅ Detected |

### Diversity & Repetition Issues

| Problem | Impact | Detection |
|---------|--------|-----------|
| Exact duplicates | Overfitting, wasted compute | ✅ Detected |
| Near-duplicates (semantic) | Overfitting | ✅ Detected (embedding analyzer) |
| High repetition within samples | Model learns repetitive patterns | ✅ Detected |
| Low vocabulary diversity | Limited expressiveness | ✅ Detected |
| Topic/domain imbalance | Capability gaps | ⚠️ Not detected |

### Distribution & Balance Issues

| Problem | Impact | Detection |
|---------|--------|-----------|
| Role imbalance | Biased learning | ✅ Detected |
| Single-turn dominance | Poor multi-turn capability | ✅ Detected |
| Statistical outliers | Noisy gradients | ✅ Detected |
| Task type imbalance | Capability gaps | ⚠️ Not detected |

### Data Quality Issues

| Problem | Impact | Detection |
|---------|--------|-----------|
| PII leakage | Privacy/legal risks, memorization | ✅ Detected |
| Encoding issues (mojibake) | Corrupted learning signal | ✅ Detected |
| Special token leakage | Training interference | ✅ Detected |
| Inconsistent instruction formats | Confusion, poor generalization | ✅ Detected |
| Low-quality responses | Model learns bad patterns | ⚠️ Not detected |
| Instruction-response misalignment | Incorrect associations | ⚠️ Not detected |

### Language Issues

| Problem | Impact | Detection |
|---------|--------|-----------|
| Unintended mixed languages | Inconsistent behavior | ✅ Detected |
| Language inconsistency | Poor coherence | ✅ Detected |

---

## Part 3: Frontier Lab Techniques (2024-2025)

### Meta - Llama 3

**Source:** [Technical Report](https://arxiv.org/pdf/2407.21783)

| Technique | Description | In oumi? |
|-----------|-------------|----------|
| Rejection Sampling | Generate 10-30 outputs, reward model selects best | ❌ No |
| Multi-round Post-training | 6 iterative rounds of SFT + DPO | N/A (workflow) |
| Semantic Deduplication | Cluster with RoBERTa embeddings | ✅ Partial |
| Quality × Difficulty Scoring | Sort by `quality_score × difficulty_score` | ❌ No |
| Topic/Complexity Categorization | Adjust mix across axes | ❌ No |
| Synthetic Data Cleaning | Remove excessive emojis, exclamation points | ❌ No |

### DeepSeek - V3/R1

**Source:** [Technical Report](https://arxiv.org/html/2412.19437v1)

| Technique | Description | In oumi? |
|-----------|-------------|----------|
| Rejection Sampling | Filter R1-generated reasoning data | ❌ No |
| Chain-of-Thought Filtering | Remove mixed language, long paragraphs, code, incorrect chains | ❌ No |
| Human Verification | For non-reasoning data | N/A (manual) |
| Domain-specific Curation | Different techniques per domain | ❌ No |
| Generative Reward Model | Score reasoning quality | ❌ No |

### Alibaba - Qwen 2.5

**Source:** [Technical Report](https://arxiv.org/pdf/2412.15115)

| Technique | Description | In oumi? |
|-----------|-------------|----------|
| LLM-based Quality Scoring | Previous-gen models filter/score | ❌ No |
| Execution-based Filtering | Verify code actually executes | ❌ No |
| Iterative Filtering | 4-stage pipeline | ❌ No |
| fastText Scoring | Text-code grounding quality | ❌ No |
| Two-stage SFT | Short first, then mixed | N/A (workflow) |

### Microsoft - Phi-4

**Source:** [Technical Report](https://arxiv.org/html/2412.08905v1)

| Technique | Description | In oumi? |
|-----------|-------------|----------|
| Multi-agent Prompting | Synthetic data generation | N/A (generation) |
| Self-revision Cycles | Iterative improvement | N/A (generation) |
| Instruction Reversal | Novel augmentation | N/A (generation) |
| "Teachable" Edge Examples | Neither too easy nor too hard | ❌ No |
| LLM-based Best Selection | Generate multiple, select best | ❌ No |
| Quality Principles | Diversity, Nuance, Accuracy, CoT | ❌ Partial |

### Allen AI - OLMo 2 / Tulu 3

**Source:** [Blog](https://allenai.org/blog/olmo2), [Paper](https://arxiv.org/pdf/2411.15124)

| Technique | Description | In oumi? |
|-----------|-------------|----------|
| Skill-targeted Curation | Prompts targeting specific capabilities | ❌ No |
| Synthetic Gap-filling | Generate where public data lacking | N/A (generation) |
| RLVR | RL with Verifiable Rewards | N/A (training) |
| Quality Filtering | Remove problematic examples | ✅ Partial |
| Prompt Diversity Analysis | Millions of prompts as starting point | ❌ No |

### Zhipu AI - GLM-4

**Source:** [Paper](https://arxiv.org/html/2406.12793v1)

| Technique | Description | In oumi? |
|-----------|-------------|----------|
| Complex Question Annotation | Multi-fact reasoning for long-context | ❌ No |
| Task-type Screening | Based on application scenarios | ❌ No |
| Hand-crafted Quality | Emphasis on authentic interactions | N/A (manual) |

---

## Part 4: Academic Research Findings (2024-2025)

### Key Papers and Techniques

#### IFD (Instruction-Following Difficulty)

**Source:** [Cherry LLM](https://github.com/tianyi-lab/Cherry_LLM), [Superfiltering](https://arxiv.org/html/2402.00530v1)

```
IFD = PPL(response | no instruction) / PPL(response | with instruction)
```

- Higher IFD = instruction provides more guidance = more valuable sample
- 5-10% of data selected via IFD matches full-data performance
- Weak models (Qwen3-0.6b) can filter for strong models

**Status in oumi:** ❌ Not implemented

#### DEITA (Data-Efficient Instruction Tuning)

**Source:** [ICLR 2024](https://arxiv.org/abs/2312.15685)

Three dimensions:

1. **EVOL Complexity** - Instruction difficulty scoring
2. **EVOL Quality** - Response quality assessment
3. **Repr Filter** - Embedding-based diversity selection

Combined score: `complexity × quality`

6K samples achieve SOTA with this approach.

**Status in oumi:** ❌ Not implemented

#### AlpaGasus

**Source:** [ICLR 2024](https://arxiv.org/abs/2307.08701v5)

- ChatGPT scores each sample 0-5 on helpfulness/accuracy
- Threshold of 4.5 filters 52k → 9k samples
- 9k filtered outperforms 52k unfiltered
- 5.7x faster training

**Status in oumi:** ❌ Not implemented

#### LIMA (Less Is More for Alignment)

**Source:** [NeurIPS 2023](https://arxiv.org/abs/2305.11206)

Key findings:

- 1,000 carefully curated samples sufficient for alignment
- Diversity matters more than quantity
- Quality has 0.5 point impact vs. filtering
- Doubling quantity doesn't improve quality

**Status in oumi:** Diversity partially covered, quality scoring not implemented

#### HelpSteer Family

**Source:** [NAACL 2024](https://aclanthology.org/2024.naacl-long.185.pdf)

Multi-attribute scoring:

- Helpfulness
- Correctness
- Coherence
- Complexity
- Verbosity

**Status in oumi:** ❌ Not implemented

#### ArmoRM (Reward Model Scoring)

**Source:** [RLHFlow](https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/)

- Multi-objective reward modeling
- Used to filter Magpie datasets (top 10-30%)
- Interpretable preferences via MoE

**Status in oumi:** ❌ Not implemented

#### 2025 Findings

**Source:** [NAACL 2025](https://aclanthology.org/2025.naacl-long.336.pdf), [Post-Training Study](https://arxiv.org/html/2506.06522v2)

- Perplexity is most reliable predictor of downstream improvement
- Quality × Difficulty scoring outperforms either alone
- Model-dependent selection (GRAPE) reduces distribution shift
- SFTMix: confidence-aware training dynamics

---

## Part 5: Gap Analysis & Recommendations

### Critical Gaps (High Priority)

| Missing Feature | Evidence | Impact |
|-----------------|----------|--------|
| **LLM-as-Judge Quality Scoring** | Used by Phi-4, Qwen, AlpaGasus, DEITA | Filters 80%+ of low-quality data |
| **Instruction Complexity Scoring** | DEITA, Phi-4 "teachable examples" | Identifies high-signal samples |
| **Quality × Difficulty Combined Score** | Llama 3, DEITA, Phi-4 | Better than either metric alone |
| **Perplexity/IFD Scoring** | Cherry LLM, Superfiltering, NAACL 2025 | Most reliable quality predictor |
| **Task/Domain Classification** | DeepSeek, Qwen, Llama 3 | Enables domain-specific curation |

### Important Gaps (Medium Priority)

| Missing Feature | Evidence | Impact |
|-----------------|----------|--------|
| **Response Helpfulness Scoring** | HelpSteer, ArmoRM | Multi-dimensional quality |
| **Response Correctness Scoring** | HelpSteer, human verification | Filters incorrect responses |
| **Instruction-Response Alignment** | General best practice | Detects mismatches |
| **Code Execution Verification** | Qwen 2.5 | Critical for code datasets |
| **Chain-of-Thought Quality** | DeepSeek R1 | For reasoning datasets |

### Nice-to-Have Gaps (Lower Priority)

| Missing Feature | Evidence | Impact |
|-----------------|----------|--------|
| **Reward Model Scoring** | ArmoRM, Llama 3 | Advanced quality signal |
| **Synthetic Data Cleaning** | Llama 3 (emoji, exclamation) | Polish synthetic data |
| **Toxicity/Safety Scoring** | General best practice | Safety filtering |
| **Math/LaTeX Validation** | Domain-specific | For math datasets |

---

## Part 6: Proposed New Analyzers

### 1. LLM Judge Analyzer (High Priority)

```yaml
- id: llm_judge
  params:
    model: gpt-4o-mini  # or local model
    criteria:
      - helpfulness
      - correctness
      - coherence
    score_range: [0, 5]
    threshold: 4.0
```

**Metrics:**

- `helpfulness_score` (0-5)
- `correctness_score` (0-5)
- `coherence_score` (0-5)
- `overall_quality_score` (0-5)
- `passes_threshold` (bool)

### 2. Instruction Complexity Analyzer (High Priority)

```yaml
- id: complexity
  params:
    method: evol  # or heuristic
    model: gpt-4o-mini  # for evol method
```

**Metrics:**

- `instruction_complexity_score` (0-10)
- `task_type` (qa, reasoning, coding, creative, math, summarization)
- `requires_multi_step` (bool)
- `requires_external_knowledge` (bool)

### 3. Difficulty Analyzer (High Priority)

```yaml
- id: difficulty
  params:
    method: ifd  # or perplexity
    model_name: gpt2  # base model for perplexity
    compute_ifd: true
```

**Metrics:**

- `perplexity` (float)
- `ifd_score` (float) - Instruction-Following Difficulty
- `difficulty_bucket` (easy, medium, hard)

### 4. Combined Quality Score Analyzer

```yaml
- id: combined_score
  params:
    quality_weight: 0.5
    complexity_weight: 0.3
    difficulty_weight: 0.2
```

**Metrics:**

- `combined_score` = quality × complexity × difficulty
- `selection_rank` (1 to N)
- `top_k_percentile` (bool)

### 5. Task Classifier Analyzer (Medium Priority)

```yaml
- id: task_classifier
  params:
    categories:
      - qa
      - reasoning
      - coding
      - math
      - creative_writing
      - summarization
      - translation
      - conversation
```

**Metrics:**

- `task_type` (string)
- `task_confidence` (0-1)
- `domain` (string)

### 6. Code Quality Analyzer (Medium Priority)

```yaml
- id: code_quality
  params:
    verify_syntax: true
    verify_execution: false  # expensive
    languages: [python, javascript]
```

**Metrics:**

- `has_code` (bool)
- `code_language` (string)
- `syntax_valid` (bool)
- `execution_success` (bool, optional)

### 7. Instruction-Response Alignment Analyzer

```yaml
- id: alignment
  params:
    model: all-MiniLM-L6-v2
    threshold: 0.5
```

**Metrics:**

- `alignment_score` (0-1) - semantic similarity
- `response_addresses_instruction` (bool)
- `is_non_sequitur` (bool)

---

## Part 7: Implementation Priorities

### Phase 1: Foundation (Highest Impact)

1. **LLM Judge Analyzer** - Most requested, highest ROI
2. **Task Classifier** - Enables domain analysis
3. **Instruction Complexity** - Part of quality × complexity scoring

### Phase 2: Advanced Metrics

4. **Difficulty/IFD Analyzer** - Research-backed effectiveness
5. **Combined Score Calculator** - Ties everything together
6. **Instruction-Response Alignment** - Catches mismatches

### Phase 3: Domain-Specific

7. **Code Quality Analyzer** - For coding datasets
8. **Chain-of-Thought Analyzer** - For reasoning datasets
9. **Math Validator** - For math datasets

### Phase 4: Advanced

10. **Reward Model Integration** - ArmoRM or similar
11. **Synthetic Data Cleaner** - For synthetic pipelines

---

## Part 8: Key Takeaways

### Research Consensus

1. **Quality > Quantity** - 1k-10k high-quality samples often sufficient
2. **Diversity matters** - Topic/task diversity as important as volume
3. **Quality × Difficulty** - Neither alone is optimal
4. **Iterative filtering** - Multiple passes improve quality
5. **Model-based scoring** - LLM judges are effective and scalable

### What Frontier Labs Do That We Don't

1. Score every sample with LLM judge or reward model
2. Compute instruction complexity
3. Use difficulty/IFD metrics
4. Classify task types for balanced mixing
5. Apply domain-specific quality checks

### Recommended Next Steps

1. Implement LLM Judge Analyzer (biggest gap)
2. Add task/domain classification
3. Implement complexity scoring
4. Add combined quality × complexity × difficulty score
5. Create filtering recommendations based on these scores

---

## References

### Frontier Model Reports

- [Llama 3 Technical Report](https://arxiv.org/pdf/2407.21783) (Meta, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1) (DeepSeek, 2024)
- [Qwen 2.5 Technical Report](https://arxiv.org/pdf/2412.15115) (Alibaba, 2024)
- [Phi-4 Technical Report](https://arxiv.org/html/2412.08905v1) (Microsoft, 2024)
- [OLMo 2 Blog](https://allenai.org/blog/olmo2) (Allen AI, 2024)
- [Tulu 3 Paper](https://arxiv.org/pdf/2411.15124) (Allen AI, 2024)
- [GLM-4 Paper](https://arxiv.org/html/2406.12793v1) (Zhipu AI, 2024)

### Academic Papers

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (NeurIPS 2023)
- [AlpaGasus: Training A Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701v5) (ICLR 2024)
- [DEITA: What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685) (ICLR 2024)
- [Cherry LLM: Self-Guided Data Selection](https://github.com/tianyi-lab/Cherry_LLM) (NAACL 2024)
- [Superfiltering: Weak-to-Strong Data Filtering](https://arxiv.org/html/2402.00530v1) (ACL 2024)
- [HelpSteer: Multi-attribute Helpfulness Dataset](https://aclanthology.org/2024.naacl-long.185.pdf) (NAACL 2024)
- [ArmoRM: Multi-Objective Reward Modeling](https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/) (2024)
- [A Rethinking on Data Selection for Fine-Tuning](https://aclanthology.org/2025.naacl-long.336.pdf) (NAACL 2025)
- [Fixing It in Post: LLM Post-Training Study](https://arxiv.org/html/2506.06522v2) (2025)

---

## Appendix A: Practical Usage Guide for oumi analyze

This appendix provides step-by-step instructions for running each currently implemented analyzer.

### Prerequisites

```bash
# Basic installation
pip install oumi

# For HTML reports and embedding analyzer
pip install "oumi[analyze_advanced]"

# For language detection (optional)
pip install langdetect
```

---

### 1. Length Analyzer

**Purpose:** Measures text length in characters, words, sentences, and tokens.

**When to use:**

- Check if samples exceed context window limits
- Identify empty or very short samples
- Understand token distribution for training cost estimation

**Command:**

```bash
oumi analyze --config configs/examples/analyze/analyze.yaml
```

**Minimal Config (`length_analysis.yaml`):**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 1000
output_path: ./analysis_output

analyzers:
  - id: length
    params:
      char_count: true      # Character count
      word_count: true      # Word count (whitespace-separated)
      sentence_count: true  # Sentence count (split on .!?)
      token_count: true     # Token count (tiktoken o200k_base default)
      # tiktoken_encoding: o200k_base  # GPT-4o/GPT-5 (default)
      # tiktoken_encoding: cl100k_base # GPT-4/GPT-3.5-turbo
```

**Using a Custom Tokenizer:**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 1000
output_path: ./analysis_output

# Use a specific tokenizer instead of tiktoken
tokenizer_name: meta-llama/Llama-3.1-8B-Instruct

analyzers:
  - id: length
    params:
      token_count: true
      tiktoken_encoding: null  # Disable tiktoken, use tokenizer_name
```

**Expected Output:**

```
┌─────────────────────────────────────────────────────────────┐
│                      Dataset Overview                       │
├─────────────────────────────────────────────────────────────┤
│ Dataset: yahma/alpaca-cleaned                               │
│ Conversations: 1000 (100% coverage)  Messages: 2000         │
│ Analyzers: length                                           │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     Analysis Metrics                         │
├────────────┬────────────────────┬──────────┬────────────────┤
│ Analyzer   │ Metric             │ Mean     │ Count          │
├────────────┼────────────────────┼──────────┼────────────────┤
│ Length     │ Token Count        │ 127.4    │ 2000           │
│            │ Char Count         │ 523.8    │ 2000           │
│            │ Word Count         │ 89.2     │ 2000           │
└────────────┴────────────────────┴──────────┴────────────────┘

Saved message analysis to: ./analysis_output/message_analysis.csv
Saved analysis summary to: ./analysis_output/analysis_summary.json
```

**Output Files:**

- `message_analysis.csv` - Per-message metrics with columns like `text_content_length_token_count`
- `conversation_analysis.csv` - Aggregated per-conversation
- `analysis_summary.json` - Statistics (mean, std, min, max, median)

---

### 2. Diversity Analyzer

**Purpose:** Measures vocabulary richness and text diversity.

**When to use:**

- Detect repetitive or templated responses
- Assess vocabulary complexity
- Identify low-diversity samples that may cause overfitting

**Command:**

```bash
oumi analyze --config configs/examples/analyze/analyze_diversity.yaml
```

**Config (`analyze_diversity.yaml`):**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 1000
output_path: ./analysis_output

generate_recommendations: true
outlier_threshold: 3.0

analyzers:
  - id: length
    params:
      token_count: true

  - id: diversity
    params:
      unique_words_ratio: true      # unique words / total words (0-1)
      type_token_ratio: true        # unique tokens / total tokens (0-1)
      vocabulary_richness: true     # Log-adjusted TTR (better for varying lengths)
      hapax_legomena_ratio: false   # Words appearing only once
      case_sensitive: false         # Treat "Hello" and "hello" as same
```

**Metrics Explained:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `unique_words_ratio` | unique/total words | 0.5 = 50% unique words |
| `type_token_ratio` | unique/total tokens | Higher = more diverse |
| `vocabulary_richness` | log-adjusted TTR | Better for varying text lengths |
| `hapax_legomena_ratio` | once-words/unique | High = many rare words |

**Expected Output:**

```
┌──────────────────────────────────────────────────────────────┐
│                     Analysis Metrics                         │
├────────────┬────────────────────────┬──────────┬────────────┤
│ Analyzer   │ Metric                 │ Mean     │ Count      │
├────────────┼────────────────────────┼──────────┼────────────┤
│ Diversity  │ Unique Words Ratio     │ 0.72     │ 2000       │
│            │ Type Token Ratio       │ 0.68     │ 2000       │
│            │ Vocabulary Richness    │ 0.81     │ 2000       │
└────────────┴────────────────────────┴──────────┴────────────┘
```

---

### 3. Format Analyzer

**Purpose:** Detects structured content (markdown, code blocks, JSON, URLs).

**When to use:**

- Analyze coding datasets for language distribution
- Detect markdown formatting patterns
- Identify samples with URLs or structured data

**Command:**

```bash
oumi analyze --config configs/examples/analyze/analyze_format.yaml
```

**Config (`analyze_format.yaml`):**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 1000
output_path: ./analysis_output

analyzers:
  - id: length
    params:
      token_count: true

  - id: format
    params:
      detect_markdown: true      # Headers (#), lists (- *), bold (**), italic (*)
      detect_json: true          # JSON code blocks and inline JSON
      detect_code_blocks: true   # Fenced code blocks (```) with language detection
      detect_urls: true          # HTTP/HTTPS URLs
      detect_emails: false       # Email addresses
      compute_complexity: true   # Overall format complexity score (0-1)
```

**Metrics Produced:**

- `has_markdown` (bool) - Contains markdown formatting
- `has_json` (bool) - Contains JSON content
- `has_code_blocks` (bool) - Contains fenced code blocks
- `code_block_languages` (list) - Detected languages (python, javascript, etc.)
- `has_urls` (bool) - Contains URLs
- `format_complexity_score` (0-1) - Composite complexity

**Expected Output:**

```
┌──────────────────────────────────────────────────────────────┐
│                     Analysis Metrics                         │
├────────────┬────────────────────────┬──────────┬────────────┤
│ Analyzer   │ Metric                 │ Mean     │ Count      │
├────────────┼────────────────────────┼──────────┼────────────┤
│ Format     │ Has Markdown           │ 0.23     │ 2000       │
│            │ Has Code Blocks        │ 0.15     │ 2000       │
│            │ Has Json               │ 0.08     │ 2000       │
│            │ Complexity Score       │ 0.31     │ 2000       │
└────────────┴────────────────────────┴──────────┴────────────┘
```

---

### 4. Quality Analyzer

**Purpose:** Detects PII, encoding issues, special token leakage, and repetitive content.

**When to use:**

- **Before training:** Check for PII that could be memorized
- **Data cleaning:** Find encoding issues (mojibake)
- **Safety:** Detect leaked special tokens that interfere with training
- **Quality:** Identify highly repetitive samples

**Command:**

```bash
oumi analyze --config configs/examples/analyze/analyze_quality.yaml
```

**Config (`analyze_quality.yaml`):**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 1000
output_path: ./analysis_output/quality

generate_recommendations: true
generate_report: true
report_title: "Dataset Quality & Safety Analysis"

analyzers:
  - id: length
    params:
      token_count: true

  - id: quality
    params:
      # PII Detection
      detect_pii: true           # Master switch
      detect_emails: true        # email@example.com
      detect_phones: true        # (123) 456-7890
      detect_ssn: true           # 123-45-6789
      detect_credit_cards: true  # 1234-5678-9012-3456
      detect_ip_addresses: false # Often legitimate in tech content
      detect_api_keys: true      # api_key=xxx, secret=xxx

      # Content Quality
      detect_encoding_issues: true   # Mojibake, invalid UTF-8
      detect_special_tokens: true    # <|endoftext|>, [INST], <s>

      # Repetition Detection
      detect_repetition: true
      repetition_ngram_size: 3       # Check 3-gram repetition
      repetition_threshold: 0.3      # Flag if >30% repeated n-grams

      # Language Detection (requires: pip install langdetect)
      detect_language: false

      # Composite Score
      compute_quality_score: true    # 0-1, higher = better quality
```

**Metrics Produced:**

| Metric | Type | Description |
|--------|------|-------------|
| `has_pii` | bool | Any PII detected |
| `pii_types` | string | Comma-separated: "email,phone,ssn" |
| `pii_count` | int | Total PII instances found |
| `has_encoding_issues` | bool | Mojibake or invalid chars |
| `has_special_tokens` | bool | Leaked LLM special tokens |
| `repetition_ratio` | float | 0-1, ratio of repeated n-grams |
| `has_high_repetition` | bool | repetition_ratio > threshold |
| `quality_score` | float | 0-1, composite (1 = clean) |
| `detected_language` | string | ISO code (en, fr, etc.) |

**Expected Recommendations:**

```
┌──────────────────────────────────────────────────────────────┐
│                   Recommendations (3)                        │
├──────────────────────────────────────────────────────────────┤
│ ● HIGH: PII (Personally Identifiable Information) detected   │
│    Found 45 messages (4.5%) containing potential PII.        │
│    Types detected: email (30), phone (15).                   │
│                                                              │
│ ● MEDIUM: Special token leakage detected                     │
│    Found 12 messages (1.2%) containing leaked special tokens │
│                                                              │
│ ● LOW: Highly repetitive content detected                    │
│    Found 23 messages (2.3%) with high repetition ratios.     │
└──────────────────────────────────────────────────────────────┘
```

---

### 5. Embedding Analyzer

**Purpose:** Semantic analysis using sentence embeddings for duplicate detection and clustering.

**When to use:**

- Find near-duplicate samples with different wording
- Cluster similar conversations
- Identify semantic repetition that exact-match dedup misses

**Prerequisites:**

```bash
pip install "oumi[analyze_advanced]"  # Installs sentence-transformers
```

**Command:**

```bash
oumi analyze --config configs/examples/analyze/analyze_embedding.yaml
```

**Config (`analyze_embedding.yaml`):**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 500  # Embeddings are compute-intensive
output_path: ./analysis_output

analyzers:
  - id: length
    params:
      token_count: true

  - id: embedding
    params:
      # Model selection
      model_name: all-MiniLM-L6-v2     # Fast (384 dims)
      # model_name: all-mpnet-base-v2  # Higher quality (768 dims)

      # Semantic duplicate detection
      detect_duplicates: true
      duplicate_threshold: 0.95        # Cosine similarity (0.9-0.99)

      # Clustering (optional)
      cluster_samples: true
      clustering_method: dbscan        # or "kmeans"
      eps: 0.5                         # DBSCAN: max distance
      min_samples: 2                   # DBSCAN: min cluster size
      # n_clusters: 10                 # Required for kmeans

      # Performance
      batch_size: 32
      # device: cuda                   # "cuda", "cpu", or null (auto)

      # Storage (warning: increases output size)
      store_embeddings: false
```

**Metrics Produced:**

| Metric | Type | Description |
|--------|------|-------------|
| `embedding_duplicate_group` | int | Group ID (-1 = no duplicate) |
| `embedding_has_semantic_duplicate` | bool | Has near-duplicate |
| `embedding_cluster` | int | Cluster label (if enabled) |

**Expected Output:**

```
┌──────────────────────────────────────────────────────────────┐
│                     Analysis Metrics                         │
├────────────┬────────────────────────┬──────────┬────────────┤
│ Analyzer   │ Metric                 │ Mean     │ Count      │
├────────────┼────────────────────────┼──────────┼────────────┤
│ Embedding  │ Has Semantic Duplicate │ 0.08     │ 1000       │
│            │ Duplicate Group        │ 12.3     │ 1000       │
│            │ Cluster                │ 5.2      │ 1000       │
└────────────┴────────────────────────┴──────────┴────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Recommendations (1)                        │
├──────────────────────────────────────────────────────────────┤
│ ● MEDIUM: Semantic duplicates detected                       │
│    Found 80 messages (8.0%) with semantic duplicates.        │
│    Consider deduplicating to improve training diversity.     │
└──────────────────────────────────────────────────────────────┘
```

---

### 6. Comprehensive Analysis (All Analyzers)

**Purpose:** Run all analyzers for complete dataset health assessment.

**Command:**

```bash
oumi analyze --config configs/examples/analyze/analyze_comprehensive.yaml --report
```

**Config (`analyze_comprehensive.yaml`):**

```yaml
dataset_name: yahma/alpaca-cleaned
split: train
sample_count: 1000
output_path: ./analysis_output/comprehensive

# Use specific tokenizer
tokenizer_name: meta-llama/Llama-3.1-8B-Instruct

# Enable recommendations and HTML report
generate_recommendations: true
outlier_threshold: 3.0
generate_report: true
report_title: "Comprehensive Dataset Health Report"

analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: true

  - id: diversity
    params:
      unique_words_ratio: true
      type_token_ratio: true
      vocabulary_richness: true
      hapax_legomena_ratio: true
      case_sensitive: false

  - id: format
    params:
      detect_markdown: true
      detect_json: true
      detect_code_blocks: true
      detect_urls: true
      detect_emails: true
      compute_complexity: true

  - id: quality
    params:
      detect_pii: true
      detect_emails: true
      detect_phones: true
      detect_ssn: true
      detect_credit_cards: true
      detect_ip_addresses: true
      detect_api_keys: true
      detect_encoding_issues: true
      detect_special_tokens: true
      detect_repetition: true
      repetition_ngram_size: 3
      repetition_threshold: 0.3
      detect_language: false
      compute_quality_score: true
```

**Output Files:**

```
./analysis_output/comprehensive/
├── message_analysis.csv           # Per-message metrics (all analyzers)
├── conversation_analysis.csv      # Per-conversation aggregates
├── analysis_summary.json          # Statistics + health score + recommendations
└── analysis_report.html           # Interactive HTML report (if --report)
```

---

### 7. Using Local Files

**For JSONL files in Oumi format:**

```yaml
dataset_path: /path/to/your/data.jsonl
output_path: ./analysis_output

analyzers:
  - id: length
    params:
      token_count: true
```

**Oumi format example (`data.jsonl`):**

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

---

### 8. CLI Options Reference

```bash
# Basic usage
oumi analyze --config path/to/config.yaml

# Override output directory
oumi analyze --config config.yaml --output ./my_output

# Change output format (csv, json, parquet)
oumi analyze --config config.yaml --format json

# Generate HTML report
oumi analyze --config config.yaml --report

# Custom report title
oumi analyze --config config.yaml --report --report-title "My Analysis"

# Verbose output (show full statistics)
oumi analyze --config config.yaml --verbose

# Override dataset inline
oumi analyze --config config.yaml --dataset_name openai/gsm8k

# Override sample count inline
oumi analyze --config config.yaml --sample_count 500
```

---

### 9. Health Score Interpretation

The health score (in `analysis_summary.json`) provides an overall grade:

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| **A** | 90-100 | Excellent - Ready for training |
| **B** | 80-89 | Good - Minor issues to review |
| **C** | 70-79 | Fair - Some quality concerns |
| **D** | 60-69 | Poor - Significant issues |
| **F** | 0-59 | Failing - Major problems |

**Components:**

- **Diversity** - Vocabulary richness across samples
- **Balance** - Role distribution, conversation lengths
- **Quality** - PII, encoding, special tokens
- **Consistency** - Format consistency, language uniformity

---

### 10. Common Workflows

#### Pre-training Data Audit

```bash
# Quick quality check
oumi analyze -c configs/examples/analyze/analyze_quality.yaml \
  --dataset_name your_dataset --sample_count 5000 --report
```

#### Find Duplicates Before Training

```bash
# Semantic deduplication analysis
oumi analyze -c configs/examples/analyze/analyze_embedding.yaml \
  --dataset_name your_dataset --sample_count 10000
```

#### Full Health Check

```bash
# Comprehensive analysis with report
oumi analyze -c configs/examples/analyze/analyze_comprehensive.yaml \
  --dataset_name your_dataset --report
```

#### Analyze Local File

```bash
oumi analyze -c configs/examples/analyze/analyze.yaml \
  --dataset_path /path/to/data.jsonl --output ./results
```
