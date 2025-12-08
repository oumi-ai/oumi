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

The `RecommendationsEngine` (in `recommendations.py`) checks for:

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

The `HealthScoreCalculator` computes scores across:

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
- Weak models (GPT-2) can filter for strong models (LLaMA-7B)

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
