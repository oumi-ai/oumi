# Oumi Analyze Feature Review

## Overview

The `oumi analyze` feature is a **plugin-based dataset analysis system** for evaluating and improving training data quality. It uses a registry pattern where analyzers self-register via decorators.

## Architecture

```
CLI (oumi/cli/analyze.py)
    ↓
AnalyzeConfig (core/configs/analyze_config.py)
    ↓
DatasetAnalyzer (core/analyze/dataset_analyzer.py)
    ├─ Registry (sample analyzers lookup)
    ├─ DataFrameAnalyzer (core analysis engine)
    └─ Individual SampleAnalyzers (22 total)
```

## Complete Analyzer Metrics Reference

---

### LengthAnalyzer (`length`)

**File**: `src/oumi/core/analyze/length_analyzer.py`
**Dependencies**: tiktoken (optional), transformers (optional)

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `token_count` | int | Number of tokens using specified tokenizer | tiktoken (default: o200k_base for GPT-4o/5) or HuggingFace tokenizer | tiktoken encodings are OpenAI-specific; HF tokenizers require model download |

---

### DiversityAnalyzer (`diversity`)

**File**: `src/oumi/core/analyze/diversity_analyzer.py`
**Dependencies**: None

| Metric | Type | Range | Description | Implementation | Limitations |
|--------|------|-------|-------------|----------------|-------------|
| `unique_words_ratio` | float | 0-1 | Ratio of unique words to total words | `len(set(words)) / len(words)` | Simple whitespace tokenization; case-insensitive by default; no stemming/lemmatization |

---

### FormatAnalyzer (`format`)

**File**: `src/oumi/core/analyze/format_analyzer.py`
**Dependencies**: None

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `has_markdown` | bool | Contains markdown formatting | Regex for headers (#), lists (- *), bold (**), italic (*_), links, images | May miss complex markdown; false positives on code comments |
| `has_json` | bool | Contains JSON content | Fenced ```json blocks + inline JSON validated with `json.loads()` | May miss partial/malformed JSON; inline detection can have false positives |
| `has_code_blocks` | bool | Contains fenced code blocks | Regex for ``` or ~~~ with optional language | Doesn't detect indented code blocks (4 spaces) |
| `code_block_count` | int | Number of code blocks | Count of fenced code block matches | Only counts fenced blocks; misses inline code |
| `code_block_languages` | str | Comma-separated list of detected languages | Extract language after opening fence | Language hints are optional; may be incorrect or missing |
| `has_urls` | bool | Contains HTTP/HTTPS URLs | Regex for `https?://...` pattern | May miss other URL schemes (ftp, mailto); false positives on malformed URLs |
| `has_emails` | bool | Contains email addresses | Standard email regex pattern | False positives on similar patterns; misses unusual TLDs |
| `format_complexity_score` | float | 0-1 | Weighted complexity score | Markdown=2, JSON=2, Code=3+bonus, URLs=1, Emails=1; normalized | Weights are arbitrary; doesn't assess content quality |

---

### QualityAnalyzer (`quality`)

**File**: `src/oumi/core/analyze/quality_analyzer.py`
**Dependencies**: langdetect (optional for language detection)

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `has_pii` | bool | Contains personally identifiable information | OR of all PII type detections | Binary flag misses severity; false positives on synthetic data |
| `pii_types` | str | Comma-separated list of detected PII types | `email, phone, ssn, credit_card, api_key` | Limited to 5 types; misses names, addresses, dates of birth |
| `pii_count` | int | Total count of PII instances | Sum of all PII pattern matches | Multiple matches of same item counted separately |
| `detected_language` | str | ISO 639-1 language code | langdetect library (if enabled and installed) | Requires langdetect; short texts unreliable; disabled by default |
| `language_confidence` | float | 0-1 | Confidence in language detection | langdetect probability score | Low confidence on short texts |
| `has_encoding_issues` | bool | Contains mojibake/encoding problems | Regex for common mojibake patterns (UTF-8 decoded as Latin-1) | Only detects common patterns; may miss subtle issues |
| `repetition_ratio` | float | 0-1 | Ratio of repeated n-grams | 1 - (unique_ngrams / total_ngrams) with n=3 | N-gram size affects sensitivity; common phrases inflate value |
| `has_high_repetition` | bool | Repetition exceeds threshold | `repetition_ratio > 0.3` (default) | Threshold is arbitrary; prose vs code have different baselines |

---

### TrainingQualityAnalyzer (`training_quality`)

**File**: `src/oumi/core/analyze/training_quality_analyzer.py`
**Dependencies**: None

Analyzes assistant response quality for SFT datasets. Only computes metrics for assistant messages.

| Metric | Type | Role | Description | Implementation | Limitations |
|--------|------|------|-------------|----------------|-------------|
| `response_completeness_score` | float | assistant | 0-1 score for response completeness | Checks proper ending, structure; penalizes truncation | Heuristic-based; may penalize intentionally brief responses |
| `has_proper_ending` | bool | assistant | Response ends properly | Checks for: no trailing ellipsis, has punctuation, no trailing conjunctions | Strict rules; may flag streaming/partial responses |
| `has_structure` | bool | assistant | Response has formatting structure | Checks for: bullets, numbered lists, code blocks, headers, bold, blockquotes | May miss well-structured prose; favors formatted responses |
| `response_word_count` | int | assistant | Word count of response | `len(text.split())` | Simple whitespace tokenization |

---

### ContentPatternAnalyzer (`content_pattern`)

**File**: `src/oumi/core/analyze/content_pattern_analyzer.py`
**Dependencies**: None

Detects AI-specific quality issues commonly found in synthetic training data.

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `has_placeholder` | bool | Contains template placeholders | Regex for `[Name]`, `[Product Name]`, `<your_company>`, etc. | May flag legitimate bracketed content; misses novel placeholder styles |
| `placeholder_count` | int | Number of placeholders found | Count of bracket + angle placeholder matches | May double-count overlapping patterns |
| `placeholder_types` | str | Types of placeholders found | `bracket` ([...]) or `angle` (<...>) | Limited to 2 types; doesn't categorize by semantic meaning |
| `has_hallucinated_experience` | bool | Contains fabricated AI experiences | Patterns: "I had to...", "When I was a...", "In my experience..." | Keyword-based; may flag legitimate first-person writing |
| `has_nooutput` | bool | Contains no-output markers | Patterns: `<nooutput>`, `N/A`, `[N/A]`, `None`, `-` | May flag legitimate use of these terms; context-insensitive |
| `has_refusal` | bool | Contains AI refusal phrases | Patterns: "I cannot...", "I'm unable to...", "This task cannot be..." | May flag educational content about AI; context-insensitive |

---

### TaskCategoryAnalyzer (`task_category`)

**File**: `src/oumi/core/analyze/task_category_analyzer.py`
**Dependencies**: None

| Metric | Type | Values | Description | Implementation | Limitations |
|--------|------|--------|-------------|----------------|-------------|
| `task_category` | str | 12 categories | Primary task classification | Pattern matching against keyword lists for each category | Single-label; misses multi-task instructions |
| `task_confidence` | float | 0-1 | Confidence in classification | Based on pattern match count and specificity | Not probabilistic; may be high for ambiguous matches |
| `is_stem` | bool | True/False | Task is STEM-related | True if category in: math, coding, data_analysis | Limited to 3 categories; may miss scientific reasoning |
| `is_conversational` | bool | True/False | Task is conversational | True if category in: advice, role_play | Very limited; misses casual chat, emotional support |

**Task Categories**:

- `math`: Calculations, proofs, equations, word problems
- `coding`: Programming, debugging, code review, algorithms
- `information_seeking`: Factual questions, definitions, explanations
- `creative_writing`: Stories, poems, scripts, creative content
- `editing`: Grammar, rewriting, summarization, paraphrasing
- `advice`: Personal advice, recommendations, guidance
- `reasoning`: Logic puzzles, analysis, problem-solving
- `brainstorming`: Idea generation, planning, lists
- `role_play`: Character personas, simulated scenarios
- `data_analysis`: Statistics, visualization, data processing
- `translation`: Language translation tasks
- `other`: Doesn't fit other categories

---

### SafetyAnalyzer (`safety`)

**File**: `src/oumi/core/analyze/safety_analyzer.py`
**Dependencies**: None

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `safety_score` | float | 0-1 | Overall safety score (higher = safer) | 1.0 minus weighted penalties for each category | Fixed penalties; context-insensitive |
| `is_safe` | bool | True/False | Content passes safety threshold | `safety_score >= threshold` (default 0.7) | Binary classification loses nuance |
| `risk_level` | str | safe/low/medium/high | Risk severity level | Based on safety_score thresholds: high<0.5, medium<0.7, low<0.9, safe≥0.9 | Thresholds are arbitrary |
| `safety_categories` | str | Comma-separated | Flagged safety categories | List of categories with detected violations | May have multiple false positives |

**Safety Categories**:

| Category | Keywords | Limitations |
|----------|----------|-------------|
| `violence` | kill, murder, attack, bomb, weapon, torture | May flag news, history, fiction |
| `hate` | racial slurs, discrimination terms | Limited vocabulary; may miss coded language |
| `sexual` | explicit terms, adult content | May flag medical/educational content |
| `self_harm` | suicide, self-harm, overdose | May flag mental health support content |
| `illegal` | hacking, theft, drug dealing | May flag security research, journalism |
| `deception` | misinformation, fraud, scam | May flag fiction, satire |
| `dangerous` | weapon building, dangerous substances | May flag chemistry education |
| `privacy` | doxxing, personal information | May flag legitimate data discussions |

---

### DifficultyAnalyzer (`difficulty`)

**File**: `src/oumi/core/analyze/difficulty_analyzer.py`
**Dependencies**: None

| Metric | Type | Values | Description | Implementation | Limitations |
|--------|------|--------|-------------|----------------|-------------|
| `difficulty_score` | float | 0-1 | Composite difficulty score | Weighted sum of reasoning, domain, constraints, multi-part | Heuristic; doesn't measure actual cognitive load |
| `difficulty_tier` | str | easy/medium/hard/expert | Difficulty classification | Based on score thresholds: easy<0.3, medium<0.5, hard<0.7, expert≥0.7 | Arbitrary thresholds; relative not absolute |
| `requires_reasoning` | bool | True/False | Requires multi-step reasoning | Patterns: "step by step", "analyze", "compare", conditionals | May miss implicit reasoning needs |
| `requires_domain_knowledge` | bool | True/False | Requires specialized knowledge | Checks for domain-specific terms in 6 domains | English keyword lists; may miss subtle domain needs |
| `constraint_count` | int | 0+ | Number of explicit constraints | Count of: must, should, required, exactly, at least, etc. | May count incidental uses; misses implicit constraints |

**Domain Knowledge Detection**:

| Domain | Example Terms | Limitations |
|--------|---------------|-------------|
| `programming` | algorithm, database, API, recursion | May miss domain-specific languages |
| `math` | theorem, proof, derivative, matrix | Limited mathematical vocabulary |
| `science` | hypothesis, molecule, quantum | Very broad; may miss specialized fields |
| `legal` | statute, liability, litigation | Jurisdiction-specific terms may be missed |
| `medical` | diagnosis, treatment, anatomy | May miss specialized medical subfields |
| `finance` | portfolio, investment, equity | May miss crypto, fintech terms |

---

### CostAnalyzer (`cost`)

**File**: `src/oumi/core/analyze/cost_analyzer.py`
**Dependencies**: None (requires LengthAnalyzer for token counts)

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `fits_context_{size}` | bool | Sample fits in context window | `token_count <= context_size` for sizes: 2k, 4k, 8k, 16k, 32k | Doesn't account for system prompts, padding |
| `context_utilization_{size}` | float | 0-1 | Fraction of context used | `token_count / context_size` | Doesn't consider optimal batch boundaries |
| `tokens_wasted_{size}` | int | Tokens wasted in packing | `context_size - token_count` if sample used alone | Theoretical; actual training may differ |

**Dataset-Level Metrics**:

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `total_tokens` | int | Sum of all tokens | Sum of token_count column | Requires LengthAnalyzer to run first |
| `packing_efficiency_{size}` | float | 0-1 | Packing efficiency | First-fit decreasing bin packing; efficiency = total_tokens / (bins × size) | Theoretical; doesn't account for gradient accumulation |
| `estimated_batches_{size}` | int | Estimated batch count | Number of bins from packing algorithm | Doesn't consider actual batch size constraints |

---

### EmbeddingAnalyzer (`embedding`)

**File**: `src/oumi/core/analyze/embedding_analyzer.py`
**Dependencies**: sentence-transformers, scikit-learn, datasketch (optional for fuzzy)

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `duplicate_group` | int | Semantic duplicate group ID | Cosine similarity > threshold (default 0.95) assigns same group | 0.95 is very strict; may miss paraphrases |
| `has_semantic_duplicate` | bool | Has semantic duplicates | True if duplicate_group has >1 member | Binary; doesn't indicate similarity degree |
| `fuzzy_duplicate_group` | int | Fuzzy duplicate group ID | MinHash LSH with Jaccard threshold (default 0.8) | Requires datasketch; sensitive to n-gram size |
| `has_fuzzy_duplicate` | bool | Has fuzzy duplicates | True if fuzzy_duplicate_group has >1 member | May flag similar but distinct content |
| `fuzzy_jaccard_score` | float | 0-1 | Jaccard similarity score | MinHash estimated Jaccard coefficient | Approximation; may have variance |
| `cluster` | int | Cluster assignment | DBSCAN (default) or KMeans clustering | DBSCAN noise points get -1; KMeans requires n_clusters |
| `embedding` | list[float] | Embedding vector | sentence-transformers model output | Large storage; model choice affects quality |

**Configuration Impact**:

- `model_name`: "all-MiniLM-L6-v2" (default) is fast but less accurate than larger models
- `duplicate_threshold`: 0.95 is very strict; 0.85-0.90 catches more near-duplicates
- `duplicate_scope`: "all" compares everything; "by_role" only within same role
- `batch_size`: 32 default; increase for GPU, decrease for memory limits

---

### QuestionDiversityAnalyzer (`question_diversity`)

**File**: `src/oumi/core/analyze/question_diversity_analyzer.py`
**Dependencies**: sentence-transformers, scikit-learn

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `cluster_id` | int | Question cluster assignment | DBSCAN (eps=0.15) or KMeans clustering | -1 = noise/unique in DBSCAN; user role only |
| `cluster_size` | int | Size of assigned cluster | Count of samples in same cluster | Large clusters may indicate redundancy |
| `is_concentrated` | bool | In concentrated cluster | True if cluster_size > concentration_threshold | Threshold is configurable; may miss gradual redundancy |

**Dataset-Level Metrics**:

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `num_question_clusters` | int | Total cluster count | Number of unique cluster IDs (excluding noise) | DBSCAN auto-determines; KMeans is fixed |
| `diversity_ratio` | float | 0-1 | Fraction of unique questions | noise_samples / total_questions | DBSCAN-specific; noise = diverse |
| `question_entropy` | float | 0+ | Shannon entropy of distribution | -Σ(p × log2(p)) for cluster sizes | Higher = more diverse; scale depends on dataset |
| `question_gini` | float | 0-1 | Gini coefficient | 0 = uniform distribution, 1 = all in one cluster | Measures inequality; doesn't indicate absolute diversity |
| `diversity_rating` | str | low/medium/high | Overall diversity assessment | Based on entropy and Gini thresholds | Thresholds are arbitrary |

---

### ReprDiversityAnalyzer (`repr_diversity`)

**File**: `src/oumi/core/analyze/repr_diversity_analyzer.py`
**Dependencies**: sentence-transformers, scikit-learn

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `nn_distance` | float | 0-2 | Distance to nearest neighbor | 1 - cosine_similarity to closest sample | Scale depends on embedding model |
| `score` | float | 0-2 | Mean distance to K neighbors | Average of distances to K nearest (default K=5) | K choice affects sensitivity |
| `is_redundant` | bool | True/False | Below diversity threshold | True if score < threshold (default 0.3) | Threshold is arbitrary |
| `percentile` | float | 0-100 | Percentile rank in dataset | Rank position among all diversity scores | Relative measure; not comparable across datasets |

**Dataset-Level Metrics**:

| Metric | Type | Description | Limitations |
|--------|------|-------------|-------------|
| `redundant_ratio` | float | Fraction of redundant samples | Depends on threshold setting |
| `mean_diversity_score` | float | Average diversity | May be skewed by outliers |
| `median_diversity_score` | float | Median diversity | More robust to outliers |
| `std_diversity_score` | float | Diversity variance | High std = heterogeneous dataset |

---

### FastTextAnalyzer (`fasttext`)

**File**: `src/oumi/core/analyze/fasttext_analyzer.py`
**Dependencies**: fast-langdetect OR fasttext + huggingface_hub

| Metric | Type | Values | Description | Implementation | Limitations |
|--------|------|--------|-------------|----------------|-------------|
| `detected_language` | str | ISO 639-1 | Language code (en, es, fr, etc.) | fast-langdetect model prediction | Model download on first use; 176+ languages |
| `language_confidence` | float | 0-1 | Detection confidence | Model probability score | Low on short texts; unreliable for code-mixed |
| `language_name` | str | Full name | Human-readable language name | Lookup from language code | May not have all language names |
| `low_confidence` | bool | True/False | Below confidence threshold | True if confidence < threshold (default 0.5) | Threshold is configurable |
| `detected_script` | str | 11 scripts | Writing script type | Unicode range pattern matching | May miss rare scripts; mixed scripts problematic |
| `is_multilingual` | bool | True/False | Contains multiple languages | Sentence-level language detection with variation check | Sentence splitting may be imperfect |

**Detected Scripts**:
`latin`, `cyrillic`, `greek`, `arabic`, `hebrew`, `devanagari`, `cjk`, `hiragana`, `katakana`, `hangul`, `thai`, `unknown`

---

### IFDAnalyzer (`ifd`)

**File**: `src/oumi/core/analyze/ifd_analyzer.py`
**Dependencies**: transformers, torch

| Metric | Type | Range | Description | Implementation | Limitations |
|--------|------|-------|-------------|----------------|-------------|
| `score` | float | 0-∞ | Instruction-Following Difficulty ratio | PPL(response\|no instruction) / PPL(response\|with instruction) | IFD > 1 = helpful instruction; IFD < 1 = confusing |
| `ppl_with_instruction` | float | 1-∞ | Perplexity with instruction | Model perplexity on response given instruction | Lower = easier to predict |
| `ppl_without_instruction` | float | 1-∞ | Perplexity without instruction | Model perplexity on response alone | Baseline difficulty |
| `response_loss` | float | 0+ | Cross-entropy loss | Average token loss on response | Raw loss value for debugging |

**Interpretation**:

- **IFD > 10**: High-value instruction significantly helps the model
- **IFD 1-10**: Good instruction provides meaningful context
- **IFD < 1**: Instruction is confusing or misleading (response harder to predict with it)

**Limitations**:

- GPU strongly recommended (CPU is very slow)
- Model must be downloaded (~600MB for Qwen3-0.6B)
- Only meaningful for instruction-response pairs
- Results depend on model choice
- Slow for large datasets (~0.5-2 samples/sec on GPU)

---

### LLMJudgeAnalyzer (`llm_judge`)

**File**: `src/oumi/core/analyze/llm_judge_analyzer.py`
**Dependencies**: oumi inference engines (transformers, vLLM, or API clients)

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `score` | float | 0-10 | Quality score from LLM | Parsed from JSON response `score` field | Depends on prompt quality; may fail to parse |
| `label` | str | Variable | Classification label | Parsed from JSON response `label` field | Depends on prompt design |
| `reasoning` | str | Free text | Explanation from LLM | Parsed from JSON response `reasoning` field | May be empty if LLM doesn't explain |
| `raw_response` | str | Full response | Complete LLM output | Stored for debugging/analysis | Large storage for full responses |

**Prompt Presets**:

| Preset | Description | Use Case |
|--------|-------------|----------|
| `instruction_quality` | Evaluate instruction clarity for SFT | Data quality filtering |
| `response_quality` | Evaluate assistant response quality | Response filtering |
| `conversation_coherence` | Evaluate multi-turn conversation flow | Dialogue quality |
| `safety` | Check for safety concerns | Content moderation |
| `helpfulness` | Evaluate how helpful a response is | User satisfaction proxy |
| `factuality` | Check for factual accuracy | Misinformation filtering |

**Limitations**:

- API costs for each sample
- Latency depends on model/API (0.5-5 sec/sample)
- JSON parsing may fail on malformed responses
- Results are subjective and model-dependent
- Prompt engineering significantly affects quality

---

### EvolComplexityAnalyzer (`evol_complexity`)

**File**: `src/oumi/core/analyze/evol_complexity_analyzer.py`
**Dependencies**: oumi inference engines

| Metric | Type | Range | Description | Implementation | Limitations |
|--------|------|-------|-------------|----------------|-------------|
| `evol_complexity_score` | float | 0-1 | Normalized complexity score | Rank position among evolved variants | Relative to generated variants |
| `evol_complexity_rank` | int | 0-n | Absolute rank among variants | Position when sorted by complexity | 0 = simplest |
| `evol_complexity_headroom` | float | 0-1 | Improvement potential | 1 - (rank / num_variants) | How much more complex it could be |

**Evolution Operators**:

| Operator | Description | Effect |
|----------|-------------|--------|
| `add_constraints` | Add explicit constraints | Increases specificity |
| `require_reasoning` | Require step-by-step reasoning | Increases cognitive load |
| `increase_depth` | Make more complex | General complexity boost |
| `add_edge_cases` | Include edge cases | Increases thoroughness |
| `require_specificity` | Require specific details | Increases precision |
| `add_domain_knowledge` | Require domain expertise | Increases difficulty |

**Limitations**:

- API cost per sample (3-6 LLM calls per sample)
- Slow (~5-15 sec/sample)
- Quality depends on LLM capability
- Subjective complexity ranking
- Maximum 6 evolution variants

---

### EvolQualityAnalyzer (`evol_quality`)

**File**: `src/oumi/core/analyze/evol_quality_analyzer.py`
**Dependencies**: oumi inference engines

| Metric | Type | Range | Description | Implementation | Limitations |
|--------|------|-------|-------------|----------------|-------------|
| `evol_quality_score` | float | 0-1 | Normalized quality score | Rank position among evolved variants | Relative to generated variants |
| `evol_quality_rank` | int | 0-n | Absolute rank among variants | Position when sorted by quality | 0 = lowest quality |
| `evol_quality_headroom` | float | 0-1 | Improvement potential | 1 - (rank / num_variants) | How much better it could be |

**Same limitations as EvolComplexityAnalyzer**

---

### InstructRewardAnalyzer (`instruct_reward`)

**File**: `src/oumi/core/analyze/instruct_reward_analyzer.py`
**Dependencies**: None

Based on the Magpie/ArmoRM framework from "Fixing It in Post" paper, this analyzer scores response quality on a 0-5 scale.

| Metric | Type | Range | Description | Implementation | Limitations |
|--------|------|-------|-------------|----------------|-------------|
| `reward_score` | float | 0-5 | Overall quality score | Weighted combination of helpfulness, completeness, clarity, safety | Heuristic-based; doesn't use actual reward model |
| `reward_tier` | str | 4 tiers | Quality tier | poor (<2), fair (2-3), good (3-4), excellent (≥4) | Arbitrary thresholds |
| `helpfulness_score` | float | 0-1 | Addresses instruction | Pattern matching for helpful/unhelpful responses | May miss nuanced helpfulness |
| `completeness_score` | float | 0-1 | Response thoroughness | Based on word count and proper endings | Length-biased |
| `clarity_score` | float | 0-1 | Organization and clarity | Structure patterns (lists, headers, code blocks) | Favors formatted responses |

**Reward Score Weights**: helpfulness=0.3, completeness=0.25, clarity=0.2, safety=0.25

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_response_words` | int | 10 | Minimum words for quality response |
| `max_response_words` | int | 2000 | Maximum words before length penalty |
| `analyze_assistant_only` | bool | True | Only analyze assistant messages |
| `include_component_scores` | bool | True | Include individual dimension scores |

---

### InputQualityAnalyzer (`input_quality`)

**File**: `src/oumi/core/analyze/input_quality_analyzer.py`
**Dependencies**: None

Based on the Magpie framework, rates input/instruction quality from "very poor" to "excellent".

| Metric | Type | Range | Description | Implementation | Limitations |
|--------|------|-------|-------------|----------------|-------------|
| `input_quality_tier` | str | 5 tiers | Quality tier | very_poor (<0.2), poor (0.2-0.4), fair (0.4-0.6), good (0.6-0.8), excellent (≥0.8) | Arbitrary thresholds |
| `input_quality_score` | float | 0-1 | Overall quality score | Weighted combination of clarity, context, answerability | Heuristic-based |
| `is_ambiguous` | bool | True/False | Instruction is ambiguous | Count of ambiguous patterns (something, stuff, etc.) ≥2 | May flag legitimate use |
| `is_answerable` | bool | True/False | Can be meaningfully answered | Checks for greeting-only, too short, contradictory | May reject valid short queries |
| `has_sufficient_context` | bool | True/False | Enough context provided | Word count + context indicators (numbers, quotes, code) | Favors longer instructions |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_user_only` | bool | True | Only analyze user messages |
| `include_component_flags` | bool | True | Include individual quality flags |

---

### ConversationStructureAnalyzer (`conversation_structure`)

**File**: `src/oumi/core/analyze/conversation_structure_analyzer.py`
**Dependencies**: None

Analyzes conversation turn patterns. The paper found that Tulu is 95% single-turn vs SmolTalk 70% multi-turn.

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `turn_count` | int | Total turns in conversation | Count of all messages | Includes system messages |
| `user_turn_count` | int | Number of user messages | Count where role="user" | - |
| `assistant_turn_count` | int | Number of assistant messages | Count where role="assistant" | - |
| `is_single_turn` | bool | Single-turn conversation | turn_count ≤ threshold (default 2) | Threshold is configurable |
| `is_multi_turn` | bool | Multi-turn conversation | turn_count > threshold | - |
| `conversation_depth` | int | Complete exchanges | min(user_turns, assistant_turns) | Doesn't account for order |
| `role_balance` | float | User to assistant ratio | user_turns / (user + assistant) | 0.5 = balanced |
| `has_system_prompt` | bool | Has system message | Any role="system" | - |
| `avg_turn_length` | float | Average words per turn | Mean word count (excludes system) | Simple word split |
| `turn_length_variance` | float | Length variance | Statistical variance | May be high for Q&A format |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `single_turn_threshold` | int | 2 | Max messages for single-turn |
| `compute_length_stats` | bool | True | Compute length statistics |

---

### ResponseCompletenessAnalyzer (`response_completeness`)

**File**: `src/oumi/core/analyze/response_completeness_analyzer.py`
**Dependencies**: None

Detects truncated/incomplete/partial responses, a common issue in synthetic data.

| Metric | Type | Description | Implementation | Limitations |
|--------|------|-------------|----------------|-------------|
| `is_complete` | bool | Response is complete | score ≥ 0.7 (or strict: no truncation + natural ending) | Binary; misses partial issues |
| `completeness_score` | float | 0-1 completeness score | Penalties for truncation, unnatural endings | Heuristic-based |
| `ends_naturally` | bool | Natural ending | Ends with .!? or ``` or closing bracket | May miss valid endings |
| `has_conclusion` | bool | Has concluding statement | Patterns in last 20% (in conclusion, hope this helps) | Only for long responses |
| `truncation_type` | str | Type of truncation | mid_sentence, incomplete_list, incomplete_code, empty | May miss novel truncation |

**Truncation Detection:**

| Type | Detection | Penalty |
|------|-----------|---------|
| `mid_sentence` | Ends with connector (and, or, but, the) | -0.5 |
| `incomplete_code` | Unclosed code block or function | -0.4 |
| `incomplete_list` | Started list not finished | -0.3 |
| `empty` | No content | score = 0 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_assistant_only` | bool | True | Only analyze assistant messages |
| `strict_mode` | bool | False | Require natural endings |
| `include_truncation_type` | bool | True | Include truncation type |

---

## Analyzer Categories Summary

| Category | Analyzers | Dependencies | Speed | Use Case |
|----------|-----------|--------------|-------|----------|
| **Text Heuristics** | length, diversity, format, quality, training_quality, content_pattern | tiktoken, langdetect (opt) | Fast (~1000+ samples/sec) | Basic quality checks |
| **Task Classification** | task_category, safety, difficulty | None | Fast (~1000+ samples/sec) | Content categorization |
| **Cost Optimization** | cost | None | Fast | Training planning |
| **Embedding-Based** | embedding, question_diversity, repr_diversity | sentence-transformers, sklearn | Medium (~50-200 samples/sec GPU) | Duplicate/diversity analysis |
| **Language Detection** | fasttext | fast-langdetect | Fast (~500+ samples/sec) | Multilingual datasets |
| **Neural Scoring** | ifd | transformers, torch | Slow (~0.5-2 samples/sec GPU) | Instruction quality |
| **LLM-Based** | llm_judge, evol_complexity, evol_quality | oumi inference | Very Slow (~0.1-0.5 samples/sec) | High-quality evaluation |
| **"Fixing It in Post"** | instruct_reward, input_quality, conversation_structure, response_completeness | None | Fast (~1000+ samples/sec) | Data curation (Magpie framework) |

---

## Available Analyzers - Detailed Breakdown

---

### 1. LengthAnalyzer

**File**: `src/oumi/core/analyze/length_analyzer.py`
**ID**: `length`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_count` | bool | True | Count tokens |
| `tokenizer` | HF tokenizer | None | Custom HuggingFace tokenizer |
| `tiktoken_encoding` | str | "o200k_base" | tiktoken encoding (GPT-4o/GPT-5) |
| `include_special_tokens` | bool | True | Include special tokens (HF only) |

**Output Columns:** `{col}_length_token_count`

**Dependencies:** tiktoken (optional), transformers (optional)

---

### 2. DiversityAnalyzer

**File**: `src/oumi/core/analyze/diversity_analyzer.py`
**ID**: `diversity`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unique_words_ratio` | bool | True | Compute unique words / total words ratio |
| `case_sensitive` | bool | False | Case-sensitive word comparison |

**Output Columns:** `{col}_diversity_unique_words_ratio`

**Dependencies:** None

---

### 3. FormatAnalyzer

**File**: `src/oumi/core/analyze/format_analyzer.py`
**ID**: `format`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_markdown` | bool | True | Headers, lists, bold, italic, links |
| `detect_json` | bool | True | Fenced JSON and inline JSON |
| `detect_code_blocks` | bool | True | Fenced code blocks |
| `detect_urls` | bool | True | HTTP/HTTPS URLs |
| `detect_emails` | bool | False | Email addresses |
| `compute_complexity` | bool | True | Format complexity score (0-1) |

**Output Columns:** `{col}_format_has_markdown`, `{col}_format_has_json`, `{col}_format_has_code_blocks`, `{col}_format_code_block_count`, `{col}_format_code_block_languages`, `{col}_format_has_urls`, `{col}_format_format_complexity_score`

**Complexity Weights:** Markdown=2, JSON=2, Code=3+bonus, URLs=1, Emails=1

**Dependencies:** None

---

### 4. QualityAnalyzer

**File**: `src/oumi/core/analyze/quality_analyzer.py`
**ID**: `quality`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_pii` | bool | True | Master PII switch |
| `detect_emails` | bool | True | Email addresses |
| `detect_phones` | bool | True | Phone numbers |
| `detect_ssn` | bool | True | Social Security Numbers |
| `detect_credit_cards` | bool | True | Credit card numbers |
| `detect_api_keys` | bool | True | API keys and secrets |
| `detect_language` | bool | False | Language detection (requires langdetect) |
| `detect_encoding_issues` | bool | True | Mojibake patterns |
| `detect_repetition` | bool | True | Repetitive content |
| `repetition_ngram_size` | int | 3 | N-gram size |
| `repetition_threshold` | float | 0.3 | High repetition threshold |

**Output Columns:** `{col}_quality_has_pii`, `{col}_quality_pii_types`, `{col}_quality_pii_count`, `{col}_quality_detected_language`, `{col}_quality_language_confidence`, `{col}_quality_has_encoding_issues`, `{col}_quality_repetition_ratio`, `{col}_quality_has_high_repetition`

**Dependencies:** langdetect (optional, for language detection)

---

### 5. TrainingQualityAnalyzer

**File**: `src/oumi/core/analyze/training_quality_analyzer.py`
**ID**: `training_quality`

Analyzes assistant response quality for SFT datasets.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compute_response_completeness` | bool | True | Completeness metrics |
| `min_response_words` | int | 5 | Min response length |

**Output Columns:** `{col}_training_quality_response_completeness_score`, `{col}_training_quality_has_proper_ending`, `{col}_training_quality_has_structure`, `{col}_training_quality_response_word_count`

**Dependencies:** None

---

### 6. ContentPatternAnalyzer

**File**: `src/oumi/core/analyze/content_pattern_analyzer.py`
**ID**: `content_pattern`

Detects AI-specific quality issues in synthetic training data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_placeholders` | bool | True | `[Name]`, `<your_company>` |
| `detect_hallucinated_experiences` | bool | True | AI fabricated stories |
| `detect_nooutput` | bool | True | `<nooutput>`, `N/A` |
| `detect_refusals` | bool | True | "I cannot provide..." |
| `placeholder_whitelist` | list | None | Patterns to ignore |
| `check_output_only` | bool | False | Only assistant messages |

**Output Columns:** `{col}_content_pattern_has_placeholder`, `{col}_content_pattern_placeholder_count`, `{col}_content_pattern_placeholder_types`, `{col}_content_pattern_has_hallucinated_experience`, `{col}_content_pattern_has_nooutput`, `{col}_content_pattern_has_refusal`

**Dependencies:** None

---

### 7. TaskCategoryAnalyzer

**File**: `src/oumi/core/analyze/task_category_analyzer.py`
**ID**: `task_category`

**Task Categories:** `math`, `coding`, `information_seeking`, `creative_writing`, `editing`, `advice`, `reasoning`, `brainstorming`, `role_play`, `data_analysis`, `translation`, `other`

**Output Columns:** `{col}_task_category_task_category`, `{col}_task_category_task_confidence`, `{col}_task_category_is_stem`, `{col}_task_category_is_conversational`

**Dependencies:** None

---

### 8. SafetyAnalyzer

**File**: `src/oumi/core/analyze/safety_analyzer.py`
**ID**: `safety`

**Safety Categories:** `violence`, `hate`, `sexual`, `self_harm`, `illegal`, `deception`, `dangerous`, `privacy`

**Output Columns:** `{col}_safety_safety_score`, `{col}_safety_is_safe`, `{col}_safety_risk_level`, `{col}_safety_safety_categories`

**Dependencies:** None

---

### 9. DifficultyAnalyzer

**File**: `src/oumi/core/analyze/difficulty_analyzer.py`
**ID**: `difficulty`

**Domain Knowledge Domains:** `programming`, `math`, `science`, `legal`, `medical`, `finance`

**Output Columns:** `{col}_difficulty_difficulty_score`, `{col}_difficulty_difficulty_tier` (easy/medium/hard/expert), `{col}_difficulty_requires_reasoning`, `{col}_difficulty_requires_domain_knowledge`, `{col}_difficulty_constraint_count`

**Dependencies:** None

---

### 10. CostAnalyzer

**File**: `src/oumi/core/analyze/cost_analyzer.py`
**ID**: `cost`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_context_windows` | list[int] | [2048, 4096, 8192, 16384, 32768] | Context sizes |
| `compute_packing_efficiency` | bool | True | First-fit decreasing packing |
| `packing_overhead_tokens` | int | 10 | Separator tokens |

**Output Columns:** `cost_fits_context_{size}`, `cost_context_utilization_{size}`, `cost_tokens_wasted_{size}`

**Dataset Metrics:** `total_tokens`, `packing_efficiency_{size}`, `estimated_batches_{size}`

**Dependencies:** None

---

### 11. EmbeddingAnalyzer

**File**: `src/oumi/core/analyze/embedding_analyzer.py`
**ID**: `embedding`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "all-MiniLM-L6-v2" | Sentence-transformers model |
| `detect_duplicates` | bool | True | Semantic duplicates |
| `duplicate_threshold` | float | 0.95 | Cosine similarity threshold |
| `duplicate_scope` | str | "all" | "all", "by_role", "user", "assistant" |
| `detect_fuzzy_duplicates` | bool | False | MinHash LSH duplicates |
| `fuzzy_threshold` | float | 0.8 | Jaccard similarity |
| `cluster_samples` | bool | False | Enable clustering |
| `clustering_method` | str | "dbscan" | "dbscan" or "kmeans" |
| `batch_size` | int | 32 | Batch size |
| `device` | str | None | "cuda", "cpu", or auto |

**Output Columns:** `{col}_embedding_duplicate_group`, `{col}_embedding_has_semantic_duplicate`, `{col}_embedding_cluster`

**Dependencies:** sentence-transformers, scikit-learn, datasketch (optional)

---

### 12. QuestionDiversityAnalyzer

**File**: `src/oumi/core/analyze/question_diversity_analyzer.py`
**ID**: `question_diversity`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cluster_questions` | bool | True | Enable clustering |
| `clustering_method` | str | "dbscan" | "dbscan" or "kmeans" |
| `eps` | float | 0.15 | DBSCAN epsilon |
| `model_name` | str | "all-MiniLM-L6-v2" | Embedding model |
| `compute_entropy` | bool | True | Shannon entropy |
| `compute_concentration` | bool | True | Gini coefficient |

**Output Columns:** `{col}_question_diversity_cluster_id`, `{col}_question_diversity_cluster_size`, `{col}_question_diversity_is_concentrated`

**Dataset Metrics:** `num_question_clusters`, `diversity_ratio`, `question_entropy`, `question_gini`, `diversity_rating`

**Dependencies:** sentence-transformers, scikit-learn

---

### 13. ReprDiversityAnalyzer

**File**: `src/oumi/core/analyze/repr_diversity_analyzer.py`
**ID**: `repr_diversity`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "all-MiniLM-L6-v2" | Embedding model |
| `k_neighbors` | int | 5 | Number of nearest neighbors |
| `diversity_threshold` | float | 0.3 | Redundancy threshold |
| `embed_field` | str | "all" | "all", "user", "assistant" |

**Output Columns:** `{col}_repr_diversity_nn_distance`, `{col}_repr_diversity_score`, `{col}_repr_diversity_is_redundant`, `{col}_repr_diversity_percentile`

**Dependencies:** sentence-transformers, scikit-learn

---

### 14. FastTextAnalyzer

**File**: `src/oumi/core/analyze/fasttext_analyzer.py`
**ID**: `fasttext`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_language` | bool | True | Language detection |
| `detect_script` | bool | True | Writing script |
| `detect_multilingual` | bool | True | Multiple languages |
| `min_confidence` | float | 0.0 | Min confidence threshold |
| `low_confidence_threshold` | float | 0.5 | Flag low-confidence |

**Output Columns:** `{col}_fasttext_detected_language`, `{col}_fasttext_language_confidence`, `{col}_fasttext_language_name`, `{col}_fasttext_detected_script`, `{col}_fasttext_is_multilingual`

**Scripts Detected:** latin, cyrillic, greek, arabic, hebrew, devanagari, cjk, hiragana, katakana, hangul, thai

**Dependencies:** fast-langdetect OR fasttext + huggingface_hub

---

### 15. IFDAnalyzer

**File**: `src/oumi/core/analyze/ifd_analyzer.py`
**ID**: `ifd`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "Qwen/Qwen3-0.6B" | HuggingFace model |
| `device` | str | None | "cuda", "cpu", "mps", auto |
| `torch_dtype` | str | None | "float16", "bfloat16", "float32" |
| `batch_size` | int | 4 | Batch size |
| `max_length` | int | 2048 | Max sequence length |

**Output Columns:** `{id}_score` (IFD ratio), `{id}_ppl_with_instruction`, `{id}_ppl_without_instruction`, `{id}_response_loss`

**Metric:** IFD = PPL(response | no instruction) / PPL(response | with instruction)

**Dependencies:** transformers, torch

---

### 16. LLMJudgeAnalyzer

**File**: `src/oumi/core/analyze/llm_judge_analyzer.py`
**ID**: `llm_judge`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | None | Custom prompt with `{text}` |
| `prompt_preset` | str | None | Preset name |
| `inference_config` | dict | None | Inference engine config |
| `batch_size` | int | 10 | Batch size |
| `max_text_length` | int | 4000 | Max text length |
| `parse_json_response` | bool | True | Parse JSON |

**Prompt Presets:** `instruction_quality`, `response_quality`, `conversation_coherence`, `safety`, `helpfulness`, `factuality`

**Output Columns:** `{col}_llm_judge_score`, `{col}_llm_judge_label`, `{col}_llm_judge_reasoning`, `{col}_llm_judge_raw_response`

**Dependencies:** oumi inference engines

---

### 17. EvolComplexityAnalyzer

**File**: `src/oumi/core/analyze/evol_complexity_analyzer.py`
**ID**: `evol_complexity`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | "api" | "api" or "local" |
| `api_provider` | str | "anthropic" | "openai" or "anthropic" |
| `api_model` | str | "claude-4-5-haiku" | API model |
| `num_evolutions` | int | 3 | Number of evolved variants (1-6) |
| `analyze_role` | str | "user" | "user", "assistant", "all" |
| `temperature` | float | 0.7 | Generation temperature |

**Evolution Operators:** `add_constraints`, `require_reasoning`, `increase_depth`, `add_edge_cases`, `require_specificity`, `add_domain_knowledge`

**Output Columns:** `evol_complexity_score`, `evol_complexity_rank`, `evol_complexity_headroom`

**Dependencies:** oumi inference engines, LLM API access

---

### 18. InstructRewardAnalyzer

**File**: `src/oumi/core/analyze/instruct_reward_analyzer.py`
**ID**: `instruct_reward`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_response_words` | int | 10 | Minimum words for quality response |
| `max_response_words` | int | 2000 | Maximum words before length penalty |
| `analyze_assistant_only` | bool | True | Only analyze assistant messages |
| `include_component_scores` | bool | True | Include individual dimension scores |

**Output Columns:** `{col}_instruct_reward_score`, `{col}_instruct_reward_tier`, `{col}_instruct_reward_helpfulness`, `{col}_instruct_reward_completeness`, `{col}_instruct_reward_clarity`

**Reward Tiers:** poor (<2), fair (2-3), good (3-4), excellent (≥4)

**Dependencies:** None

---

### 19. InputQualityAnalyzer

**File**: `src/oumi/core/analyze/input_quality_analyzer.py`
**ID**: `input_quality`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_user_only` | bool | True | Only analyze user messages |
| `include_component_flags` | bool | True | Include individual quality flags |

**Output Columns:** `{col}_input_quality_tier`, `{col}_input_quality_score`, `{col}_input_quality_is_ambiguous`, `{col}_input_quality_is_answerable`, `{col}_input_quality_has_sufficient_context`

**Quality Tiers:** very_poor (<0.2), poor (0.2-0.4), fair (0.4-0.6), good (0.6-0.8), excellent (≥0.8)

**Dependencies:** None

---

### 20. ConversationStructureAnalyzer

**File**: `src/oumi/core/analyze/conversation_structure_analyzer.py`
**ID**: `conversation_structure`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `single_turn_threshold` | int | 2 | Max messages to consider single-turn |
| `compute_length_stats` | bool | True | Compute length statistics |

**Output Columns:** `conversation_structure_turn_count`, `conversation_structure_user_turn_count`, `conversation_structure_assistant_turn_count`, `conversation_structure_is_single_turn`, `conversation_structure_is_multi_turn`, `conversation_structure_conversation_depth`, `conversation_structure_role_balance`, `conversation_structure_has_system_prompt`, `conversation_structure_avg_turn_length`, `conversation_structure_turn_length_variance`

**Dependencies:** None

---

### 21. ResponseCompletenessAnalyzer

**File**: `src/oumi/core/analyze/response_completeness_analyzer.py`
**ID**: `response_completeness`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_assistant_only` | bool | True | Only analyze assistant messages |
| `strict_mode` | bool | False | Require natural endings for completeness |
| `include_truncation_type` | bool | True | Include type of truncation detected |

**Output Columns:** `{col}_response_completeness_is_complete`, `{col}_response_completeness_score`, `{col}_response_completeness_ends_naturally`, `{col}_response_completeness_has_conclusion`, `{col}_response_completeness_truncation_type`

**Truncation Types:** mid_sentence, incomplete_list, incomplete_code, empty

**Dependencies:** None

## CLI Usage

```bash
oumi analyze --config <config_file> [OPTIONS]
```

**Options:**

- `--config/-c`: Configuration file path (required)
- `--output/-o`: Output directory
- `--format/-f`: Output format: csv, json, parquet
- `--report/-r`: Generate interactive HTML report
- `--report-title`: Custom title for HTML report
- `--verbose/-v`: Verbose output with statistics

## Configuration Presets

1. **sft_quality** - Basic length, diversity, format, quality (default)
2. **sft_comprehensive** - All analyzers + training quality + cost
3. **sft_fast** - Fast heuristics only (no embeddings/LLM)

## Output Files

- `message_analysis.{csv|json|parquet}` - Message-level metrics
- `conversation_analysis.{csv|json|parquet}` - Conversation-level metrics
- `analysis_summary.json` - Statistics and recommendations
- `report.html` - Interactive HTML report (optional)

## Supporting Features - Detailed Breakdown

---

### Recommendations Engine

**File**: `src/oumi/core/analyze/recommendations.py` (2634 lines)

**Data Structures:**

```python
RecommendationCategory: WARNING | INSIGHT | SUGGESTION
RecommendationSeverity: HIGH | MEDIUM | LOW

@dataclass
class Recommendation:
    category, severity, title, description
    affected_samples, metric_name, threshold
    details, sample_indices (max 20)
```

**Default Thresholds:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `outlier_std_threshold` | 3.0 | Standard deviations for outliers |
| `duplicate_warn_threshold` | 5% | Duplicate fraction warning |
| `imbalance_threshold` | 80% | Role distribution imbalance |
| `empty_content_threshold` | 5 chars | Empty content detection |
| `short_content_threshold` | 10 words | Short content detection |
| `language_consistency_threshold` | 90% | Dominant language requirement |
| `pii_warn_threshold` | 1% | PII prevalence |
| `response_completeness_threshold` | 0.5 | Response completeness minimum |

**Check Categories:**

1. **Content Quality**: Outliers (multimodal-aware), duplicates, empty content, short content
2. **Distribution**: Role distribution, token lengths, conversation length
3. **Quality Analyzer**: Language consistency, format consistency, PII, encoding issues, repetition
4. **Training Quality**: Response completeness, truncated responses
5. **AI-Specific**: Placeholders, hallucinated experiences, nooutput markers, refusals
6. **Diversity**: Instruction diversity, concentrated clusters
7. **Advanced**: IFD scores, task categories, instruct reward, input quality, conversation structure, safety, difficulty

---

### Health Score Calculator

**File**: `src/oumi/core/analyze/health_score.py` (779 lines)

**Data Structures:**

```python
@dataclass
class HealthScoreComponent:
    name, score (0-100), weight, description, details

@dataclass
class DatasetHealthScore:
    overall (0-100), grade (A-F), components[]
    recommendations_count, high_severity_count, summary
```

**Component Weights (sum to 1.0):**

| Component | Weight | Description |
|-----------|--------|-------------|
| `training_quality` | 0.25 | Instruction clarity & response completeness |
| `diversity` | 0.15 | Vocabulary richness (TTR) |
| `quality` | 0.15 | PII, encoding, safety issues |
| `data_hygiene` | 0.15 | Duplicates, special tokens, encoding |
| `balance` | 0.10 | Role & conversation length balance |
| `consistency` | 0.10 | Language & format consistency |
| `length_distribution` | 0.10 | Token/word length distribution |

**Grade Thresholds:**

| Grade | Range | Label |
|-------|-------|-------|
| A | 90-100 | Excellent |
| B | 80-89 | Good |
| C | 70-79 | Fair |
| D | 60-69 | Poor |
| F | 0-59 | Critical |

**Penalty System:**

- Base penalty: 2.0 points per recommendation
- High severity penalty: Additional 5.0 points

**Key Scoring Features:**

- **Multimodal-Aware**: Detects bimodal distributions (e.g., short questions + long responses)
- **Within-Mode CV**: Scores coherence within each mode separately
- **Separation Bonus**: Rewards clear separation between modes

---

### HTML Report Generator

**File**: `src/oumi/core/analyze/report_generator.py` (1465 lines)
**Template**: `src/oumi/core/analyze/templates/report_template.html.jinja` (1827 lines)

**Initialization Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_charts` | True | Generate distribution charts |
| `include_tables` | True | Include statistics tables |
| `include_recommendations` | True | Show recommendations section |
| `include_anomaly_visualization` | True | Anomaly scatter plots |
| `include_health_score` | True | Health score visualization |
| `chart_height` | 400 | Chart height in pixels |
| `max_charts` | 10 | Maximum distribution charts |
| `outlier_std_threshold` | 3.0 | Outlier detection threshold |

**Output Structure:**

```
output_dir/
├── index.html          # Main report (lightweight)
└── data/
    ├── recommendations.json  # Recommendation samples
    ├── duplicates.json       # Duplicate group samples
    ├── clusters.json         # Cluster samples
    └── charts.json           # Plotly chart specs
```

**Visualizations:**

1. **Health Score Ring**: Animated circular progress (0-100) with grade
2. **Component Breakdown**: Individual scores with color-coded bars
3. **Distribution Histograms**: With multimodal detection and mode annotations
4. **Role Distribution**: Donut chart for user/assistant/system
5. **Anomaly Scatter Plots**: Outlier highlighting with categories
6. **IFD Distribution**: Log-scale with 3-tier coloring (High >10, Good 1-10, Low <1)
7. **Quality Score Distribution**: Good vs low quality overlay

**Interactive Features:**

- **Recommendations Modal**: Slide-in panel with full conversation context
- **Expandable Groups**: Duplicate groups and cluster samples
- **Lazy Loading**: External JSON files loaded on-demand
- **Dark Theme**: Sophisticated color palette with accent colors

**Design System:**

- Fonts: Instrument Serif (display), Source Sans 3 (body), JetBrains Mono (code)
- Colors: Beige accent (#d4a574), Success (#7cb97c), Warning (#e0b854), Danger (#d66a6a)
- Responsive: Desktop, tablet (900px), mobile breakpoints

## Recent Additions (Dec 2024)

1. **FastText analyzer** - Fast 176+ language detection with script detection
2. **"Fixing It in Post" analyzers** (Magpie framework):
   - `instruct_reward` - Response quality scoring (0-5 scale)
   - `input_quality` - Instruction quality rating (5 tiers)
   - `response_completeness` - Truncation/incompleteness detection
   - `conversation_structure` - Single/multi-turn analysis
3. **DEITA support** - EvolComplexity and EvolQuality analyzers
4. **Health score system** - Composite quality grading (A-F)
5. **Recommendations engine** - 27 automated check categories
6. **Total analyzers**: 22 sample analyzers across 8 categories

## Key Files

- CLI: `src/oumi/cli/analyze.py`
- Config: `src/oumi/core/configs/analyze_config.py`
- Main orchestrator: `src/oumi/core/analyze/dataset_analyzer.py`
- DataFrame analyzer: `src/oumi/core/analyze/dataframe_analyzer.py`
- Analyzers: `src/oumi/core/analyze/*_analyzer.py` (22 analyzer files)
- Supporting: `src/oumi/core/analyze/{recommendations,health_score,report_generator}.py`
- Presets: `src/oumi/core/analyze/presets.py`
- Example configs: `configs/examples/analyze/` (analysis recipe examples)
