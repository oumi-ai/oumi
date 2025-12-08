# Fixing It in Post: Dataset Quality Analysis for LLM Post-Training

This document summarizes the key methods from the paper ["Fixing It in Post: A Systematic Study of Open-Source Post-Training Datasets"](https://arxiv.org/html/2506.06522v2) and describes how they were implemented in the Oumi Analyze framework.

## Paper Overview

The paper provides a systematic comparison of open-source post-training datasets (Tulu-3-SFT-Mix and SmolTalk) and introduces a quality-based curation methodology. Key finding: **808k curated samples outperformed 1M+ samples across 14 benchmarks**, demonstrating that quality-aware curation is more important than dataset size.

---

## Quality Dimensions from the Paper

The paper uses the Magpie framework to annotate ~2M samples across multiple quality dimensions:

| Dimension | Description | Scale |
|-----------|-------------|-------|
| **Task Category** | Classification into 12 task types | Categorical |
| **Input Quality** | Rating of instruction quality | very poor → excellent |
| **Instruct Reward** | Response quality score | 0-5 continuous |
| **Safety** | Llama-Guard 2 assessment | safe/unsafe |
| **Difficulty** | Sample difficulty estimation | Categorical |
| **Turn Structure** | Single vs multi-turn analysis | Binary + counts |

---

## Our Implementation

We implemented 7 new analyzers in `src/oumi/core/analyze/` that capture these quality dimensions using heuristic-based approaches (no external model dependencies required).

### 1. TaskCategoryAnalyzer (`task_category`)

**Purpose**: Classify instructions into task categories to understand dataset composition and identify task distribution imbalances.

**Method**: Pattern-based classification using regex matching against category-specific vocabulary.

**Categories Supported**:
- `math` - Mathematical problems, calculations, proofs
- `coding` - Programming tasks, debugging, code generation
- `information_seeking` - Factual questions, definitions, explanations
- `creative_writing` - Stories, poems, creative content
- `editing` - Text editing, rewriting, grammar correction
- `advice` - Personal advice, recommendations
- `reasoning` - Logical reasoning, analysis, problem-solving
- `brainstorming` - Idea generation, planning, lists
- `role_play` - Character personas, simulated scenarios
- `data_analysis` - Data processing, statistics, visualization
- `translation` - Language translation tasks
- `other` - Tasks that don't fit other categories

**Metrics Produced**:
```
{column}_task_category_category      # Primary category
{column}_task_category_confidence    # Confidence score (0-1)
{column}_task_category_is_stem       # Boolean: math/coding/data_analysis
{column}_task_category_is_conversational  # Boolean: advice/role_play/brainstorming
```

**Implementation Details**:
- Each category has multiple regex patterns capturing domain-specific vocabulary
- Confidence is calculated as the proportion of matches for the best category
- Minimum confidence threshold (default 0.3) prevents low-confidence classifications
- By default, only analyzes user messages (instructions)

**File**: `src/oumi/core/analyze/task_category_analyzer.py`

---

### 2. InstructRewardAnalyzer (`instruct_reward`)

**Purpose**: Score response quality similar to the Magpie/ArmoRM framework (0-5 scale).

**Method**: Heuristic scoring across multiple quality dimensions, combined with configurable weights.

**Quality Dimensions**:
| Dimension | Weight | What it Measures |
|-----------|--------|------------------|
| Helpfulness | 30% | Does the response address the instruction? |
| Completeness | 25% | Is the response thorough and complete? |
| Clarity | 20% | Is the response well-organized and clear? |
| Safety | 25% | Is the response appropriate and safe? |

**Scoring Logic**:

1. **Helpfulness** (0-1):
   - Starts at 0.5 (neutral)
   - +0.15 for helpful opening patterns ("Here is...", "Let me...")
   - -0.3 for unhelpful patterns ("I don't know", "N/A")

2. **Completeness** (0-1):
   - Based on word count relative to configurable min/max thresholds
   - +0.1 for proper sentence endings (., !, ?)
   - -0.2 for trailing ellipsis (...)

3. **Clarity** (0-1):
   - +0.1 for each structural element (bullets, headers, code blocks)
   - +0.2 for optimal sentence length (10-25 words)
   - -0.1 for excessive hedging language

4. **Safety** (0-1):
   - Starts at 1.0 (safe)
   - -0.1 for each unsafe pattern match

**Metrics Produced**:
```
{column}_instruct_reward_score        # Overall score (0-5)
{column}_instruct_reward_tier         # poor/fair/good/excellent
{column}_instruct_reward_helpfulness  # Component score (0-1)
{column}_instruct_reward_completeness # Component score (0-1)
{column}_instruct_reward_clarity      # Component score (0-1)
```

**Tier Thresholds**:
- `excellent`: score >= 4.0
- `good`: score >= 3.0
- `fair`: score >= 2.0
- `poor`: score < 2.0

**File**: `src/oumi/core/analyze/instruct_reward_analyzer.py`

---

### 3. InputQualityAnalyzer (`input_quality`)

**Purpose**: Rate instruction/input quality from "very poor" to "excellent".

**Method**: Multi-dimensional assessment of instruction clarity, completeness, and answerability.

**Quality Signals**:

1. **Clarity Score**:
   - +0.2 for clear imperative patterns ("Write...", "Explain...", "Calculate...")
   - +0.2 for clear question structures ("What is...", "How do...")
   - -0.1 for each ambiguous term ("something", "stuff", "kind of")

2. **Ambiguity Detection**:
   - Flags instructions with 2+ vague/ambiguous patterns
   - Patterns: "something", "stuff", "things", "whatever", "kind of", "sort of"

3. **Answerability Check**:
   - Returns False for greetings ("hi", "hello")
   - Returns False for acknowledgments ("thanks", "ok")
   - Returns False for very short inputs (< 2 words)

4. **Context Sufficiency**:
   - Requires minimum word count (default: 5 words)
   - Checks for context indicators (numbers, quotes, code, named entities)

**Metrics Produced**:
```
{column}_input_quality_tier                  # very_poor/poor/fair/good/excellent
{column}_input_quality_score                 # Overall score (0-1)
{column}_input_quality_is_ambiguous          # Boolean
{column}_input_quality_is_answerable         # Boolean
{column}_input_quality_has_sufficient_context # Boolean
```

**Tier Thresholds**:
- `excellent`: score >= 0.8
- `good`: score >= 0.6
- `fair`: score >= 0.4
- `poor`: score >= 0.2
- `very_poor`: score < 0.2

**File**: `src/oumi/core/analyze/input_quality_analyzer.py`

---

### 4. ConversationStructureAnalyzer (`conversation_structure`)

**Purpose**: Analyze conversation patterns and turn dynamics.

**Paper Finding**: Tulu-3 is 95% single-turn vs SmolTalk 70% multi-turn, which significantly impacts model behavior.

**Method**: Group messages by conversation_id and compute structural metrics.

**Metrics Produced**:
```
{analyzer_id}_turn_count           # Total messages in conversation
{analyzer_id}_user_turn_count      # Number of user messages
{analyzer_id}_assistant_turn_count # Number of assistant messages
{analyzer_id}_is_single_turn       # Boolean (turns <= 2)
{analyzer_id}_is_multi_turn        # Boolean (turns > 2)
{analyzer_id}_conversation_depth   # Number of complete exchanges
{analyzer_id}_role_balance         # user_turns / (user + assistant) ratio
{analyzer_id}_has_system_prompt    # Boolean
{analyzer_id}_avg_turn_length      # Average words per turn
{analyzer_id}_turn_length_variance # Variance in turn lengths
```

**Implementation Details**:
- Groups by `conversation_id` column if present
- Falls back to flat format handling (each row = single conversation)
- Excludes system messages from length statistics
- Role balance of 0.5 indicates perfect balance

**File**: `src/oumi/core/analyze/conversation_structure_analyzer.py`

---

### 5. SafetyAnalyzer (`safety`)

**Purpose**: Detect potentially unsafe content across multiple safety categories.

**Paper Method**: Llama-Guard 2 for safety assessment.

**Our Method**: Heuristic pattern matching across 7 safety categories (model-free).

**Safety Categories**:
| Category | Weight | Examples |
|----------|--------|----------|
| `violence` | 0.90 | kill, murder, attack, weapon, bomb |
| `hate` | 0.85 | hate speech, discrimination, slurs |
| `self_harm` | 0.95 | suicide, self-harm, overdose |
| `illegal` | 0.80 | hack, steal, drug dealing, fraud |
| `dangerous` | 0.90 | make bomb, synthesize chemicals |
| `privacy` | 0.70 | dox, reveal personal info, SSN |
| `deception` | 0.75 | scam, phishing, misinformation |

**Scoring Logic**:
1. Each category has multiple regex patterns
2. Match count per category determines category score: `1.0 - (matches * 0.3)`
3. Overall safety score is weighted average of category scores
4. `is_safe` threshold: score >= 0.7 (configurable via strict_mode)

**Risk Levels**:
- `safe`: score >= 0.9
- `low`: score >= 0.7
- `medium`: score >= 0.5
- `high`: score < 0.5

**Metrics Produced**:
```
{column}_safety_score       # Overall safety (0-1, higher = safer)
{column}_safety_is_safe     # Boolean
{column}_safety_risk_level  # safe/low/medium/high
{column}_safety_categories  # Comma-separated flagged categories
```

**File**: `src/oumi/core/analyze/safety_analyzer.py`

---

### 6. DifficultyAnalyzer (`difficulty`)

**Purpose**: Estimate sample difficulty for curriculum learning and dataset stratification.

**Method**: Multi-signal heuristic analysis combining instruction complexity, domain requirements, and structural indicators.

**Difficulty Signals**:

1. **Reasoning Requirements**:
   - Patterns: "why", "explain why", "step-by-step", "compare", "contrast"
   - Conditional language: "if", "assuming", "given that"
   - Requires 2+ matches to flag as requiring reasoning

2. **Domain Knowledge Detection**:
   - Programming: algorithm, API, database, async, recursion
   - Math: theorem, derivative, integral, probability
   - Science: hypothesis, molecule, quantum, genome
   - Legal: statute, liability, jurisdiction, precedent
   - Medical: diagnosis, treatment, pathology, prognosis
   - Finance: portfolio, derivative, valuation, hedge

3. **Constraint Counting**:
   - Requirement words: must, should, required, mandatory
   - Quantifiers: at least, at most, exactly, maximum, minimum
   - Exclusions: without, except, avoid, don't

4. **Multi-Part Detection**:
   - Numbered lists: "1)", "2)", "a)", "b)"
   - Sequence words: first, second, additionally, furthermore

**Score Calculation**:
```python
score = 0.3  # Base difficulty
score += 0.15 if word_count > 100 else (0.1 if word_count > 50 else 0)
score += min(0.2, constraint_count * 0.05)
score += 0.15 if requires_reasoning
score += min(0.2, len(domains) * 0.1)
score += 0.1 if is_multi_part
```

**Tier Thresholds**:
- `expert`: score >= 0.75
- `hard`: score >= 0.5
- `medium`: score >= 0.3
- `easy`: score < 0.3

**Metrics Produced**:
```
{column}_difficulty_score                  # Overall difficulty (0-1)
{column}_difficulty_tier                   # easy/medium/hard/expert
{column}_difficulty_requires_reasoning     # Boolean
{column}_difficulty_requires_domain_knowledge # Boolean
{column}_difficulty_constraint_count       # Integer
```

**File**: `src/oumi/core/analyze/difficulty_analyzer.py`

---

### 7. ResponseCompletenessAnalyzer (`response_completeness`)

**Purpose**: Detect truncated, partial, or incomplete responses (common in synthetic data).

**Method**: Pattern-based detection of truncation indicators combined with structural analysis.

**Truncation Types Detected**:

1. **Mid-Sentence Truncation**:
   - Ends with connector words: "and", "but", "the", "to", "because"
   - Ends with incomplete phrases: "such as", "for example", "e.g."
   - Ends with ellipsis: "..."
   - Ends with comma/colon expecting more content

2. **Incomplete Lists**:
   - Numbered list that ends abruptly
   - "First..." without "second" or "finally"

3. **Incomplete Code**:
   - Unclosed code blocks (``` without closing)
   - Unclosed function/class definitions
   - Unclosed brackets in code

**Natural Ending Detection**:
- Proper punctuation: `.`, `!`, `?`
- Closed code blocks: \`\`\`
- Closing brackets for structured data: `}`, `]`, `)`

**Conclusion Detection**:
- Scans last 20% of text for conclusion patterns
- Patterns: "in conclusion", "to summarize", "hope this helps", "let me know"

**Score Calculation**:
```python
score = 1.0  # Start complete
score -= 0.5 if mid_sentence_truncation
score -= 0.4 if incomplete_code
score -= 0.3 if incomplete_list
score += 0.1 if ends_naturally else -0.2
score += 0.1 if has_conclusion and word_count > 50
score -= 0.3 if word_count < 5  # Very short
```

**Metrics Produced**:
```
{column}_response_completeness_is_complete     # Boolean
{column}_response_completeness_score           # Completeness score (0-1)
{column}_response_completeness_ends_naturally  # Boolean
{column}_response_completeness_has_conclusion  # Boolean
{column}_response_completeness_truncation_type # mid_sentence/incomplete_list/incomplete_code/empty
```

**File**: `src/oumi/core/analyze/response_completeness_analyzer.py`

---

## Recommendations Integration

Each analyzer has corresponding recommendation checks in `src/oumi/core/analyze/recommendations.py`:

| Analyzer | Recommendation Check | Severity Triggers |
|----------|---------------------|-------------------|
| TaskCategory | `_check_task_category_distribution` | >50% single category, missing important categories |
| InstructReward | `_check_instruct_reward_scores` | >10% low-quality (score < 2.5) |
| InputQuality | `_check_input_quality_issues` | >10% poor/very_poor inputs |
| ConversationStructure | `_check_conversation_structure` | >90% single-turn (insight) |
| Safety | `_check_safety_issues` | Any unsafe content (>5% = HIGH) |
| Difficulty | `_check_difficulty_distribution` | >70% easy or >70% hard (skewed) |
| ResponseCompleteness | `_check_response_completeness_analyzer` | >5% incomplete responses |

---

## Usage Example

```python
from oumi.core.analyze import (
    DatasetAnalyzer,
    TaskCategoryAnalyzer,
    InstructRewardAnalyzer,
    InputQualityAnalyzer,
    ConversationStructureAnalyzer,
    SafetyAnalyzer,
    DifficultyAnalyzer,
    ResponseCompletenessAnalyzer,
)

# Create analyzer with new analyzers
analyzer = DatasetAnalyzer(
    dataset_name="my_dataset",
    plugins=[
        TaskCategoryAnalyzer(),
        InstructRewardAnalyzer(),
        InputQualityAnalyzer(),
        ConversationStructureAnalyzer(),
        SafetyAnalyzer(),
        DifficultyAnalyzer(),
        ResponseCompletenessAnalyzer(),
    ],
)

# Run analysis
analyzer.analyze_dataset(dataset)

# Get recommendations
from oumi.core.analyze import RecommendationsEngine
engine = RecommendationsEngine()
recommendations = engine.generate_recommendations(
    analyzer.message_df,
    analyzer.conversation_df,
    analyzer.analysis_summary,
)

# Generate report
from oumi.core.analyze import HTMLReportGenerator
generator = HTMLReportGenerator()
generator.generate_report(analyzer, output_path="report/")
```

---

## Key Insights from the Paper

1. **Quality > Quantity**: 808k curated samples outperformed 1M+ samples
2. **Task Distribution Matters**: Imbalanced task distribution impacts capabilities
3. **Turn Structure Impact**: Single-turn vs multi-turn ratio affects dialogue abilities
4. **Input Quality Correlation**: 80%+ high-quality inputs correlate with better training
5. **Response Quality**: Instruct reward scores are the core curation signal

---

## References

- Paper: ["Fixing It in Post: A Systematic Study of Open-Source Post-Training Datasets"](https://arxiv.org/html/2506.06522v2)
- Magpie Framework: Used for quality annotation at scale
- ArmoRM: Reward model for response quality scoring
- Llama-Guard 2: Safety assessment model
