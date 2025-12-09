# Automatic Data Quality Analysis in Oumi

## Overview

Oumi implements a comprehensive suite of automated analyzers to assess and improve training data quality. These analyzers detect issues ranging from PII contamination to semantic duplicates, enabling data-driven curation for better model performance. Research shows that selecting just 5-10% of high-quality data can match full-dataset performance while reducing training costs.

## Key Research Papers

1. **DEITA** - "What Makes Good Data for Alignment?" (Liu et al., 2023) - https://arxiv.org/abs/2312.15685
   - Introduces diversity-based filtering and complexity scoring via evolved variants
2. **Cherry LLM & Superfiltering** - https://arxiv.org/html/2402.00530v1
   - Instruction-Following Difficulty (IFD) metric for data quality assessment
3. **Fixing It in Post** - Llama-Guard inspired safety framework
   - Safety scoring and response quality metrics

## Prototyped Analyzers

| Analyzer | Metrics | Requires LLM | Usefulness | Use Case |
|----------|---------|--------------|------------|----------|
| **QualityAnalyzer** | `has_pii`, `pii_types`, `pii_count`, `detected_language`, `language_confidence`, `has_encoding_issues`, `repetition_ratio`, `has_high_repetition` | No | ⭐⭐⭐⭐⭐ | Critical for filtering datasets with privacy issues, encoding corruption, or low-quality repetitive text |
| **EmbeddingAnalyzer** | `duplicate_group`, `has_semantic_duplicate`, `fuzzy_duplicate_group`, `fuzzy_jaccard_score`, `cluster` | No (embeddings) | ⭐⭐⭐⭐⭐ | Essential for deduplication; reduces redundancy and prevents overfitting on repeated examples |
| **ReprDiversityAnalyzer** | `nn_distance`, `diversity_score`, `is_redundant`, `diversity_percentile` | No (embeddings) | ⭐⭐⭐⭐⭐ | DEITA paper: Select diverse samples that maximize information content; proven to reduce data needs by 90% |
| **IFDAnalyzer** | `ifd_score`, `ppl_with_instruction`, `ppl_without_instruction`, `response_loss` | Yes (local) | ⭐⭐⭐⭐⭐ | Cherry LLM: Identify where instructions add value; weak models can filter for strong models; IFD<1.0 signals bad pairs |
| **EvolComplexityAnalyzer** | `evol_complexity_score`, `evol_complexity_rank`, `evol_complexity_headroom` | Yes (API/local) | ⭐⭐⭐⭐ | DEITA: Score instruction complexity via ranking; enables curriculum learning and filtering simple tasks |
| **SafetyAnalyzer** | `safety_score`, `is_safe`, `risk_level`, `safety_categories` | No | ⭐⭐⭐⭐⭐ | Llama-Guard inspired: Detect violence, hate, self-harm, illegal content; critical for deployment safety |
| **DifficultyAnalyzer** | `difficulty_score`, `difficulty_tier`, `requires_reasoning`, `requires_domain_knowledge`, `constraint_count` | No | ⭐⭐⭐⭐ | Identify challenging samples for curriculum learning; filter trivial examples |
| **InstructRewardAnalyzer** | `reward_score`, `reward_tier`, `helpfulness_score`, `completeness_score`, `clarity_score` | No | ⭐⭐⭐⭐ | Magpie/ArmoRM framework: Quality assessment across multiple dimensions |
| **ContentPatternAnalyzer** | `has_placeholder`, `placeholder_count`, `has_hallucinated_experience`, `has_nooutput`, `has_refusal` | No | ⭐⭐⭐⭐ | Detect AI-generated dataset artifacts; filter templates, refusals, and fabricated stories |
| **TrainingQualityAnalyzer** | `response_completeness_score`, `has_proper_ending`, `has_structure`, `response_word_count` | No | ⭐⭐⭐⭐ | SFT quality: Detect truncated responses and ensure well-formed training examples |
| **LLMJudgeAnalyzer** | `llm_judge_score`, `llm_judge_label`, `llm_judge_reasoning` | Yes (API) | ⭐⭐⭐⭐ | Custom evaluation with GPT-4o-mini; flexible but expensive; 6 preset prompts available |
| **FormatAnalyzer** | `has_markdown`, `has_json_blocks`, `has_code_blocks`, `code_block_count`, `format_complexity_score` | No | ⭐⭐⭐ | Detect structured content; useful for filtering code/technical data |
| **LengthAnalyzer** | `token_count` | No | ⭐⭐⭐ | Basic filtering by length; essential for managing context windows |
| **DiversityAnalyzer** | `unique_words_ratio` | No | ⭐⭐ | Simple vocabulary diversity; limited value compared to embedding-based methods |
