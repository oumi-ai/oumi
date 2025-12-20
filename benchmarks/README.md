# Dataset Pipeline Benchmarks

This directory contains benchmarks for profiling the Oumi dataset loading pipeline.

## Scripts

### 1. `dataset_pipeline_benchmark.py` - Full Pipeline Benchmark

Comprehensive benchmark that tests 10 diverse datasets across the entire loading pipeline.

**What it measures:**

- Raw HuggingFace load time (baseline)
- Oumi dataset loading (via registry or GenericSftDataset)
- `to_hf()` conversion time (map and iterable modes)
- Iteration throughput
- Sampling performance
- Oversampling performance (the deep copy bottleneck)
- Peak memory usage for each phase

**Datasets tested:**

| Key | Dataset | Type | Description |
|-----|---------|------|-------------|
| `alpaca_small` | yahma/alpaca-cleaned | SFT | Alpaca instruction format |
| `sharegpt_medium` | ShareGPT_Vicuna | SFT | Multi-turn conversations |
| `openhermes_large` | teknium/OpenHermes-2.5 | SFT | Large instruction dataset |
| `ultrafeedback_dpo` | HuggingFaceH4/ultrafeedback | DPO | Preference tuning |
| `code_alpaca` | sahil2801/CodeAlpaca-20k | SFT | Code instructions |
| `metamath` | meta-math/MetaMathQA | SFT | Math reasoning |
| `wikitext` | Salesforce/wikitext | Pretraining | Raw text |
| `aya_multilingual` | CohereForAI/aya_dataset | SFT | Multilingual |
| `longalpaca` | Yukang/LongAlpaca-12k | SFT | Long context |
| `glaive_function` | glaiveai/glaive-function-calling | SFT | Function calling |

**Usage:**

```bash
# Full benchmark (takes ~30-60 minutes)
python benchmarks/dataset_pipeline_benchmark.py --output baseline_results.json

# Quick mode (smaller samples, ~10-15 minutes)
python benchmarks/dataset_pipeline_benchmark.py --quick --output baseline_quick.json

# Specific datasets only
python benchmarks/dataset_pipeline_benchmark.py --datasets alpaca_small,code_alpaca

# Save baseline before optimizations
python benchmarks/dataset_pipeline_benchmark.py --output baseline.json

# Save results after optimizations
python benchmarks/dataset_pipeline_benchmark.py --output optimized.json
```

---

### 2. `dataset_microbenchmarks.py` - Targeted Bottleneck Benchmarks

Micro-benchmarks focusing on specific identified bottlenecks.

**Benchmarks included:**

1. **Oversampling Strategies** - Compares:
   - Current: `copy.deepcopy()` (O(n*k) memory)
   - Proposed: Lazy index mapping (O(1) memory)
   - Alternative: HuggingFace `select()` with repeated indices

2. **Feature Detection Sampling** - Compares:
   - Current: Sample 1/8 of dataset
   - Proposed: Fixed 100 samples
   - Minimal: Fixed 10 samples

3. **DataFrame Conversion** - Compares:
   - Current: HF → pandas → process → HF
   - Proposed: Direct HF `map()`
   - Alternative: Generator approach

4. **Converter Auto-Detection** - Measures:
   - Detection time per format
   - Cached vs uncached lookup

5. **Mixture Operations** - Compares:
   - Concatenate
   - Interleave (equal probability)
   - Interleave (weighted)

**Usage:**

```bash
# Run all micro-benchmarks
python benchmarks/dataset_microbenchmarks.py

# Run specific benchmark
python benchmarks/dataset_microbenchmarks.py --benchmark oversampling
python benchmarks/dataset_microbenchmarks.py --benchmark feature_detection
python benchmarks/dataset_microbenchmarks.py --benchmark dataframe
python benchmarks/dataset_microbenchmarks.py --benchmark converter
python benchmarks/dataset_microbenchmarks.py --benchmark mixture
```

---

### 3. `compare_benchmarks.py` - Result Comparison Tool

Compare two benchmark runs and generate a diff report.

**Usage:**

```bash
# Compare baseline vs optimized
python benchmarks/compare_benchmarks.py baseline.json optimized.json

# Generate markdown report
python benchmarks/compare_benchmarks.py baseline.json optimized.json --output comparison_report.md

# Show all metrics (including unchanged)
python benchmarks/compare_benchmarks.py baseline.json optimized.json --show-all
```

**Output:**

- Console table showing metric changes with improvement indicators
- Overall improvement summary
- Optional markdown report for documentation

---

## Workflow for Optimization

1. **Establish baseline:**

   ```bash
   python benchmarks/dataset_pipeline_benchmark.py --output baseline.json
   python benchmarks/dataset_microbenchmarks.py > microbenchmark_baseline.txt
   ```

2. **Identify bottlenecks:** Review microbenchmark results to prioritize optimizations.

3. **Implement optimizations:** Make changes to the codebase.

4. **Re-run benchmarks:**

   ```bash
   python benchmarks/dataset_pipeline_benchmark.py --output optimized.json
   ```

5. **Compare results:**

   ```bash
   python benchmarks/compare_benchmarks.py baseline.json optimized.json --output report.md
   ```

---

## Key Metrics to Watch

| Metric | Target | Why |
|--------|--------|-----|
| `load_raw_duration_s` | Lower is better | Raw loading overhead |
| `to_hf_map_duration_s` | Lower is better | Transform pipeline cost |
| `to_hf_map_peak_memory_mb` | Lower is better | Memory efficiency |
| `oversampling_duration_s` | 10x improvement | Deep copy is expensive |
| `oversampling_peak_memory_mb` | 5x reduction | Should be O(1) not O(n*k) |
| `examples_per_second` | Higher is better | Overall throughput |

---

## Known Bottlenecks (Pre-Optimization)

Based on code review, these are the identified bottlenecks:

1. **Oversampling** (`builders/data.py:209-225`): Uses `copy.deepcopy()` for each copy, causing O(n*k) memory usage.

2. **Feature Detection** (`base_map_dataset.py:291-299`): Samples 1/8 of dataset to detect features, expensive for large datasets.

3. **DataFrame Intermediary** (`base_map_dataset.py:456-503`): Converts HF → pandas → HF, causing double memory usage.

4. **Converter Auto-Detection**: Runs for every dataset load, even when converter could be cached.
