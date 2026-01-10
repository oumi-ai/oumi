# Special Token Detection Enhancement

## Summary

Enhanced the `EmptyContentAnalyzer` to detect special tokens that indicate data quality issues or non-text output placeholders.

## Changes Made

### 1. Added Special Token Detection to EmptyContentAnalyzer

**New Parameters**:
- `error_tokens`: List of tokens indicating data errors (e.g., `["nan", "<noinput>"]`)
- `placeholder_tokens`: List of acceptable placeholder tokens (e.g., `["<nooutput>"]`)

**New Output Columns**:
- `{column}_contains_error_token`: Boolean flag for error token presence
- `{column}_error_token_type`: Which error token was found (if any)
- `{column}_contains_placeholder_token`: Boolean flag for placeholder token presence
- `{column}_placeholder_token_type`: Which placeholder token was found (if any)

### 2. Configuration Update

Updated `quality_comprehensive.yaml`:
```yaml
- id: empty_content
  params:
    min_content_length: 10
    error_tokens: ["nan", "<noinput>"]  # Data quality issues
    placeholder_tokens: ["<nooutput>"]  # Acceptable non-text output markers
```

### 3. Test Coverage

Added 5 new tests:
- `test_detects_error_tokens`: Verifies error token detection
- `test_detects_placeholder_tokens`: Verifies placeholder token detection
- `test_detects_both_token_types`: Verifies simultaneous detection
- `test_no_tokens_configured`: Verifies backward compatibility
- Updated existing tests to match new return signature

**All 7 tests passing** ✅

## Token Analysis from Alpaca Dataset

### Error Tokens (Should be Removed)

**1. "nan" - 28 occurrences**
- **Issue**: Missing responses in legitimate questions
- **Examples**:
  - "Convert 20 inches to centimetres" → "nan" (should be "50.8")
  - "Calculate sum of digits in 18047" → "nan" (should be "20")
  - "Standardize date July 26th, 2021 to ISO 8601" → "nan" (should be "2021-07-26")
- **Root Cause**: Data pipeline conversion errors (pandas NaN → string "nan")
- **Action**: ❌ **REMOVE or REGENERATE** - These teach the model bad behavior

**2. "<noinput>" - 22 occurrences**
- **Issue**: Wrong field - token appears in output instead of input
- **Examples**:
  - "Create a new logo for Ferris Consulting" → "<noinput>"
  - "Design a 3D model for modern house" → "<noinput>"
  - "Send text message to John" → "<noinput>"
- **Root Cause**: Token meant for input field mistakenly in output field
- **Action**: ❌ **REMOVE or FIX** - Replace with proper responses

### Placeholder Tokens (Debatable)

**3. "<nooutput>" - 24 occurrences**
- **Use**: Tasks requiring non-text output (images, videos, UI designs)
- **Examples**:
  - "Design a company logo" → "<nooutput>" (can't generate images)
  - "Create visual representation of climate change" → "<nooutput>"
  - "Generate ML algorithm to predict stock market" → "<nooutput>"
- **Acceptable?**: ⚠️ Depends on training goals
  - ✅ If you want model to refuse multimodal tasks
  - ❌ If you want model to provide text descriptions instead
- **Action**: **DOCUMENT** behavior or replace with descriptions

## Impact

**Before Special Token Detection**:
- No visibility into data errors
- "nan" and "<noinput>" mixed with valid content
- No way to filter problematic samples

**After Special Token Detection**:
- 50 problematic samples identified (0.096% of dataset)
- Can filter with: `df[df['text_content_contains_error_token'] == True]`
- Clear distinction between errors and placeholders
- Actionable cleanup recommendations

## Usage Examples

### Filter Error Tokens
```python
import pandas as pd

# Load analysis
df = pd.read_csv('message_analysis.csv')

# Find all error tokens
errors = df[df['text_content_contains_error_token'] == True]
print(f"Found {len(errors)} error tokens")

# Show error breakdown
print(errors['text_content_error_token_type'].value_counts())
# Output:
#   nan          28
#   <noinput>    22
```

### Generate Cleanup Report
```python
# Group by error type
for error_type in errors['text_content_error_token_type'].unique():
    samples = errors[errors['text_content_error_token_type'] == error_type]
    print(f"\n{error_type}: {len(samples)} occurrences")
    print("Sample conversations:", samples['conversation_id'].head().tolist())
```

### Filter for Training
```python
# Exclude error tokens from training data
clean_df = df[df['text_content_contains_error_token'] == False]
print(f"Clean samples: {len(clean_df)} / {len(df)} ({len(clean_df)/len(df)*100:.1f}%)")
```

## Recommendations

### Immediate Actions (Priority: HIGH)

1. **Remove "nan" samples** (28 samples)
   ```bash
   # Filter and remove
   csvgrep -c text_content_error_token_type -m "nan" message_analysis.csv > nan_errors.csv
   # Review and regenerate responses
   ```

2. **Fix "<noinput>" samples** (22 samples)
   ```bash
   # Filter and review
   csvgrep -c text_content_error_token_type -m "<noinput>" message_analysis.csv > noinput_errors.csv
   # Replace with proper responses or <nooutput>
   ```

3. **Document "<nooutput>" handling** (24 samples)
   - Decide: Keep as refusal signal OR replace with text descriptions
   - Document behavior in model card

### Configuration Best Practices

**For instruction-following datasets**:
```yaml
error_tokens: ["nan", "<noinput>", "N/A", "null", "undefined"]
placeholder_tokens: ["<nooutput>", "<image>", "<video>"]
```

**For dialogue datasets**:
```yaml
error_tokens: ["nan", "null", "undefined", "[deleted]", "[removed]"]
placeholder_tokens: []  # Usually want text responses only
```

**For code generation datasets**:
```yaml
error_tokens: ["nan", "<error>", "TODO", "FIXME"]
placeholder_tokens: ["<code>", "<output>"]
```

## Next Steps

1. ✅ Enhanced EmptyContentAnalyzer with special token detection
2. ✅ Added comprehensive test coverage (7 tests)
3. ✅ Updated configuration with recommended tokens
4. ⏭️ Run full analysis on Alpaca dataset with new detection
5. ⏭️ Generate cleanup recommendations for identified tokens
6. ⏭️ Document special token handling in training pipeline

## Related Files

- `src/oumi/core/analyze/empty_content_analyzer.py` - Enhanced analyzer
- `tests/unit/core/analyze/test_quality_analyzers.py` - Test suite
- `configs/examples/analyze/quality_comprehensive.yaml` - Updated config
- `/tmp/analyzer_accuracy_report.md` - Original investigation findings
