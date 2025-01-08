# Dataset Pre-tokenization Script

This script allows you to pre-tokenize datasets for faster training. It supports both HuggingFace datasets and local files in various formats (jsonl, parquet, arrow).

## Features

- Pre-tokenize text using any HuggingFace tokenizer
- Support for both HuggingFace datasets and local files
- Multiple input formats: jsonl, parquet, arrow
- Parallel processing for faster tokenization
- Configurable output sharding
- Option to skip tokenization for format conversion only

## Usage

### Using with HuggingFace Datasets

```bash
python process_dataset.py \
    -c path/to/config.yaml \
    --input_dataset "dataset_name" \
    --dataset_split "train" \
    --target_col "text" \
    --output_dir "output/path" \
    --num_shards 512
```

### Using with Local Files

```bash
python process_dataset.py \
    -c path/to/config.yaml \
    --input_path "data/*.jsonl" \
    --input_format "jsonl" \
    --target_col "text" \
    --output_dir "output/path"
```

## Arguments

### Required Arguments
- `--output_dir`: Path to the output directory

### Optional Arguments
- `-c, --config`: Path to the configuration file (required if not skipping tokenization)
- `-v, --verbose`: Enable verbose logging
- `--target_col`: Target text column to tokenize
- `--overwrite`: Whether to overwrite existing output files
- `--num_proc`: Number of processes for parallel execution (-1 for all CPU cores)
- `--skip_tokenize`: Skip tokenization (useful for format conversion only)
- `--max_shard_size`: Max shard size (e.g., "256MB")
- `--num_shards`: Number of output shards (default: 512)

### HuggingFace Dataset Arguments
- `--input_dataset`: HuggingFace dataset name or local path
- `--dataset_subset`: Dataset subset name
- `--dataset_split`: Dataset split (e.g., "train")
- `--trust_remote_code`: Whether to trust remote code when loading datasets

### Local File Arguments
- `--input_path`: Path(s) to input files or directories
- `--input_format`: Input format (jsonl, parquet, or arrow)

## Output

The script outputs tokenized data in either:
- A sharded HuggingFace dataset format when using `--input_dataset`
- Individual parquet files when using `--input_path`

## Examples

### Pre-tokenize a HuggingFace Dataset
```bash
python process_dataset.py \
    -c config.yaml \
    --input_dataset "EleutherAI/pile" \
    --dataset_split "train" \
    --target_col "text" \
    --output_dir "tokenized_pile" \
    --num_shards 1024
```

### Convert Local JSONL Files to Parquet
```bash
python process_dataset.py \
    --input_path "data/*.jsonl" \
    --input_format "jsonl" \
    --output_dir "converted_data" \
    --skip_tokenize
```
