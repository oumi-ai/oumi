# Dataset Pre-tokenization Script

This script allows you to pre-tokenize datasets for faster training. It supports both HuggingFace datasets and local files in various formats (jsonl, parquet, arrow).

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
