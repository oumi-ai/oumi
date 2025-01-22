# Bulk Inference

Configs for running bulk inference on a large dataset in GCP. This script will grab all files from the input bucket, run inference on them, and save the results to an output bucket, using filenames mirroring those in the input bucket.

NOTES:

1. The input files are required to be JSONL.
2. The script will skip inference on input files which already have corresponding output files. This prevents repetitive work if running the script multiple times.
3. If you use multiple GPUs, the script will distribute the model across them using parallel tensors.

## How to run

1. Update the input and output GCS buckets to your desired locations.
2. Update `INFERENCE_CONFIG` to your desired inference config. We use Llama 3.1 8B by default.
3. Run `oumi launch up -c configs/examples/bulk_inference/gcp_job.yaml --cluster oumi-bulk-inference`
