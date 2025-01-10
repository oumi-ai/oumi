# GCP Inference Script

A script for running batch inference on files stored in Google Cloud Storage buckets using vLLM.

## Overview

This script monitors an input GCS bucket for new files, runs inference on them using a specified model, and writes the results to an output GCS bucket. It's designed to work with vLLM for efficient inference.

## Usage

```bash
# Basic usage (single worker)
python gcp_inference.py

# With multiple workers for parallel inference
python gcp_inference.py <num_workers>
```

## Configuration

The script requires the following environment variables:

- `INFERENCE_CONFIG`: Path to the YAML configuration file containing inference settings
