# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501

"""Post-training guidance resource.

Covers downloading weights, running evaluation on-cluster, tearing down
infrastructure, merging LoRA adapters, and pushing to HuggingFace Hub.
Exposed as `guidance://post-training`.
"""

POST_TRAINING_RESOURCE = """# Post-Training Guide

## Priority Order

When training completes on a cloud VM, act quickly to minimize billing.

1. **Download model weights** (do this first)
2. **Run evaluation on the cluster** (optional, while cluster is still up)
3. **Tear down the cluster**
4. **Merge LoRA adapter locally** (if applicable)
5. **Push to HuggingFace Hub** (optional)

---

## Step 1: Download Model Weights

The MCP has no file-transfer tool. Use SkyPilot CLI directly in the user's terminal:

```bash
# Download the output directory from the remote VM
sky rsync-down <cluster-name> ~/sky_workdir/<output_dir> ./output/

# Example:
sky rsync-down sky-abc-user ~/sky_workdir/output/llama8b-sft ./output/
```

**LoRA adapters are small** (~5-50 MB depending on rank and target modules).
Full fine-tuned models are the size of the base model (e.g. ~16 GB for 8B in bf16).

The output directory typically contains:
- `adapter_model.safetensors` + `adapter_config.json` (LoRA)
- OR `model-*.safetensors` + config files (full fine-tune)
- `trainer_state.json`, `training_args.bin` (training metadata)

## Step 2: Run Evaluation (Optional, On-Cluster)

While the cluster is still running, evaluate the fine-tuned model to get benchmark
scores. Use `run_oumi_job` with `command="evaluate"`:

- Point `model.model_name` at the output path on the remote VM
- Use the same cluster (it already has the model weights)
- Common benchmarks: MMLU, HellaSwag, ARC, or task-specific evals

This avoids downloading the full model just to evaluate it.

## Step 3: Tear Down the Cluster

Once weights are downloaded, stop billing immediately.

| Action | MCP Tool | SkyPilot CLI | Effect |
|--------|----------|-------------|--------|
| **Stop** (pause) | `stop_cluster(cloud, cluster_name)` | `sky stop <cluster>` | Stops compute billing, keeps disk. Can restart later. |
| **Down** (delete) | `down_cluster(cloud, cluster_name, confirm, user_confirmation)` | `sky down <cluster>` | Deletes everything. Irreversible. |

- **Use `stop`** if you might want to run more jobs on the same cluster.
- **Use `down`** if you are done — avoids ongoing disk storage fees.
- ALWAYS confirm with the user before calling `down_cluster`.

## Step 4: Merge LoRA Adapter (Local)

If you trained with LoRA/QLoRA, merge the adapter into the base model for deployment:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base, "./output/llama8b-sft")
merged = model.merge_and_unload()
merged.save_pretrained("./output/llama8b-sft-merged")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.save_pretrained("./output/llama8b-sft-merged")
```

Or use Oumi CLI:
```bash
oumi merge -c merge_config.yaml
```

## Step 5: Push to HuggingFace Hub (Optional)

```bash
huggingface-cli upload <your-org>/<model-name> ./output/llama8b-sft-merged
# Or just the adapter:
huggingface-cli upload <your-org>/<model-name>-lora ./output/llama8b-sft
```

## Quick Reference: MCP Capabilities Post-Training

| Task | MCP Support | How |
|------|------------|-----|
| Check job status | Yes | `get_job_status(job_id)` |
| View training logs | Yes | `get_job_logs(job_id)` |
| Run evaluation | Yes | `run_oumi_job(config_path=..., command="evaluate", client_cwd=...)` |
| Run inference | Yes | `run_oumi_job(config_path=..., command="infer", client_cwd=...)` |
| Stop/delete cluster | Yes | `stop_cluster(...)` / `down_cluster(...)` |
| Download files | No | Use `sky rsync-down` in terminal |
| Merge LoRA adapter | No | Use `peft` or `oumi merge` locally |
| Push to HF Hub | No | Use `huggingface-cli upload` in terminal |
"""
