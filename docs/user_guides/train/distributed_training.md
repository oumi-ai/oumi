# Distributed Training

```{attention}
Section under construction. Contributions welcome!
```

## Multi-GPU Training on a Single Node

To configure multi-GPU training, edit the `accelerators` section of your SkyPilot config
 to use `N` GPUs. For example, for 2 `A100` GPUs, set `accelerators: {"A100": 2}`.

There are two options for multi-GPU training:
[DDP (Distributed Data Parallel)](https://huggingface.co/docs/transformers/en/perf_train_gpu_many#dataparallel-vs-distributeddataparallel) and
[FSDP (Fully Sharded Data Parallel)](https://huggingface.co/docs/transformers/en/fsdp).
If your model training can run on a single GPU (i.e. one GPU's memory can hold the model,
optimizer state, etc.), then consider using DDP. Otherwise, consider using FSDP, which
shards the model across your GPUs.

### DDP (Distributed Data Parallel)

To properly configure your machine to do DDP training, either invoke training with the
[`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) command or
[`accelerate launch`](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch)
 using the `--multi_gpu` flag.

Then run `sky launch ...` as before.

### FSDP (Fully Sharded Data Parallel)

NOTE: PyTorch FSDP paper: <https://arxiv.org/abs/2304.11277>

For example, for Phi3 DPO model, there are two related sample configs provided:

- SkyPilot config: [configs/recipes/phi3/dpo/fsdp_sky_job.yaml](../../configs/recipes/phi3/dpo/fsdp_sky_job.yaml)
  - Set the `accelerators:` section as follows: `accelerators: {"A40": N}`, where `N` is the number of GPUs to use e.g., `2`.
- [`accelerate`](https://github.com/huggingface/accelerate) config: [configs/recipes/phi3/dpo/accelerate.yaml](../../configs/recipes/phi3/dpo/accelerate.yaml)
  - Set `num_processes: N`, where `N` is the number of GPUs. NOTE: This step can be optional as the value is overridden in SkyPilot config as: `--num_processes ${OUMI_TOTAL_NUM_GPUS}`)
  - Update `fsdp_transformer_layer_cls_to_wrap` to match transformer layer class name in your model.
  - Review and tune other parameters in the config, as described in [FSDP Configuration](https://huggingface.co/docs/transformers/main/en/fsdp#fsdp-configuration) and in [accelerate FSDP usage guide](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp). They control various performance trade-offs.

Then run `sky launch ...` as before.
