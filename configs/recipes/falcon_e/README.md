# Falcon-E

## Summary

Configs for TII's Falcon-E model family. See the [official blog post](https://falcon-lm.github.io/blog/falcon-edge/) for technical details, benchmarks, **SFT** and **DPO** settings, and throughput comparisons.

All models are supported in both **native** and **VLLM** inference modes. Models in this family include:

* **tiiuae/Falcon-E-1B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-E-1B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-E-1B-Instruct), [Instruct (GGUF)](https://huggingface.co/tiiuae/Falcon-E-1B-Instruct-GGUF),
* **tiiuae/Falcon-E-3B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-E-3B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-E-3B-Instruct), [Instruct (GGUF)](https://huggingface.co/tiiuae/Falcon-E-3B-Instruct-GGUF)

---

## Quickstart

> [!IMPORTANT]
> If you are running Falcon-E locally, you will need to set up additional Python library dependencies. Please run the `pip` and `uv` commands under the `-setup` section of the [remote job configs](https://github.com/oumi-ai/oumi/blob/main/configs/recipes/falcon_h1/evaluation/falcon_h1_0_5b/lambda_job.yaml#L36-L43) and in addition: `pip install onebitllms`.

1. Follow the [Oumi quickstart guide](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) To launch jobs on remote clusters, follow the [SkyPilot job launcher guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired `oumi` command with the provided config paths (examples below).
4. (Optional) For development, clone the Oumi repository and strip the `oumi://` prefix from config paths to use local files.

---

## Example Commands

### Training

Launch full fine-tuning on Falcon-E-1B-Instruct locally:

```bash
oumi train -c oumi://configs/recipes/falcon_e/sft/falcon_e_1b/full_train.yaml
```

### Evaluation

Evaluate Falcon-E-Instruct locally:

```bash
oumi evaluate -c oumi://configs/recipes/falcon_e/evaluation/falcon_e_1b/eval.yaml
```

Evaluate using the VLLM engine:

```bash
oumi evaluate -c oumi://configs/recipes/falcon_e/evaluation/falcon_e_1b/eval.yaml --inference_engine VLLM
```
