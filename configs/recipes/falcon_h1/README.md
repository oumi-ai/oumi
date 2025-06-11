# Falcon-H1

## Summary

Configs for TII's Falcon-H1 model family. See the [official blog post](https://falcon-lm.github.io/blog/falcon-h1/) and [GitHub page](https://github.com/tiiuae/Falcon-H1) for technical details, benchmarks, **SFT** and **DPO** settings, and throughput comparisons.

All models are supported in both **native** and **VLLM** inference modes. Models in this family include:

* **tiiuae/Falcon-H1-0.5B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct)
* **tiiuae/Falcon-H1-1.5B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct)
* **tiiuae/Falcon-H1-1.5B-Deep-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct)
* **tiiuae/Falcon-H1-3B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-H1-3B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-H1-3B-Instruct)
* **tiiuae/Falcon-H1-7B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-H1-7B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct)
* **tiiuae/Falcon-H1-34B-\[Base|Instruct]**
  [Base](https://huggingface.co/tiiuae/Falcon-H1-34B-Base), [Instruct](https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct)

---

## Quickstart

> [!IMPORTANT]
> If you are running Falcon-H1 locally, you will need to set up additional Python library dependencies. Please run the `pip` and `uv` commands under the `-setup` section of the [remote job configs](https://github.com/oumi-ai/oumi/blob/main/configs/recipes/falcon_h1/evaluation/falcon_h1_0_5b/lambda_job.yaml#L36-L43).

1. Follow the [Oumi quickstart guide](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) To launch jobs on remote clusters, follow the [SkyPilot job launcher guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired `oumi` command with the provided config paths (examples below).
4. (Optional) For development, clone the Oumi repository and strip the `oumi://` prefix from config paths to use local files.

---

## Example Commands

### Training

Launch full fine-tuning on Falcon-H1-7B-Instruct locally:

```bash
oumi train -c oumi://configs/recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml
```

Launch full fine-tuning on Falcon-H1-7B-Instruct remotely via Lambda:

```bash
oumi launch up -c oumi://configs/recipes/falcon_h1/sft/falcon_h1_7b/full_lambda_job.yaml --cluster falcon-h1-7b-fft
```

### Evaluation

Evaluate Falcon-H1-7B-Instruct locally:

```bash
oumi evaluate -c oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_7b/eval.yaml
```

Evaluate using the VLLM engine:

```bash
oumi evaluate -c oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_7b/eval.yaml --inference_engine VLLM
```

Evaluate on Lambda 4xA100 cluster:

```bash
oumi launch up -c oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_7b/lambda_job.yaml --cluster falcon-h1-7b-instruct-eval
```

### Inference

Run interactive inference locally:

```bash
oumi infer -i -c oumi://configs/recipes/falcon_h1/inference/7b_infer.yaml
```

Run interactive inference using VLLM engine:

```bash
oumi infer -i -c oumi://configs/recipes/falcon_h1/inference/7b_infer.yaml --engine VLLM
```
