# DCVLR - Getting Under the Hood

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/Conferences/2025)
[![Competition](https://img.shields.io/badge/Competition-Open-green.svg)](https://dcvlr.org)

---

<div align="center">

  <h3>
   🌐 <a href="https://dcvlr-neurips.github.io">Official webpage</a> •
   🚀 <a href="https://oumi-ai.typeform.com/to/LnYoisi5">Sign up for updates</a> •
   🎯 <a href="https://oumi-ai.typeform.com/to/OGPuRt6U">Apply for GPU credits (sponsored by Lambda Labs)</a>
   </h3>
</div>

---

## What is this directory?

This directory is intended to accompany the [2025 DCVLR (Data Curation for Vision-Language Reasoning) NeurIPS competition](https://dcvlr-neurips.github.io/). If you don't know what that is, you should go read the competition website and then come back here!

## DCVLR: Digging Deeper

The DCVLR competition was explicitly designed to have a *low barrier to entry*, allowing a diverse collection of teams to compete. However, we know that many teams may be interested in digging deeper into the data and the tasks in order to optimize the performance of their allowed submissions. If that's you, you've come to the right place. This directory will give you all the building blocks necessary to reproduce the train and eval pipeline used in the DCVLR competition on your own cluster.

## What You Will Need

In order to reproduce our experimental pipeline with the model architectures we consider for this competition (which range from 7B to 10B parameters), you will need access to a cluster with at least 8 A100 GPUs, and 1TB of disk space. If you don't have access, you can rent a cluster, e.g. on [Lambda](https://lambdalabs.com/service/gpu-cloud). All DCVLR participants are eligible for a credit on Lambda which they can use to run experiments for the competition.

We plan to provide add examples of how to experiment on smaller architectures (e.g. 1B parameters) to this directory at a later date, so stay tuned. You can also refer to the [Oumi documentation](https://oumi.ai/docs/en/latest/index.html) for more information on how to run experiments on smaller clusters.

### Data Sourcing

Where can you source data that might be suitable for training for this competition? If you want to draw on existing datasets, here are a few we recommend looking into --

[Llava-O1](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)

[Math-Llava](https://huggingface.co/datasets/Zhiqiang007/MathV360K)

[Geo-170K](https://huggingface.co/datasets/Luckyjhg/Geo170K)

[Open-R1](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified)

[AIDC Ovis](https://huggingface.co/datasets/AIDC-AI/Ovis-dataset)

[Llava 1V](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)

### Data Curation

We will add documentation on how to use Oumi for synthetic data curation and data transformation here soon. Stay tuned!

For now, you will have to BYOD (bring your own dataset) in an Oumi-supported dataset format. For this competition, we highly recommend the flexible "hf_vision" format, which allows you to load a wide range of VL datasets from the Hugging Face Hub. Here's an example we used for training on a filtered version of the Multimodal Open-R1 dataset:

```bash
datasets:
    - dataset_name: "hf_vision"
    split: "train"
    shuffle: True
    seed: 42
    trust_remote_code: True
    transform_num_workers: "auto"
    dataset_kwargs:
        hf_dataset_path: "penfever/multimodal-open-r1-8192-filtered-tighter"
        image_column: "image"
        question_column: "problem"
        answer_column: "solution"
        return_tensors: True
```

### Model Training

#### Setup and Environment

DCVLR experiments can be run using the main branch of the Oumi repository. We provide a [DOCKERFILE](https://github.com/oumi-ai/oumi/blob/main/Dockerfile) for building Oumi, or you can follow the instructions in the [Quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html).

#### Commands

Model training is extremely straightforward, requiring only a single command:

```bash
export MY_CONFIG=<PATH/TO/qwenvl-openr1.yaml>
torchrun --nproc-per-node 8 --standalone -m oumi train -c $MY_CONFIG
```

We provide configurations for three models; Molmo-D, Molmo-O, and QwenVL-2.5. Other models such as  InternVL3 may also be used in the competition.

Depending on how `training: output_dir` is set in the config file, the model checkpoints will be saved in the base of the specified directory.

We then recommend syncing the trained model to HuggingFace Hub using the `huggingface-cli` tool to enable version control and ease of future access. The repository need not exist in advance, it will be automatically created when you use this command.

```bash
huggingface-cli upload-large-folder <YOUR_HF_REPO> <YOUR_OUTPUT_DIRECTORY> --repo-type=model
```

### Model Evaluation

#### Setup and Environment

We use a modified version of [VLMEvalKit](https://github.com/oumi-ai/VLMEvalKit) for our evaluation harness. You can clone and install it following the directions in the repo, or use the provided [DOCKERFILE](https://github.com/oumi-ai/VLMEvalKit/blob/main/docker/Dockerfile.cuda12.9-oumi-molmo-qwen).

#### Commands

Model evaluation can also be conducted using a simple one-line command. We give an example with four datasets; these datasets are not guaranteed to be the ones we use in the competition, however, they are a good starting point for the types of tasks we are considering.

```bash
export MODEL_NAME=<YOUR/HF/MODEL/PATH>
export WORK_DIR=<YOUR/OUTPUT/DIRECTORY>
mkdir -p "$WORK_DIR"
export DATASETS="VMCBench_DEV WeMath MathVista_MINI LiveXivVQA"
python scripts/wandb_logger.py --run-and-log \
                               --data $DATASETS \
                               --work-dir $WORK_DIR \
                               --use-vllm \
                               --save-detailed-eval \
                               --save-judge-responses \
                               --max-output-tokens 4096 \
                               --pass-custom-model $MODEL_NAME
```

## How to Cite DCVLR

If you wish to refer to DCVLR in your work, please cite the following:

```bib
@misc{DCVLR: Data Curation for Vision-Language Reasoning,
  author = {Feuer, Benjamin and Tripathi, Rohun and Elachqar, Oussama and Zhang, Yuhui and Hulkund, Neha and Nguyen, Thao and Shabtay, Nimrod and Udandarao, Vishaal and Wang, Xiaohan and Webb, Stefan and Koukoumidis, Emmanouil and Schmidt, Ludwig and Xie, Saining and Yeung-Levy, Serena and Liang, Paul and Beery, Sara and Gkioxari, Georgia}
  month = June,
  title = {{DCVLR}},
  year = {2025}
}
```
