## **InternVL 3.0 1B — A Multimodal Large Language Model (MLLM)**

**InternVL 3.0 1B** is an efficient multimodal model that integrates both **text and visual inputs**, following the **ViT-MLP-LLM** design paradigm.

📘 Learn more: [huggingface.co/OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

---

### ⚠️CAUTION⚠️ points:

This model has been integrated in Oumi with significant *limitations*.

1. It requires the latest transformers version (4.52.0.dev0), which you can install with:
```sh
    pip install git+https://github.com/huggingface/transformers.git
```
Use this with caution, as it might break some of Oumi's features which is as of today (April 23rd 2025) is using transformers <4.52.

2. The model is currently working with a *single* GPU environment, and requires its weights in full precision (fp32). Multi-GPU and half-point precision version are WIP.
