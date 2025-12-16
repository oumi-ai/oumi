# GOLD Trainer Examples

This directory contains example configurations for training with the GOLD (Generalized Online Logit Distillation) trainer.

## Overview

GOLD extends GKD (Generalized Knowledge Distillation) to support:
- **Cross-tokenizer distillation** through Universal Logit Distillation (ULD)
- **vLLM acceleration** for faster on-policy generation
- **Standard same-tokenizer distillation** like GKD


## References

- GOLD Paper: [Unlocking On-Policy Distillation for Any Model Family](https://arxiv.org/abs/2501.xxxxx)
- TRL Documentation: https://huggingface.co/docs/trl
- vLLM Documentation: https://docs.vllm.ai/
