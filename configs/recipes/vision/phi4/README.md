# **Phi-4-multimodal-instruct 5.6B**

Configs for Phi-4-multimodal-instruct 5.6Β model.
🔗 **Reference:** [Phi-4-multimodal-instruct on Hugging Face](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

---

This is a multimodal model that combines text, visual, and audio inputs.
It uses a "Mixture of LoRAs" approach, allowing you to plug in adapters for each
modality without needing to retrain the base model. For more information consider
reading the following:

- [Mixture-of-LoRAs](https://arxiv.org/abs/2403.03432)
- [Phi-4 Multimodal Technical Report](https://arxiv.org/abs/2503.01743)

⚠️ This model requires `flash attention 2`. Run the following if executing in a custom fashion:
```sh
pip install -U flash-attn --no-build-isolation
```
