# **Qwen2.5-VL 3B**

Configs for the **`Qwen2.5-VL`** 3B model.
🔗 **Reference:** [Qwen2.5-VL-3B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

---

❗ **Important Note**
As of **February 2025**, `Qwen2.5-VL` is integrated into the **latest** `transformers` _dev_ version.

⚠️ **Earlier versions may cause a runtime error:**
KeyError: ‘qwen2_5_vl’

Oumi has successfully tested this integration with:
- **SFT training**
- **Native inference** using **`transformers 4.49.0.dev0`**

To update `transformers` to this version, run:

```sh
pip install git+https://github.com/huggingface/transformers.git
```

⚠️ Caution: This upgrade may break other Oumi utilities. Proceed carefully.
