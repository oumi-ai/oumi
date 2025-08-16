# macOS GGUF Inference Configurations

To find all macOS-compatible GGUF inference configurations, you can use one of the following search methods:

### Method 1: Search by filename pattern
```bash
find configs/recipes -name "*macos_infer.yaml" -type f
```

### Method 2: Search by engine type
```bash
grep -r "engine: LLAMACPP" configs/recipes --include="*.yaml"
```

## See Also

- [Oumi Documentation](https://oumi.ai/docs/en/latest/user_guides/infer/infer.html)
- [Inference Configuration Reference](https://github.com/oumi-ai/oumi/blob/main/src/oumi/core/configs/inference_config.py)
