# pyproject.toml vLLM Version Range Widened

## Changes Made

Updated the vLLM version constraint in `pyproject.toml`:

- Before: `vllm>=0.10,<0.11`
- After: `vllm>=0.10,<0.18`
- Appears in both `[project.optional-dependencies].gpu` and `ci_cpu`

## Note on Compatibility

vLLM v0.17.1 declares `transformers<5` in its requirements. However, it works at runtime with transformers v5.3.0. pip will emit a warning but installation succeeds.

All vLLM code changes use `is_vllm_v017_or_later()` for backward compatibility with vLLM 0.10.x.
