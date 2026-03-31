# pyproject.toml Dependency Version Updates

## Changes Made

Updated the core dependency version constraints in `pyproject.toml` to support the transformers v5 ecosystem:

### transformers
- Before: `transformers>=4.57,<4.58`
- After: `transformers>=4.57,<6`
- Why: Widened range to support both v4.57+ and v5.x. All code changes use `is_transformers_v5()` for backward compatibility.

### trl (Transformer Reinforcement Learning)
- Before: `trl>=0.24,<0.27`
- After: `trl>=0.24,<0.30`
- Why: Widened range to support both older TRL (v0.24-v0.26) and new TRL (v0.29). All code changes use `is_trl_v0_29_or_later()` for backward compatibility. TRL v0.29 includes breaking changes of its own (DPO column renames, GKD removal, KTO parameter changes) documented in separate issue files.

### Other dependencies (unchanged but noted)
- `peft>=0.17,<0.19` — already compatible with transformers v5
- `accelerate>=1.10.0,<2.0` — already compatible with transformers v5
- `datasets>=3.2,<5` — already compatible with transformers v5

## How to Reproduce

```bash
# With old constraints, transformers v5 cannot be installed
pip install "transformers>=4.57,<4.58"  # pins to v4
# With new constraints
pip install "transformers>=5.3,<6"      # allows v5
```
