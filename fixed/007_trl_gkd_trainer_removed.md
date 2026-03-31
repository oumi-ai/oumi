# GKDTrainer Moved to trl.experimental in TRL v0.29

## Breaking Change

`trl.GKDTrainer` and `trl.GKDConfig` were moved from `trl` to `trl.experimental.gkd` in TRL v0.29. Importing from the top-level `trl` module raises `AttributeError: module 'trl' has no attribute 'GKDTrainer'`.

## Root Cause

TRL v0.29 reorganized several trainers into the `trl.experimental` namespace. GKD was moved to `trl.experimental.gkd`. KTO was also moved there but remains importable from the top-level with a deprecation warning.

## How to Reproduce

```python
import trl
# Fails on TRL>=0.29
trainer = trl.GKDTrainer(...)
# AttributeError: module 'trl' has no attribute 'GKDTrainer'

# Works on TRL>=0.29
from trl.experimental.gkd import GKDTrainer, GKDConfig
```

## Files Changed

- `src/oumi/builders/training.py` — trainer builder
- `src/oumi/core/configs/params/training_params.py` — config class selection

## Fix Applied

Used `is_trl_v0_29_or_later()` to conditionally import from the correct location.

```python
# In training.py builder:
from oumi.utils.packaging import is_trl_v0_29_or_later

if is_trl_v0_29_or_later():
    from trl.experimental.gkd import GKDTrainer
else:
    GKDTrainer = trl.GKDTrainer

# In training_params.py config (bug fix: was incorrectly using is_transformers_v5):
if is_trl_v0_29_or_later():
    from trl.experimental.gkd import GKDConfig
    config_class = GKDConfig
else:
    config_class = trl.GKDConfig
```
