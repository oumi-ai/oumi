# max_grad_norm: null Breaks on transformers v5

## Breaking Change

Existing Qwen3 training configs use `max_grad_norm: null`, which causes a `TypeError` on transformers v5:

```
TypeError: '>' not supported between instances of 'NoneType' and 'int'
```

## Root Cause

In transformers v5, the training loop checks `if self.args.max_grad_norm > 0:` without guarding against `None`. In v4, this was either handled differently or `None` was converted to a default value.

The comparison `None > 0` raises `TypeError` in Python.

## How to Reproduce

```yaml
# Any training config with:
training:
  max_grad_norm: null
```

```bash
oumi train -c configs/recipes/qwen3/sft/0.6b_full/train.yaml
# TypeError: '>' not supported between instances of 'NoneType' and 'int'
```

## Files Affected

- `configs/recipes/qwen3/sft/0.6b_full/train.yaml`
- `configs/recipes/qwen3/sft/1.7b_full/train.yaml`
- `configs/recipes/qwen3/sft/4b_full/train.yaml`

## Fix Required

Update `max_grad_norm: null` to an explicit value in all affected configs. Common choices:

- `max_grad_norm: 1.0` — standard default, enables gradient clipping
- `max_grad_norm: 0.0` — disables gradient clipping (equivalent to the old `null` behavior)

The transformers default for `max_grad_norm` is `1.0`.
