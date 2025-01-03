# Custom LM-Harness Tasks

## Overview

While Oumi provides integration with the LM Evaluation Harness and its extensive task collection, you may need to create custom evaluation tasks for specific use cases. This guide walks you through the process of creating and implementing custom evaluation tasks using the `LM Evaluation Harness` (`lm_eval`) framework.

## Resources
The new task guide is a good starting point:
- `lm_eval` New Task Guide <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md>

## Example

### Building a new custom task
```{attention}
Section under construction. Contributions welcome!
```

### Running Evaluation

Now that you have a custom task, you can include it in the evaluation config:

```yaml
model:
  model_name: "my-model"
  trust_remote_code: True

lm_harness_params:
  tasks:
    - "custom_task"
  num_fewshot: 5
  num_samples: 100
```

Finally, run evaluation with your custom task:

```python
from oumi import evaluate
from oumi.core.configs import EvaluationConfig

config = EvaluationConfig.from_yaml("config.yaml")
evaluate(config)
```

## API Reference

See the following classes for complete documentation:

- {external:class}`lm_eval.base.Task`
- {py:class}`~oumi.core.configs.EvaluationConfig`
