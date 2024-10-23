#! /bin/bash

set -xe

# Summarize datasets classes into markdown tables
python docs/_summarize_module.py summarize-module "oumi.datasets.vision_language" --filter-type "class" --output-file docs/api/summary/vl_sft_datasets.md
python docs/_summarize_module.py summarize-module "oumi.datasets.pretraining" --filter-type "class" --output-file docs/api/summary/pretraining_datasets.md
python docs/_summarize_module.py summarize-module "oumi.datasets.sft" --filter-type "class" --output-file docs/api/summary/sft_datasets.md
python docs/_summarize_module.py summarize-module "oumi.datasets.preference_tuning" --filter-type "class" --output-file docs/api/summary/preference_tuning_datasets.md
python docs/_summarize_module.py summarize-module "oumi.inference" --parent-class "oumi.core.inference.BaseInferenceEngine" --filter-type "class" --output-file docs/api/summary/inference_engines.md
python docs/_summarize_module.py summarize-module "oumi.judges.judge_court"  --filter-type "function" --exclude-imported --output-file docs/api/summary/judges.md

# Summarize configs files into markdown tables
python docs/_summarize_module.py summarize-configs configs "oumi.core.configs.TrainingConfig" --output-file docs/api/summary/training_config.md > /dev/null 2>&1
python docs/_summarize_module.py summarize-configs configs "oumi.core.configs.EvaluationConfig" --output-file docs/api/summary/evaluation_config.md > /dev/null 2>&1
python docs/_summarize_module.py summarize-configs configs "oumi.core.configs.AsyncEvaluationConfig" --output-file docs/api/summary/async_evaluation_config.md > /dev/null 2>&1
python docs/_summarize_module.py summarize-configs configs "oumi.core.configs.JobConfig" --output-file docs/api/summary/job_config.md > /dev/null 2>&1
python docs/_summarize_module.py summarize-configs configs "oumi.core.configs.InferenceConfig" --output-file docs/api/summary/inference_config.md > /dev/null 2>&1
