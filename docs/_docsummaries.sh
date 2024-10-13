#! /bin/bash

set -xe

python docs/_summarize_module.py "oumi.datasets.vision_language" --filter-type "class" --output-file docs/api/summary/vl_sft_datasets.md
python docs/_summarize_module.py "oumi.datasets.pretraining" --filter-type "class" --output-file docs/api/summary/pretraining_datasets.md
python docs/_summarize_module.py "oumi.datasets" --parent-class "oumi.core.datasets.BaseLMSftDataset" --filter-type "class" --output-file docs/api/summary/sft_datasets.md
