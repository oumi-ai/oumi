#!/bin/bash

echo "***ENV BEGIN (PMI_RANK: $PMI_RANK)***"
echo "LEMA_MASTER_ADDR: $LEMA_MASTER_ADDR"
echo "LEMA_MASTER_PORT: $LEMA_MASTER_PORT"
echo "LEMA_NUM_NODES: $LEMA_NUM_NODES"
echo "PMI_LOCAL_RANK: $PMI_LOCAL_RANK"
echo "PMI_RANK: $PMI_RANK"
echo "NCCL_COLLNET_ENABLE: $NCCL_COLLNET_ENABLE"
echo "NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
echo "NCCL_DEBUG: $NCCL_DEBUG"
nvidia-smi -L
echo "***ENV END (Host: $HOSTNAME, PMI_RANK: $PMI_RANK)***"

set -x  # Print "torchrun" command with expanded variables

TRAIN_DATASETS="data.train.datasets=
- dataset_name: \"/eagle/community_ai/datasets/fineweb-edu/sample-10BT\"
  subset: \"default\"
  split: \"train\"
"

torchrun \
    --nnodes=${LEMA_NUM_NODES} \
    --node-rank=${PMI_RANK:=0} \
    --nproc-per-node=4 \
    --master-addr=${LEMA_MASTER_ADDR} \
    --master-port=8007 \
    -m lema.train \
    -c configs/lema/llama2b.pt.yaml \
    "model.compile=false" \
    "$TRAIN_DATASETS" \
    "training.run_name='polaris.llama2b.pt.${PBS_JOBID}'" \
    "training.max_steps=20" \
    "training.save_steps=0" \
    "training.save_final_model=False" \
    "training.per_device_train_batch_size=2" \
    "training.gradient_accumulation_steps=128" \
    "training.output_dir=output/llama2b.pt/" \
    "training.dataloader_num_workers=2" \
    "training.dataloader_prefetch_factor=4" \
    "training.include_performance_metrics=true" \
    "training.ddp_find_unused_parameters=false" \
    "training.try_resume_from_last_checkpoint=false" \
    "training.enable_wandb=true"

echo "Node ${PMI_RANK:=0} is all done!"
