#!/bin/bash

echo "SkyPilot task ID: ${SKYPILOT_TASK_ID}"
echo "SkyPilot cluster: ${SKYPILOT_CLUSTER_INFO}"
echo "Current dir: $(pwd)"
echo "SkyPilot node IPs: ${SKYPILOT_NODE_IPS}"
echo ""
echo "Running on host: $(hostname)"
echo "SkyPilot node rank: ${SKYPILOT_NODE_RANK}"
export LEMA_NUM_NODES=`echo "$SKYPILOT_NODE_IPS" | wc -l`
export LEMA_MASTER_ADDR=`echo "$SKYPILOT_NODE_IPS" | head -n1`
echo "Master address: ${LEMA_MASTER_ADDR}"
echo "Number of nodes: ${LEMA_NUM_NODES}"
echo "Number of GPUs per node: ${SKYPILOT_NUM_GPUS_PER_NODE}"

export LEMA_DEFAULT_DATALOADER_WORKERS=$((2*${SKYPILOT_NUM_GPUS_PER_NODE}))
if [[ ${LEMA_DEFAULT_DATALOADER_WORKERS} -lt 1 ]]; then
  export LEMA_DEFAULT_DATALOADER_WORKERS=1
elif [[ ${LEMA_DEFAULT_DATALOADER_WORKERS} -gt 8 ]]; then
  export LEMA_DEFAULT_DATALOADER_WORKERS=8
fi
echo "Default number of dataloader workers: ${LEMA_DEFAULT_DATALOADER_WORKERS}"

if [[ -z "${LEMA_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi
