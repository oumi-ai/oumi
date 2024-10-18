#!/bin/bash
# Run some checks, export "OUMI_*" env vars, and print cluster info.

echo "SkyPilot task ID: ${SKYPILOT_TASK_ID}"
echo "SkyPilot cluster: ${SKYPILOT_CLUSTER_INFO}"
echo "Current dir: $(pwd)"
echo "SkyPilot node IPs: ${SKYPILOT_NODE_IPS}"
echo ""
echo "Running on host: $(hostname)"
echo "SkyPilot node rank: ${SKYPILOT_NODE_RANK}"
export OUMI_NUM_NODES=$(echo "$SKYPILOT_NODE_IPS" | wc -l)
export OUMI_TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${SKYPILOT_NUM_GPUS_PER_NODE}))
export OUMI_MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
echo "Master address: ${OUMI_MASTER_ADDR}"
echo "Number of nodes: ${OUMI_NUM_NODES}"
echo "Number of GPUs per node: ${SKYPILOT_NUM_GPUS_PER_NODE}"

if [[ -z "${OUMI_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi
