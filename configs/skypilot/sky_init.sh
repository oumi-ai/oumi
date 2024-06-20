#!/bin/bash

echo "SkyPilot node IPs: ${SKYPILOT_NODE_IPS}\n"
echo "SkyPilot node rank: ${SKYPILOT_NODE_RANK}"
export LEMA_NUM_NODES=`echo "$SKYPILOT_NODE_IPS" | wc -l`
export LEMA_MASTER_ADDR=`echo "$SKYPILOT_NODE_IPS" | head -n1`
echo "Master address: ${LEMA_MASTER_ADDR}"
echo "Number of nodes: ${LEMA_NUM_NODES}"
echo "Number of GPUs per node: ${SKYPILOT_NUM_GPUS_PER_NODE}"

if [[ -z "${LEMA_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi
