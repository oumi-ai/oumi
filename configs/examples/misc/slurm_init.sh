#!/bin/bash
# export "OUMI_*" env vars, and print cluster info.

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

export OUMI_NUM_NODES=${SLURM_JOB_NUM_NODES}
export OUMI_TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${SLURM_GPUS_ON_NODE}))
export OUMI_MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo
echo "------------------------------------------"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Slurm job name: ${SLURM_JOB_NAME}"
echo "Slurm task PID: ${SLURM_TASK_PID}"
echo "Job start time: $(date -d @"${SLURM_JOB_START_TIME}")"
echo "Slurm job nodelist: ${SLURM_JOB_NODELIST}"
echo "Current dir: $(pwd)"
echo
echo "Head node: ${head_node}"
echo "Master address: ${OUMI_MASTER_ADDR}"
echo "Number of nodes: ${OUMI_NUM_NODES}"
echo "Number of tasks per node: $((SLURM_NTASKS / SLURM_JOB_NUM_NODES))"
echo "Number of CPUs per node: ${SLURM_CPUS_ON_NODE}"
echo "Number of GPUs per node: ${SLURM_GPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

if [[ -z "${OUMI_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi

echo "------------------------------------------"
echo

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
