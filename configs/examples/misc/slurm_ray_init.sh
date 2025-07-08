#!/bin/bash
# Env vars required by this script (should be set by SLURM):
# SLURM_JOB_NODELIST: list of nodes in the job.
# SLURM_CPUS_PER_TASK: number of CPUs per node.
# SLURM_GPUS_ON_NODE: number of GPUs per node.

set -e
source ~/miniconda3/etc/profile.d/conda.sh # Required for conda.
conda activate oumi
pip install uv && uv pip install 'oumi[gpu]'

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
scontrol show hostnames "$SLURM_JOB_NODELIST" > nodes.txt
nodes_array=($nodes)
echo "nodes_array: $nodes_array" > nodesarray.txt
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
head_node_ip=${ADDR[1]}
else
head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "Info----------"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_ON_NODE" \
    --block &

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_ON_NODE" \
        --block &
    sleep 5
done

# Print cluster status to confirm setup.
ray status
