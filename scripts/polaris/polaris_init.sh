#!/bin/bash

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd "${PBS_O_WORKDIR}"

echo "Polaris job ID: ${PBS_JOBID}"
echo "Running on host: $(hostname)"
echo "Polaris queue: ${PBS_QUEUE}"
echo "Current dir: $(pwd)"
echo "Work dir: ${PBS_O_WORKDIR}"
echo "Polaris node file: ${PBS_NODEFILE}"
echo ""
export OUMI_NUM_NODES=$(wc -l <"${PBS_NODEFILE}")
export OUMI_POLARIS_NUM_GPUS_PER_NODE=4
export OUMI_TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${OUMI_POLARIS_NUM_GPUS_PER_NODE}))
export OUMI_MASTER_ADDR=$(head -n1 "${PBS_NODEFILE}")
echo "Master address: ${OUMI_MASTER_ADDR}"
echo "Number of nodes: ${OUMI_NUM_NODES}"
echo "All nodes: $(cat "${PBS_NODEFILE}")"

if [[ -z "${OUMI_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi

# "2083804.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov" -> "2083804"
export OUMI_JOBNUM=$(echo $PBS_JOBID | cut -d'.' -f1)
if [[ -z "${OUMI_JOBNUM}" ]]; then
    echo "Job number is empty for PBS_JOBID: ${PBS_JOBID}!"
    exit 1
fi

# NCCL settings:
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/#multi-gpu-multi-node-scale-up
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN # INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Polaris has 32 "physical" CPU cores, and 64 "logical" cores per node
# (Hyper-threading makes 1 physical core appear as 2 logical cores)
# Physical cores: 0..31. Additional "logical" cores: 32..63.
# https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
NRANKS=1  # Number of MPI ranks to spawn per node (1 worker per node)
NDEPTH=64 # Number of hardware threads per rank (Polaris has 64 CPU cores per node)
CPU_BIND="depth"

# Set up default modules and load conda.
module use /soft/modulefiles
module load conda

# Activate the Oumi Conda environment.
conda activate "/home/${USER}/miniconda3/envs/oumi"
echo "Conda path: ${CONDA_PREFIX}"
