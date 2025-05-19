#!/bin/bash

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${SLURM_SUBMIT_DIR} ..."
cd "${SLURM_SUBMIT_DIR}"

echo "Frontier job ID: ${SLURM_JOBID}"
echo "Running on host: $(hostname)"
# echo "Frontier queue: ${PBS_QUEUE}"
echo "Current dir: $(pwd)"
echo "Work dir: ${SLURM_SUBMIT_DIR}"
echo "Frontier node file: ${SLURM_NODELIST}"
echo ""
export OUMI_NUM_NODES=$(wc -l <"${PBS_NODEFILE}")
export OUMI_FRONTIER_NUM_GPUS_PER_NODE=8
export OUMI_TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${OUMI_FRONTIER_NUM_GPUS_PER_NODE}))
export OUMI_MASTER_ADDR=$(head -n1 "${PBS_NODEFILE}")
echo "Master address: ${OUMI_MASTER_ADDR}"
echo "Number of nodes: ${OUMI_NUM_NODES}"
echo "All nodes: $(cat "${PBS_NODEFILE}")"

if [[ -z "${OUMI_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi

# "2083804.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov" -> "2083804"
export OUMI_JOBNUM=$(echo $SLURM_JOBID | cut -d'.' -f1)
if [[ -z "${OUMI_JOBNUM}" ]]; then
    echo "Job number is empty for SLURM_JOBID: ${SLURM_JOBID}!"
    exit 1
fi

# NCCL settings:
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/#multi-gpu-multi-node-scale-up
# export NCCL_COLLNET_ENABLE=1
# export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_DEBUG=WARN # INFO
## export NCCL_DEBUG_SUBSYS=ALL

# Polaris has 32 "physical" CPU cores, and 64 "logical" cores per node
# (Hyper-threading makes 1 physical core appear as 2 logical cores)
# Physical cores: 0..31. Additional "logical" cores: 32..63.
# https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
NRANKS=1  # Number of MPI ranks to spawn per node (1 worker per node)
NDEPTH=64 # Number of hardware threads per rank (Frontier has 64 CPU cores per node)
CPU_BIND="depth"

# Setup the environment variables.
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
export HF_HUB_CACHE=/lustre/orion/lrn081/scratch/$USER/.cache/huggingface/hub/

# Set up default modules.
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Activate the Oumi Conda environment.
conda activate --no-stack /lustre/orion/lrn081/scratch/$USER/miniconda3/envs/oumi
echo "Conda path: ${CONDA_PREFIX}"
