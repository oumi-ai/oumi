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
echo "SLURM_NNODES         =" $SLURM_NNODES
echo "SLURM_NTASKS         =" $SLURM_NTASKS
echo "SLURM_TASKS_PER_NODE =" $SLURM_TASKS_PER_NODE
echo "SLURM_CPUS_PER_TASK  =" $SLURM_CPUS_PER_TASK
echo "SLURM_NPROCS         =" $SLURM_NPROCS
echo "SLURM_NODELIST       =" $SLURM_NODELIST
echo "SLURM_JOB_NODELIST   =" $SLURM_JOB_NODELIST
echo ""

export OUMI_NUM_NODES=$SLURM_NNODES
export OUMI_FRONTIER_NUM_GPUS_PER_NODE=8
export OUMI_TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${OUMI_FRONTIER_NUM_GPUS_PER_NODE}))
export OUMI_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Master address: ${OUMI_MASTER_ADDR}"
echo "Number of nodes: ${OUMI_NUM_NODES}"
echo "All nodes: " $(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ -z "${OUMI_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi

# DELETE: export OUMI_JOBNUM=$(echo $SLURM_JOBID | cut -d'.' -f1)
export OUMI_JOBNUM="${SLURM_JOBID}"
if [[ -z "${OUMI_JOBNUM}" ]]; then
    echo "Job number is empty for SLURM_JOBID: ${SLURM_JOBID}!"
    exit 1
fi

# For more details on these params, refer to NCCL (RCCL) settings:
# https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#environment-variables
FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export NCCL_DEBUG=WARN # INFO
## export NCCL_DEBUG_SUBSYS=ALL

# Frontier has 64 "physical" CPU cores, and 128 "logical" cores per node
# (Hyper-threading makes 1 physical core appear as 2 logical cores)
# Physical cores: 0..63. Additional "logical" cores: 64..127.
# Note that 8 out of 64 cores are reserved for the system in the "low noise" mode,
# so there are only 56 allocatable cores by default:
# https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#low-noise-mode-layout
NRANKS=1  # Number of MPI ranks to spawn per node (1 worker per node)
NDEPTH=64 # Number of hardware threads per rank (Frontier has 64 CPU physical cores per node, 56 available, 2X logical ones)
CPU_BIND="depth"

# Setup the environment variables.
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
# TODO: Append "/slurm_job_${SLURM_JOBID}" suffix in case of cache file lock contention.
export HF_HOME="/lustre/orion/lrn081/scratch/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"

echo "Loading Frontier modules..."

# Set up default modules.
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Activate the Oumi Conda environment.
set +x
source activate "/lustre/orion/lrn081/scratch/$USER/miniconda3/envs/oumi"
echo "Conda path: ${CONDA_PREFIX}"
set -x

echo "frontier_init.sh is done!"
