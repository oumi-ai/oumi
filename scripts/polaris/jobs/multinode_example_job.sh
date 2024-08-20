#!/bin/bash

#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

# Set up conda.
module load conda

# Activate the LeMa Conda environment.
conda activate /home/$USER/miniconda3/envs/lema

TRAINING_MODE="ddp1gpu"
echo "Starting ${TRAINING_MODE} training with ${LEMA_NUM_NODES} node(s)..."

# NCCL settings:
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/#multi-gpu-multi-node-scale-up
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_DEBUG=INFO # WARN # INFO
# export NCCL_DEBUG_SUBSYS=ALL

if [ "${TRAINING_MODE}" == "ddp1gpu" ]; then
    NRANKS_PER_NODE=4  # Spawn 4 ranks per Polaris node (1 `torchrun` for each GPU)
    NDEPTH=8 # Number of hardware threads per rank (Polaris has 64 CPU cores per node)
    CPU_BIND="numa"
else
    NRANKS_PER_NODE=1  # Number of MPI ranks to spawn per node (1 `torchrun` per node)
    NDEPTH=64 # Number of hardware threads per rank (Polaris has 64 CPU logical cores per node)
    CPU_BIND="depth"
fi


# NDEPTH=64 # Number of hardware threads per rank (Polaris has 64 CPU cores per node)
# CPU_BIND="depth"

# NDEPTH=32 # Number of physical CPU cores per Polaris node
# CPU_BIND="verbose,list:0,8,16,24"

# Remove -d and --cpu-bind altogether
# -d ${NDEPTH}  --cpu-bind "${CPU_BIND}"


#FIXME Should we set --envall, --noenvall, or only pass specific env vars?
set -x  # Print "mpiexec" command with expanded variables
mpiexec --verbose \
    --np $((${LEMA_NUM_NODES} * ${NRANKS_PER_NODE})) \
    -ppn ${NRANKS_PER_NODE} \
    -d ${NDEPTH}  --cpu-bind "${CPU_BIND}" \
    ./scripts/polaris/jobs/multinode_example_worker.sh -m "${TRAINING_MODE}"

echo -e "Finished ${TRAINING_MODE} training on ${LEMA_NUM_NODES} node(s):\n$(cat $PBS_NODEFILE)"
echo "Polaris job is all done!"
