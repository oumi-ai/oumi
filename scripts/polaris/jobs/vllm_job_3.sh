#!/bin/bash

#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/lema-balerion
#PBS -e /eagle/community_ai/jobs/logs/lema-balerion

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

NRANKS=1  # Number of MPI ranks to spawn per node (1 `torchrun` per node)
NDEPTH=64 # Number of hardware threads per rank (Polaris has 64 CPU cores per node)
export POLARIS_GPUS_PER_NODE=4

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

# Set up conda.
module load conda

# Activate the LeMa Conda environment.
conda activate /home/$USER/miniconda3/envs/lema
echo "Conda path:"
echo $CONDA_PREFIX

# NCCL settings:
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/#multi-gpu-multi-node-scale-up
# export NCCL_COLLNET_ENABLE=1
# export NCCL_NET_GDR_READ=1
# export NCCL_NET_GDR_LEVEL=SYS
# export NCCL_CROSS_NIC=1
#export NCCL_SET_STACK_SIZE=1
#export NCCL_DEBUG=WARN # WARN


#export NCCL_NET="aws-ofi-nccl"
#export NCCL_NET="Socket"
#export NCCL_SOCKET_IFNAME="bond"
# export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1
# export NCCL_HOME=/eagle/community_ai/soft/libraries/nccl/2.22.3-1/lib
# export NCCL_INCLUDE_DIR=/eagle/community_ai/soft/libraries/nccl/2.22.3-1/include
# export NCCL_LIB_DIR=/eagle/community_ai/soft/libraries/nccl/2.22.3-1/lib
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$NCCL_HOME:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/eagle/community_ai/soft/libraries/hwloc/2.11.1/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/eagle/community_ai/soft/libraries/aws-ofi-nccl/1.10.0/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/eagle/community_ai/soft/libraries/libfabric/1.22.0/lib:$LD_LIBRARY_PATH
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_MR_CACHE_MONITOR=userfaultfd
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export VLLM_NCCL_SO_PATH=/eagle/community_ai/soft/libraries/aws-ofi-nccl/1.10.0/lib/libnccl-net.so

export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SET_THREAD_NAME=1
export VLLM_TRACE_FUNCTION=1
export VERBOSE=1

export SHARED_DIR=/eagle/community_ai
export HF_HOME="${SHARED_DIR}/.cache/huggingface"

REPO="meta-llama"
MODEL="Meta-Llama-3.1-8B-Instruct"
MODEL_REPO="${REPO}/${MODEL}"
export SNAPSHOT_DIR="${REPO}--${MODEL}"
export SNAPSHOT=$(ls "${HF_HOME}/hub/models--${SNAPSHOT_DIR}/snapshots")
echo "${SNAPSHOT}"

echo "Setting up vLLM inference with ${LEMA_NUM_NODES} node(s)..."

set -x  # Print "mpiexec" command with expanded variables

# # Start worker nodes
mpiexec --verbose \
    --np $LEMA_NUM_NODES \
    --ppn ${NRANKS} \
    --depth ${NDEPTH} \
    --cpu-bind depth \
    ./scripts/polaris/jobs/vllm_worker_3.sh

echo "Polaris job is all done!"
